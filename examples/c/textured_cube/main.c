#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "ez_gfx_api.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if EZ_GFX_ABI_VERSION != 30u
#error "textured_cube requires ez-gfx ABI v30"
#endif

#define WIDTH 640u
#define HEIGHT 480u
#define MAX_ARTIFACT_BYTES (16u * 1024u * 1024u)
#define MAX_OBSERVATIONS 4096u

typedef struct Vec4 { float x, y, z, w; } Vec4;
typedef struct Primitive {
    uint32_t first_index;
    uint32_t index_count;
    uint32_t vertex_offset;
    uint32_t normal_offset;
} Primitive;
typedef struct Options {
    EzGfxBackend backend;
    const char *backend_name;
    const char *artifact_path;
    const char *snapshot_path;
    uint32_t max_frames;
} Options;

_Static_assert(sizeof(Vec4) == 16, "Vec4 must match Slang float4");
_Static_assert(sizeof(Primitive) == 16, "Primitive must match the shader record");

static const Vec4 POSITIONS[24] = {
    {-1, -1, 1, 1}, {1, -1, 1, 1}, {1, 1, 1, 1}, {-1, 1, 1, 1},
    {1, -1, -1, 1}, {-1, -1, -1, 1}, {-1, 1, -1, 1}, {1, 1, -1, 1},
    {-1, -1, -1, 1}, {-1, -1, 1, 1}, {-1, 1, 1, 1}, {-1, 1, -1, 1},
    {1, -1, 1, 1}, {1, -1, -1, 1}, {1, 1, -1, 1}, {1, 1, 1, 1},
    {-1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, -1, 1}, {-1, 1, -1, 1},
    {-1, -1, -1, 1}, {1, -1, -1, 1}, {1, -1, 1, 1}, {-1, -1, 1, 1},
};
static const Vec4 NORMALS[24] = {
    {0, 0, 1, 0}, {0, 0, 1, 0}, {0, 0, 1, 0}, {0, 0, 1, 0},
    {0, 0, -1, 0}, {0, 0, -1, 0}, {0, 0, -1, 0}, {0, 0, -1, 0},
    {-1, 0, 0, 0}, {-1, 0, 0, 0}, {-1, 0, 0, 0}, {-1, 0, 0, 0},
    {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0},
    {0, 1, 0, 0}, {0, 1, 0, 0}, {0, 1, 0, 0}, {0, 1, 0, 0},
    {0, -1, 0, 0}, {0, -1, 0, 0}, {0, -1, 0, 0}, {0, -1, 0, 0},
};
static const uint32_t INDICES[36] = {
    0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7,
    8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15,
    16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23,
};

static int g_running = 1;


/* Every status-bearing ABI call uses the library's stable diagnostic text. */
static int checked(EzGfxResult result, const char *operation) {
    char message[64];
    size_t required = 0;

    if (result == EzGfxResult_Ok) return 1;
    if (ez_gfx_print_error(result, message, sizeof(message), &required) != EzGfxResult_Ok) {
        snprintf(message, sizeof(message), "unknown error");
    }
    fprintf(stderr, "%s: %s (%u)\n", operation, message, (unsigned)result);
    return 0;
}

/* Usage is emitted for missing values, unknown switches, and unsupported backends. */
static void usage(const char *program) {
    fprintf(stderr,
        "usage: %s --backend vulkan|dx12 [--artifact PATH] [--max-frames 1..10000] [--snapshot PATH]\n",
        program);
}

/* Frame limits reject zero, overflow, trailing text, and values above the work bound. */
static int parse_u32(const char *text, uint32_t *value) {
    char *end = NULL;
    unsigned long parsed;
    errno = 0;
    parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 || parsed > 10000) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

/* Omitted artifact/frame options are bounded defaults; a backend is always mandatory. */
static int parse_options(int argc, char **argv, Options *options) {
    int index;
    memset(options, 0, sizeof(*options));
    options->artifact_path = "textured_cube.ezgfxshader";
    options->max_frames = 300;

    for (index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--backend") == 0 && index + 1 < argc) {
            const char *name = argv[++index];
            if (strcmp(name, "vulkan") == 0) {
                options->backend = EzGfxBackend_Vulkan;
                options->backend_name = "vulkan";
            } else if (strcmp(name, "dx12") == 0) {
                options->backend = EzGfxBackend_Dx12;
                options->backend_name = "dx12";
            } else return 0;
        } else if (strcmp(argv[index], "--artifact") == 0 && index + 1 < argc) {
            options->artifact_path = argv[++index];
        } else if (strcmp(argv[index], "--snapshot") == 0 && index + 1 < argc) {
            options->snapshot_path = argv[++index];
        } else if (strcmp(argv[index], "--max-frames") == 0 && index + 1 < argc) {
            if (!parse_u32(argv[++index], &options->max_frames)) return 0;
        } else return 0;
    }
    return options->backend_name != NULL;
}

/* Empty, oversized, and partially read artifacts fail before crossing the ABI boundary. */
static int read_file(const char *path, uint8_t **bytes, size_t *size) {
    FILE *file = NULL;
    long length;
    size_t read_count;
    *bytes = NULL;
    *size = 0;
    if (fopen_s(&file, path, "rb") != 0 || file == NULL) {
        fprintf(stderr, "open artifact %s failed\n", path);
        return 0;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) <= 0 ||
        (unsigned long)length > MAX_ARTIFACT_BYTES || fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "artifact size is invalid: %s\n", path);
        fclose(file);
        return 0;
    }
    *bytes = (uint8_t *)malloc((size_t)length);
    if (*bytes == NULL) {
        fprintf(stderr, "allocate artifact buffer failed\n");
        fclose(file);
        return 0;
    }
    read_count = fread(*bytes, 1, (size_t)length, file);
    if (fclose(file) != 0 || read_count != (size_t)length) {
        fprintf(stderr, "read artifact failed: %s\n", path);
        free(*bytes);
        *bytes = NULL;
        return 0;
    }
    *size = (size_t)length;
    return 1;
}

/* Capture rejects wrong dimensions and all-zero output before creating a file. */
static int write_snapshot(const char *path, const uint8_t *bytes, size_t size) {
    FILE *file = NULL;
    size_t index, written;
    int close_result, nonzero = 0;
    if (size != (size_t)WIDTH * (size_t)HEIGHT * 4u) {
        fprintf(stderr, "snapshot has unexpected size: %zu\n", size);
        return 0;
    }
    for (index = 0; index < size; ++index) nonzero |= bytes[index] != 0;
    if (!nonzero) {
        fprintf(stderr, "snapshot is empty\n");
        return 0;
    }
    if (fopen_s(&file, path, "wb") != 0 || file == NULL) {
        fprintf(stderr, "open snapshot failed: %s\n", path);
        return 0;
    }
    written = fwrite(bytes, 1, size, file);
    close_result = fclose(file);
    if (written != size || close_result != 0) {
        fprintf(stderr, "write snapshot failed: %s\n", path);
        return 0;
    }
    printf("snapshot %s: %zu RGBA bytes (%ux%u)\n", path, size, WIDTH, HEIGHT);
    return 1;
}

/* Polling is bounded; dropped, failed, and error-level observations fail the run. */
static int poll_observability(EzGfxContext context) {
    uint32_t count;
    uint64_t dropped;
    uint8_t present;
    EzGfxRuntimeRecord record;
    EzGfxDiagnostic diagnostic;

    for (count = 0; count < MAX_OBSERVATIONS; ++count) {
        dropped = 0;
        present = 0;
        if (!checked(ez_gfx_poll_runtime_event(&record, &present, &dropped, context),
                "poll runtime event")) return 0;
        if (dropped != 0) {
            fprintf(stderr, "runtime event queue dropped %llu records\n", (unsigned long long)dropped);
            return 0;
        }
        if (!present) break;
        printf("runtime event %llu backend=%u phase=%u status=%u\n",
            (unsigned long long)record.correlation_id, (unsigned)record.backend,
            (unsigned)record.phase, (unsigned)record.status);
        if (record.status != EzGfxResult_Ok) return 0;
    }
    if (count == MAX_OBSERVATIONS) {
        fprintf(stderr, "runtime event drain exceeded its bound\n");
        return 0;
    }

    for (count = 0; count < MAX_OBSERVATIONS; ++count) {
        dropped = 0;
        present = 0;
        if (!checked(ez_gfx_poll_diagnostic(&diagnostic, &present, &dropped, context),
                "poll diagnostic")) return 0;
        if (dropped != 0) {
            fprintf(stderr, "diagnostic queue dropped %llu records\n", (unsigned long long)dropped);
            return 0;
        }
        if (!present) break;
        fprintf(stderr, "diagnostic level=%u phase=%u status=%u\n",
            (unsigned)diagnostic.level, (unsigned)diagnostic.record.phase,
            (unsigned)diagnostic.record.status);
        if (diagnostic.level == EzGfxDiagnosticLevel_Error ||
            diagnostic.record.status != EzGfxResult_Ok) return 0;
    }
    if (count == MAX_OBSERVATIONS) {
        fprintf(stderr, "diagnostic drain exceeded its bound\n");
        return 0;
    }
    return 1;
}

/* Closing during a bounded run stops future frame submission. */
static LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
    if (message == WM_CLOSE) {
        DestroyWindow(window);
        return 0;
    }
    if (message == WM_DESTROY) {
        g_running = 0;
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcA(window, message, wparam, lparam);
}

/* Partial Win32 creation unregisters its class before returning failure. */
static HWND create_window(HINSTANCE instance) {
    const char *class_name = "EzGfxTexturedCubeWindow";
    WNDCLASSA window_class;
    RECT bounds = {0, 0, (LONG)WIDTH, (LONG)HEIGHT};
    HWND window;
    memset(&window_class, 0, sizeof(window_class));
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = instance;
    window_class.lpszClassName = class_name;
    window_class.hCursor = LoadCursorA(NULL, IDC_ARROW);
    if (RegisterClassA(&window_class) == 0) {
        fprintf(stderr, "RegisterClassA failed: %lu\n", GetLastError());
        return NULL;
    }
    if (!AdjustWindowRect(&bounds, WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU, FALSE)) {
        fprintf(stderr, "AdjustWindowRect failed: %lu\n", GetLastError());
        UnregisterClassA(class_name, instance);
        return NULL;
    }
    window = CreateWindowExA(0, class_name, "ez-gfx textured cube",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU, CW_USEDEFAULT, CW_USEDEFAULT,
        bounds.right - bounds.left, bounds.bottom - bounds.top, NULL, NULL, instance, NULL);
    if (window == NULL) {
        fprintf(stderr, "CreateWindowExA failed: %lu\n", GetLastError());
        UnregisterClassA(class_name, instance);
        return NULL;
    }
    ShowWindow(window, SW_SHOW);
    UpdateWindow(window);
    return window;
}

/* Every partially initialized path joins reverse-order cleanup through one exit. */
int main(int argc, char **argv) {
    const char *class_name = "EzGfxTexturedCubeWindow";
    Options options;
    HINSTANCE instance = GetModuleHandleA(NULL);
    HWND window = NULL;
    MSG message;
    uint8_t *artifact = NULL;
    size_t artifact_size = 0;
    uint8_t *snapshot = NULL;
    size_t snapshot_size = 0;
    EzGfxContext context = 0;
    EzGfxSurface surface = 0;
    EzGfxShader shader = 0;
    EzGfxStructuredBuffer primitives = 0;
    EzGfxVertexHeap positions_heap = 0, normals_heap = 0;
    EzGfxVertexAllocation positions = 0, normals = 0;
    EzGfxIndexAllocation indices = 0;
    EzGfxIndirectBuffer indirect = 0;
    uint32_t first_index = 0, index_count = 0, frame;
    int success = 0;
    EzGfxBackendContextDesc context_desc;
    EzGfxSurfaceDesc surface_desc;
    Primitive primitive;
    EzGfxBinding bindings[2];
    EzGfxDynamicState dynamic_state;
    EzGfxResult frame_result;

    if (ez_gfx_abi_version() != EZ_GFX_ABI_VERSION) {
        fprintf(stderr, "ez-gfx ABI mismatch: header=%u library=%u\n",
            EZ_GFX_ABI_VERSION, ez_gfx_abi_version());
        return EXIT_FAILURE;
    }
    if (!parse_options(argc, argv, &options)) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (!read_file(options.artifact_path, &artifact, &artifact_size)) goto cleanup;
    window = create_window(instance);
    if (window == NULL) goto cleanup;

    context_desc = (EzGfxBackendContextDesc){0, 0, EzGfxSurfacePlatform_Win32, options.backend, 0};
    if (!checked(ez_gfx_context_create_backend(&context_desc, &context), "create context")) goto cleanup;
    surface_desc = (EzGfxSurfaceDesc){window, instance, EzGfxSurfacePlatform_Win32,
        WIDTH, HEIGHT, options.snapshot_path != NULL};
    if (!checked(ez_gfx_surface_create(&surface_desc, &surface, context), "create surface")) goto cleanup;
    if (!checked(ez_gfx_context_init_device(surface, context), "initialize surface device")) goto cleanup;
    if (!checked(ez_gfx_surface_resize(surface, WIDTH, HEIGHT, context), "resize surface")) goto cleanup;
    if (!checked(ez_gfx_index_heap_create(sizeof(INDICES), "cube indices", sizeof("cube indices") - 1, context), "create index heap")) goto cleanup;
    if (!checked(ez_gfx_vertex_upload_indices(INDICES, 36, &indices, context), "upload indices")) goto cleanup;
    if (!checked(ez_gfx_index_allocation_get_range(indices, &first_index, &index_count, context), "query index allocation")) goto cleanup;
    if (!checked(ez_gfx_vertex_heap_create("positions", sizeof("positions") - 1, sizeof(POSITIONS), sizeof(Vec4), &positions_heap, context), "create positions heap")) goto cleanup;
    if (!checked(ez_gfx_vertex_upload(positions_heap, POSITIONS, 24, sizeof(Vec4), &positions, context), "upload positions")) goto cleanup;
    if (!checked(ez_gfx_vertex_heap_create("normals", sizeof("normals") - 1, sizeof(NORMALS), sizeof(Vec4), &normals_heap, context), "create normals heap")) goto cleanup;
    if (!checked(ez_gfx_vertex_upload(normals_heap, NORMALS, 24, sizeof(Vec4), &normals, context), "upload normals")) goto cleanup;

    primitive = (Primitive){first_index, index_count, 0, 0};
    if (!checked(ez_gfx_shader_load_artifact(artifact, artifact_size, &shader, context), "load shader artifact")) goto cleanup;

    dynamic_state = (EzGfxDynamicState){EzGfxCullMode_None, EzGfxFrontFace_CounterClockwise,
        EzGfxPrimitiveType_TriangleList, EzGfxBlendMode_None};

    for (frame = 0; frame < options.max_frames && g_running; ++frame) {
        while (PeekMessageA(&message, NULL, 0, 0, PM_REMOVE)) {
            if (message.message == WM_QUIT) { g_running = 0; break; }
            TranslateMessage(&message);
            DispatchMessageA(&message);
        }
        if (!g_running) break;
        if (!checked(ez_gfx_begin_render(surface, context), "begin render")) goto cleanup;
        if (!checked(ez_gfx_structured_acquire(sizeof(Primitive), 1, "primitives", sizeof("primitives") - 1, &primitives, context), "acquire primitives")) goto cleanup;
        if (!checked(ez_gfx_structured_write(primitives, 0, &primitive, 1, sizeof(Primitive), context), "write primitives")) goto cleanup;
        if (!checked(ez_gfx_acquire_indirect(1, "draw commands", sizeof("draw commands") - 1, &indirect, context), "acquire indirect")) goto cleanup;
        if (!checked(ez_gfx_indirect_publish_compute_count(indirect, 1, context), "publish compute indirect count")) goto cleanup;
        bindings[0] = (EzGfxBinding){"primitives", sizeof("primitives") - 1, primitives, 0, 0};
        bindings[1] = (EzGfxBinding){"draw_commands", sizeof("draw_commands") - 1, 0, indirect, 0};
        if (!checked(ez_gfx_render_add_compute_pipeline(shader, 1, 1, 1, bindings, 2,
                NULL, 0, context), "record compute")) goto cleanup;
        if (!checked(ez_gfx_render_add_vertex_pipeline(shader, indirect, bindings, 2,
                &dynamic_state, NULL, 0, context), "record graphics")) goto cleanup;
        frame_result = ez_gfx_finish_render(context);
        if (!checked(frame_result, "submit and present")) {
            (void)poll_observability(context);
            goto cleanup;
        }
        primitives = 0;
        indirect = 0;
        if (!poll_observability(context)) goto cleanup;
    }
    if (frame == 0) {
        fprintf(stderr, "no frame was presented\n");
        goto cleanup;
    }

    if (options.snapshot_path != NULL) {
        if (!checked(ez_gfx_frame_readback(NULL, 0, &snapshot_size, context), "query snapshot size")) goto cleanup;
        snapshot = (uint8_t *)malloc(snapshot_size);
        if (snapshot == NULL) {
            fprintf(stderr, "allocate snapshot failed\n");
            goto cleanup;
        }
        if (!checked(ez_gfx_frame_readback(snapshot, snapshot_size, &snapshot_size, context), "read snapshot")) goto cleanup;
        if (!write_snapshot(options.snapshot_path, snapshot, snapshot_size)) goto cleanup;
    }
    printf("rendered %u frame(s) with %s\n", frame, options.backend_name);
    success = 1;

cleanup:
    if (context != 0) ez_gfx_context_destroy(context);
    if (window != NULL && IsWindow(window)) DestroyWindow(window);
    if (window != NULL) UnregisterClassA(class_name, instance);
    free(snapshot);
    free(artifact);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
