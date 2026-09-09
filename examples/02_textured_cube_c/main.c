#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_WAYLAND
#endif
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "ez_gfx_api.h"
#include "teardown_policy.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if EZ_GFX_ABI_VERSION != 37u
#error "textured_cube requires ez-gfx ABI v37"
#endif

#define WIDTH 640u
#define HEIGHT 480u
#define MAX_ARTIFACT_BYTES (16u * 1024u * 1024u)

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
    int hidden;
} Options;
typedef struct Observations {
    uint8_t *snapshot;
    size_t snapshot_capacity;
    size_t snapshot_size;
    int failed;
} Observations;

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

/* File opening maps the secure MSVC API and portable C API to one checked result. */
static int open_file(FILE **file, const char *path, const char *mode) {
#ifdef _MSC_VER
    return fopen_s(file, path, mode) == 0 && *file != NULL;
#else
    *file = fopen(path, mode);
    return *file != NULL;
#endif
}

/* Every status-bearing ABI call uses the library's stable diagnostic text. */
static int checked(EzGfxResult result, const char *operation) {
    char message[64];
    size_t required = 0;

    if (result == EzGfxResult_Ok) return 1;
    if (ez_gfx_error_print(result, message, sizeof(message), &required) != EzGfxResult_Ok) {
        snprintf(message, sizeof(message), "unknown error");
    }
    fprintf(stderr, "%s: %s (%u)\n", operation, message, (unsigned)result);
    return 0;
}


/* Usage is emitted for missing values, unknown switches, and unsupported backends. */
static void usage(const char *program) {
    fprintf(stderr,
        "usage: %s --backend vulkan"
#ifdef _WIN32
        "|dx12"
#endif
        " [--artifact PATH] [--max-frames 1..10000] [--snapshot PATH] [--hidden]\n",
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
#ifdef _WIN32
            } else if (strcmp(name, "dx12") == 0) {
                options->backend = EzGfxBackend_Dx12;
                options->backend_name = "dx12";
#endif
            } else return 0;
        } else if (strcmp(argv[index], "--artifact") == 0 && index + 1 < argc) {
            options->artifact_path = argv[++index];
        } else if (strcmp(argv[index], "--snapshot") == 0 && index + 1 < argc) {
            options->snapshot_path = argv[++index];
        } else if (strcmp(argv[index], "--max-frames") == 0 && index + 1 < argc) {
            if (!parse_u32(argv[++index], &options->max_frames)) return 0;
        } else if (strcmp(argv[index], "--hidden") == 0) {
            options->hidden = 1;
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
    if (!open_file(&file, path, "rb")) {
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
static int write_snapshot(
    const char *path, const uint8_t *bytes, size_t size, uint32_t width, uint32_t height) {
    FILE *file = NULL;
    size_t index, written;
    int close_result, nonzero = 0;
    if (size != (size_t)width * (size_t)height * 4u) {
        fprintf(stderr, "snapshot has unexpected size: %zu\n", size);
        return 0;
    }
    for (index = 0; index < size; ++index) nonzero |= bytes[index] != 0;
    if (!nonzero) {
        fprintf(stderr, "snapshot is empty\n");
        return 0;
    }
    if (!open_file(&file, path, "wb")) {
        fprintf(stderr, "open snapshot failed: %s\n", path);
        return 0;
    }
    written = fwrite(bytes, 1, size, file);
    close_result = fclose(file);
    if (written != size || close_result != 0) {
        fprintf(stderr, "write snapshot failed: %s\n", path);
        return 0;
    }
    printf("snapshot %s: %zu RGBA bytes (%ux%u)\n", path, size, width, height);
    return 1;
}

/* Callback payloads are borrowed; readback bytes are copied before returning. */
static void observe(const EzGfxEvent *event, void *user_data) {
    Observations *observations = (Observations *)user_data;
    if (event == NULL || observations == NULL) return;

    switch (event->kind) {
    case EzGfxEventKind_Upload:
        if (event->upload.status == EzGfxUploadStatus_Failed) observations->failed = 1;
        break;
    case EzGfxEventKind_Runtime:
        printf("runtime event %llu backend=%u phase=%u status=%u\n",
            (unsigned long long)event->record.correlation_id, (unsigned)event->record.backend,
            (unsigned)event->record.phase, (unsigned)event->record.status);
        if (event->record.status != EzGfxResult_Ok) observations->failed = 1;
        break;
    case EzGfxEventKind_Diagnostic:
        fprintf(stderr, "diagnostic level=%u phase=%u status=%u\n",
            (unsigned)event->level, (unsigned)event->record.phase,
            (unsigned)event->record.status);
        if (event->level == EzGfxDiagnosticLevel_Error ||
            event->record.status != EzGfxResult_Ok) observations->failed = 1;
        break;
    case EzGfxEventKind_ObservationsDropped:
        fprintf(stderr, "event queue dropped %llu records\n",
            (unsigned long long)event->dropped);
        observations->failed = 1;
        break;
    case EzGfxEventKind_Readback:
    case EzGfxEventKind_Snapshot:
        if (event->readback_byte_count != observations->snapshot_capacity ||
            event->readback_bytes == NULL || observations->snapshot == NULL) {
            observations->failed = 1;
            break;
        }
        memcpy(observations->snapshot, event->readback_bytes, event->readback_byte_count);
        observations->snapshot_size = event->readback_byte_count;
        break;
    default:
        observations->failed = 1;
        break;
    }
}

/* GLFW may report initialization and platform errors before a window exists. */
static void glfw_error(int code, const char *description) {
    fprintf(stderr, "GLFW error %d: %s\n", code, description != NULL ? description : "unknown");
}

static GLFWwindow *create_window(int hidden) {
    glfwDefaultWindowHints();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, hidden ? GLFW_FALSE : GLFW_TRUE);
    return glfwCreateWindow((int)WIDTH, (int)HEIGHT, "ez-gfx textured cube", NULL, NULL);
}

static int framebuffer_extent(GLFWwindow *window, uint32_t *width, uint32_t *height) {
    int native_width = 0, native_height = 0;
    if (window == NULL || width == NULL || height == NULL) return 0;
    glfwGetFramebufferSize(window, &native_width, &native_height);
    if (native_width < 0 || native_height < 0) return 0;
    *width = (uint32_t)native_width;
    *height = (uint32_t)native_height;
    return 1;
}

static int native_surface_desc(GLFWwindow *window, int snapshot, EzGfxWindowSurfaceDesc *desc) {
    int platform;
    if (window == NULL || desc == NULL) return 0;
    memset(desc, 0, sizeof(*desc));
    desc->cache_presented_snapshots = (uint8_t)(snapshot ? 1 : 0);
    platform = glfwGetPlatform();
    switch (platform) {
#ifdef _WIN32
    case GLFW_PLATFORM_WIN32: {
        HWND hwnd = glfwGetWin32Window(window);
        LONG_PTR module = (hwnd == NULL) ? 0 : GetWindowLongPtrW(hwnd, GWLP_HINSTANCE);
        if (hwnd == NULL || module == 0) return 0;
        desc->system = EzGfxNativeWindowSystem_Win32;
        desc->handle_a = (uint64_t)(uintptr_t)hwnd;
        desc->handle_b = (uint64_t)(uintptr_t)module;
        return 1;
    }
#endif
#ifdef __linux__
    case GLFW_PLATFORM_X11: {
        Display *display = glfwGetX11Display();
        Window xid = glfwGetX11Window(window);
        if (display == NULL || xid == 0) return 0;
        desc->system = EzGfxNativeWindowSystem_Xlib;
        desc->handle_a = (uint64_t)(uintptr_t)display;
        desc->handle_b = (uint64_t)xid;
        return 1;
    }
    case GLFW_PLATFORM_WAYLAND:
        desc->system = EzGfxNativeWindowSystem_Wayland;
        desc->handle_a = (uint64_t)(uintptr_t)glfwGetWaylandDisplay();
        desc->handle_b = (uint64_t)(uintptr_t)glfwGetWaylandWindow(window);
        return desc->handle_a != 0 && desc->handle_b != 0;
#endif
    default:
        fprintf(stderr, "unsupported GLFW platform %d for native Vulkan surfaces\n", platform);
        return 0;
    }
}

/* Every partially initialized path joins reverse-order cleanup through one exit. */
int main(int argc, char **argv) {
    Options options;
    GLFWwindow *window = NULL;
    uint32_t surface_width = 0, surface_height = 0;
    uint8_t *artifact = NULL;
    size_t artifact_size = 0;
    uint8_t *snapshot = NULL;
    size_t snapshot_size = 0;
    EzGfxContext context = 0;
    EzGfxSurface surface = 0;
    EzGfxShader shader = 0;
    EzGfxBuffer primitives = 0;
    EzGfxVertexHeap positions_heap = 0, normals_heap = 0;
    EzGfxVertexAllocation positions = 0, normals = 0;
    EzGfxIndexAllocation indices = 0;
    EzGfxCounterBuffer indirect = 0;
    EzGfxFrame active_frame = 0;
    uint32_t first_index = 0, index_count = 0, frame_index = 0;
    int glfw_initialized = 0, success = 0, retain_host = 0;
    EzGfxBackendContextDesc context_desc;
    EzGfxWindowSurfaceDesc surface_desc;
    Primitive primitive;
    EzGfxBinding bindings[2];
    EzGfxDynamicState dynamic_state;
    EzGfxResult frame_result;
    Observations observations = {0};

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
    glfwSetErrorCallback(glfw_error);
    if (!glfwInit()) goto cleanup;
    glfw_initialized = 1;
    window = create_window(options.hidden);
    if (window == NULL || !framebuffer_extent(window, &surface_width, &surface_height) ||
        surface_width == 0 || surface_height == 0) goto cleanup;
    if (options.snapshot_path != NULL) {
        snapshot_size = (size_t)surface_width * (size_t)surface_height * 4u;
        snapshot = (uint8_t *)malloc(snapshot_size);
        if (snapshot == NULL) {
            fprintf(stderr, "allocate snapshot failed\n");
            goto cleanup;
        }
        observations.snapshot = snapshot;
        observations.snapshot_capacity = snapshot_size;
    }

    context_desc = (EzGfxBackendContextDesc){0, 0, options.backend, 0, 0, NULL};
    if (!checked(ez_gfx_context_create_backend(&context_desc, &context), "create context")) goto cleanup;
    if (!checked(ez_gfx_context_register_callback(context, observe, &observations), "register callback")) goto cleanup;
    if (!native_surface_desc(window, options.snapshot_path != NULL, &surface_desc)) goto cleanup;
    if (!checked(ez_gfx_surface_create_window(context, &surface_desc, &surface), "create surface")) goto cleanup;
    if (!checked(ez_gfx_context_init_device(context, surface), "initialize surface device")) goto cleanup;
    /* Publishes the host extent for Vulkan window systems whose native extent is undefined. */
    if (!checked(ez_gfx_surface_resize(
            context, surface, surface_width, surface_height), "publish initial surface extent")) {
        goto cleanup;
    }
    if (!checked(ez_gfx_index_allocation_create(context, INDICES, 36, &indices), "upload indices")) goto cleanup;
    if (!checked(ez_gfx_index_allocation_get_range(context, indices, &first_index, &index_count), "query index allocation")) goto cleanup;
    if (!checked(ez_gfx_vertex_heap_create(context, "positions", sizeof("positions") - 1, sizeof(Vec4), &positions_heap), "create positions heap")) goto cleanup;
    if (!checked(ez_gfx_vertex_heap_upload(context, positions_heap, POSITIONS, 24, sizeof(Vec4), &positions), "upload positions")) goto cleanup;
    if (!checked(ez_gfx_vertex_heap_create(context, "normals", sizeof("normals") - 1, sizeof(Vec4), &normals_heap), "create normals heap")) goto cleanup;
    if (!checked(ez_gfx_vertex_heap_upload(context, normals_heap, NORMALS, 24, sizeof(Vec4), &normals), "upload normals")) goto cleanup;

    primitive = (Primitive){first_index, index_count, 0, 0};
    if (!checked(ez_gfx_shader_load_artifact(context, artifact, artifact_size, &shader), "load shader artifact")) goto cleanup;

    dynamic_state = (EzGfxDynamicState){EzGfxCullMode_None, EzGfxFrontFace_CounterClockwise,
        EzGfxPrimitiveType_TriangleList, EzGfxBlendMode_None};

    for (frame_index = 0; frame_index < options.max_frames; ++frame_index) {
        uint32_t framebuffer_width = 0, framebuffer_height = 0;
        glfwPollEvents();
        if (glfwWindowShouldClose(window)) break;
        if (!framebuffer_extent(window, &framebuffer_width, &framebuffer_height)) goto cleanup;
        if (framebuffer_width == 0 || framebuffer_height == 0) continue;
        if (options.snapshot_path != NULL &&
            (framebuffer_width != surface_width || framebuffer_height != surface_height)) {
            fprintf(stderr, "snapshot framebuffer changed from %ux%u\n",
                surface_width, surface_height);
            goto cleanup;
        }
        if (framebuffer_width != surface_width || framebuffer_height != surface_height) {
            if (!checked(ez_gfx_surface_resize(
                    context, surface, framebuffer_width, framebuffer_height), "resize surface")) {
                goto cleanup;
            }
            surface_width = framebuffer_width;
            surface_height = framebuffer_height;
        }
        /* Buffers are one-frame values: the first frame execution using each binding claims them. */
        if (!checked(ez_gfx_value_buffer_acquire(context, &primitive, sizeof(primitive), "primitives", sizeof("primitives") - 1, &primitives), "acquire primitives")) goto cleanup;
        if (!checked(ez_gfx_counter_buffer_acquire(context, sizeof(EzGfxDrawIndexedCommand), 1, "draw commands", sizeof("draw commands") - 1, &indirect), "acquire counter")) goto cleanup;
        /* Compute writes both the draw command and its GPU-produced visible count. */
        if (!checked(ez_gfx_frame_begin(context, surface, &active_frame), "begin frame")) goto cleanup;
        bindings[0] = (EzGfxBinding){"primitives", sizeof("primitives") - 1, primitives, 0, 0};
        bindings[1] = (EzGfxBinding){"draw_commands", sizeof("draw_commands") - 1, 0, indirect, 0};
        if (!checked(ez_gfx_frame_bind(context, active_frame, &bindings[0]), "bind primitives") ||
            !checked(ez_gfx_frame_bind(context, active_frame, &bindings[1]), "bind draw commands") ||
            !checked(ez_gfx_frame_execute_compute(context, active_frame, shader, 1, 1, 1), "execute compute")) {
            (void)ez_gfx_frame_abort(context, active_frame);
            active_frame = 0;
            /* Abort consumes claimed handles; release covers any handle not claimed. */
            ez_gfx_buffer_release(context, primitives);
            primitives = 0;
            ez_gfx_counter_buffer_release(context, indirect);
            indirect = 0;
            goto cleanup;
        }
        if (!checked(ez_gfx_frame_execute_graphics(context, active_frame, shader, indirect, &dynamic_state), "execute graphics")) {
            (void)ez_gfx_frame_abort(context, active_frame);
            active_frame = 0;
            /* Abort consumes claimed handles; release covers any handle not claimed. */
            ez_gfx_buffer_release(context, primitives);
            primitives = 0;
            ez_gfx_counter_buffer_release(context, indirect);
            indirect = 0;
            goto cleanup;
        }
        frame_result = ez_gfx_frame_end(context, active_frame);
        active_frame = 0;
        /* The terminal frame consumes both buffers. */
        primitives = 0;
        indirect = 0;
        if (!checked(frame_result, "submit and present")) goto cleanup;
        if (observations.failed) goto cleanup;
    }

    if (options.snapshot_path != NULL) {
        if (observations.snapshot_size != snapshot_size) {
            fprintf(stderr, "snapshot callback was not delivered\n");
            goto cleanup;
        }
        if (!write_snapshot(
                options.snapshot_path, snapshot, snapshot_size, surface_width, surface_height)) {
            goto cleanup;
        }
    }
    printf("rendered %u frame(s) with %s\n", frame_index, options.backend_name);
    success = 1;

cleanup:
    if (active_frame != 0) (void)ez_gfx_frame_abort(context, active_frame);
    if (context != 0 && primitives != 0) ez_gfx_buffer_release(context, primitives);
    if (context != 0 && indirect != 0) ez_gfx_counter_buffer_release(context, indirect);
    if (context != 0) {
        EzGfxResult surface_teardown = EzGfxResult_Ok;
        EzGfxResult context_teardown;
        (void)ez_gfx_context_register_callback(context, NULL, NULL);
        if (surface != 0) {
            surface_teardown = ez_gfx_surface_destroy(context, surface);
            surface = 0;
            (void)checked(surface_teardown, "destroy surface");
        }
        context_teardown = ez_gfx_context_destroy(context);
        context = 0;
        (void)checked(context_teardown, "destroy context");
        retain_host = teardown_requires_host_retention(surface_teardown, context_teardown);
        if (surface_teardown != EzGfxResult_Ok || context_teardown != EzGfxResult_Ok) {
            success = 0;
        }
    }
    if (!retain_host) {
        if (window != NULL) glfwDestroyWindow(window);
        if (glfw_initialized) glfwTerminate();
    }
    free(snapshot);
    free(artifact);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
