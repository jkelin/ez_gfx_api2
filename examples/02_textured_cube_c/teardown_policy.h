#ifndef EZ_GFX_TEXTURED_CUBE_TEARDOWN_POLICY_H
#define EZ_GFX_TEXTURED_CUBE_TEARDOWN_POLICY_H

#include "ez_gfx_api.h"

/* Unknown result values retain the host; only library-returned statuses reach normal cleanup. */
static inline int teardown_requires_host_retention(
    EzGfxResult surface,
    EzGfxResult context) {
    switch (surface) {
        case EzGfxResult_Ok:
            return 0;
        case EzGfxResult_TeardownAbandoned:
            return 1;
        case EzGfxResult_InvalidArgument:
        case EzGfxResult_InvalidContext:
        case EzGfxResult_NativeFailure:
        case EzGfxResult_NotReady:
        case EzGfxResult_Unsupported:
        case EzGfxResult_DeviceLost:
        case EzGfxResult_QueueFull:
        case EzGfxResult_Cancelled:
            break;
        default:
            return 1;
    }

    switch (context) {
        case EzGfxResult_Ok:
        case EzGfxResult_InvalidArgument:
        case EzGfxResult_NativeFailure:
        case EzGfxResult_NotReady:
        case EzGfxResult_Unsupported:
        case EzGfxResult_DeviceLost:
        case EzGfxResult_QueueFull:
        case EzGfxResult_Cancelled:
            return 0;
        case EzGfxResult_InvalidContext:
        case EzGfxResult_TeardownAbandoned:
        default:
            return 1;
    }
}

#endif
