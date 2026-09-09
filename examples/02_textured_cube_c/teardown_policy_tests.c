#include "teardown_policy.h"

#include <stddef.h>

static int verify_pairs(
    const EzGfxResult *surface_results,
    size_t surface_count,
    const EzGfxResult *context_results,
    size_t context_count,
    int expected) {
    size_t surface_index;
    size_t context_index;

    for (surface_index = 0; surface_index < surface_count; ++surface_index) {
        for (context_index = 0; context_index < context_count; ++context_index) {
            if (teardown_requires_host_retention(
                    surface_results[surface_index], context_results[context_index]) != expected) {
                return 0;
            }
        }
    }
    return 1;
}

int main(void) {
    static const EzGfxResult surface_ok[] = {EzGfxResult_Ok};
    static const EzGfxResult surface_abandoned[] = {EzGfxResult_TeardownAbandoned};
    static const EzGfxResult surface_preterminal[] = {
        EzGfxResult_InvalidArgument,
        EzGfxResult_InvalidContext,
        EzGfxResult_NativeFailure,
        EzGfxResult_NotReady,
        EzGfxResult_Unsupported,
        EzGfxResult_DeviceLost,
        EzGfxResult_QueueFull,
        EzGfxResult_Cancelled,
    };
    static const EzGfxResult context_terminal[] = {
        EzGfxResult_Ok,
        EzGfxResult_InvalidArgument,
        EzGfxResult_NativeFailure,
        EzGfxResult_NotReady,
        EzGfxResult_Unsupported,
        EzGfxResult_DeviceLost,
        EzGfxResult_QueueFull,
        EzGfxResult_Cancelled,
    };
    static const EzGfxResult context_unproven[] = {
        EzGfxResult_InvalidContext,
        EzGfxResult_TeardownAbandoned,
    };

    if (!verify_pairs(surface_ok, 1, context_terminal, 8, 0)) return 1;
    if (!verify_pairs(surface_ok, 1, context_unproven, 2, 0)) return 2;
    if (!verify_pairs(surface_abandoned, 1, context_terminal, 8, 1)) return 3;
    if (!verify_pairs(surface_abandoned, 1, context_unproven, 2, 1)) return 4;
    if (!verify_pairs(surface_preterminal, 8, context_terminal, 8, 0)) return 5;
    if (!verify_pairs(surface_preterminal, 8, context_unproven, 2, 1)) return 6;
    return 0;
}
