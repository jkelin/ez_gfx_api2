// Uses Basis Universal's Apache-2.0 transcoder headers; see vendor/PROVENANCE.txt.
#include "basisu_transcoder.h"
#include <cstdint>
#include <new>

struct EzBasisTranscoder {
    basist::basisu_transcoder value;
};

extern "C" {
uint32_t ez_basis_metadata(const uint8_t* data, uint32_t size) {
    if (!data || !size) {
        return 0;
    }
    basist::basisu_transcoder transcoder;
    basist::basisu_file_info info;
    if (!transcoder.validate_header(data, size) ||
        !transcoder.get_file_info(data, size, info) ||
        transcoder.get_texture_type(data, size) != basist::cBASISTexType2D ||
        transcoder.get_total_images(data, size) != 1) {
        return 0;
    }
    return 1u | (info.m_etc1s ? 2u : 0u) | (info.m_has_alpha_slices ? 4u : 0u) |
        (info.m_srgb ? 8u : 0u);
}

EzBasisTranscoder* ez_basis_create(const uint8_t* data, uint32_t size, uint32_t target) {
    if (!data || !size || size > 64u * 1024u * 1024u) {
        return nullptr;
    }
    basist::basisu_transcoder_init();
    auto* transcoder = new (std::nothrow) EzBasisTranscoder;
    if (!transcoder || !transcoder->value.validate_header(data, size) ||
        transcoder->value.get_texture_type(data, size) != basist::cBASISTexType2D ||
        transcoder->value.get_total_images(data, size) != 1) {
        delete transcoder;
        return nullptr;
    }
    // Validate the complete output budget before native codebook decompression.
    const uint32_t levels = transcoder->value.get_total_image_levels(data, size, 0);
    uint64_t total = 0;
    uint32_t prior_width = 0, prior_height = 0;
    for (uint32_t level = 0; level < levels && level < 32; ++level) {
        uint32_t width = 0, height = 0, blocks = 0;
        if (!transcoder->value.get_image_level_desc(data, size, 0, level, width, height, blocks) ||
            !width || !height || !blocks ||
            (level && (width != (prior_width > 1 ? prior_width / 2 : 1) ||
                       height != (prior_height > 1 ? prior_height / 2 : 1) ||
                       (prior_width == 1 && prior_height == 1)))) {
            delete transcoder;
            return nullptr;
        }
        const uint64_t elements = target == 13 ? uint64_t(width) * height : blocks;
        const uint32_t stride = target == 13 ? 4 : (target == 2 ? 8 : 16);
        if (elements > (64u * 1024u * 1024u) / stride) {
            delete transcoder;
            return nullptr;
        }
        total += elements * stride;
        if (total > 64u * 1024u * 1024u) {
            delete transcoder;
            return nullptr;
        }
        prior_width = width;
        prior_height = height;
    }
    if (!levels || levels > 32 || !transcoder->value.start_transcoding(data, size)) {
        delete transcoder;
        return nullptr;
    }
    return transcoder;
}

void ez_basis_destroy(EzBasisTranscoder* transcoder) {
    delete transcoder;
}

uint32_t ez_basis_level_count(const EzBasisTranscoder* transcoder, const uint8_t* data, uint32_t size) {
    return transcoder ? transcoder->value.get_total_image_levels(data, size, 0) : 0;
}

bool ez_basis_level_description(
    const EzBasisTranscoder* transcoder,
    const uint8_t* data,
    uint32_t size,
    uint32_t level,
    uint32_t* width,
    uint32_t* height,
    uint32_t* blocks) {
    return transcoder && width && height && blocks &&
        transcoder->value.get_image_level_desc(data, size, 0, level, *width, *height, *blocks);
}

bool ez_basis_transcode_level(
    const EzBasisTranscoder* transcoder,
    const uint8_t* data,
    uint32_t size,
    uint32_t level,
    uint32_t target,
    uint8_t* output,
    uint32_t output_elements) {
    if (!transcoder || !output || target >= static_cast<uint32_t>(basist::transcoder_texture_format::cTFTotalTextureFormats)) {
        return false;
    }
    return transcoder->value.transcode_image_level(
        data,
        size,
        0,
        level,
        output,
        output_elements,
        static_cast<basist::transcoder_texture_format>(target));
}
}
