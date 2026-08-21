#pragma once

#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/MemoryLayoutConfig.h"

namespace rtp_llm {

struct BlockPoolConfig {
    std::string pool_name = "unnamed";

    // all memory layouts share the same block id space
    uint32_t block_num = 0;

    size_t total_size_bytes = 0;

    std::vector<MemoryLayoutConfig> memory_layouts;

    // global_layer_id -> {layout_idx, local_layer_id}; empty = fall back to the
    // contiguous cursor-built mapping (no effect on legacy paths such as MTP). Used for
    // the dual-layout case with interleaved GA/SWA layers (MiMo V2.5), where the
    // interleaved layer ids cannot be expressed as contiguous ranges.
    std::vector<std::pair<int, int>> explicit_layer_mapping;
};

}  // namespace rtp_llm
