#pragma once

#include <vector>

#include <torch/extension.h>
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

struct BlockBufferPtrInfo {
    torch::Tensor kv_addr;
    torch::Tensor kv_scale_addr;
};

struct CacheLayerLayout {
    std::vector<int>               layer_to_groups;
    std::vector<std::vector<int>>  layer_to_group_ids;
    std::vector<std::vector<int>>  layer_region_to_group_id;
    std::vector<CacheGroupType>    group_types;
    std::vector<KVCacheRegionName> group_region_names;
    std::vector<size_t>            group_seq_size_per_block;
    std::vector<CacheGroupType>    layer_group_types;
    // Per-layer kv head num on this rank (empty = all layers homogeneous, use the global
    // value); MiMo V2.5 has GA=4 / SWA=8
    std::vector<int>                        layer_local_kv_head_num;
    std::vector<torch::Tensor>              layers_to_kv_buffer_ptrs;
    std::vector<torch::Tensor>              layers_to_scale_buffer_ptrs;
    std::vector<std::vector<torch::Tensor>> layers_to_kv_buffer_ptrs_by_attn;
    std::vector<std::vector<torch::Tensor>> layers_to_scale_buffer_ptrs_by_attn;
};

struct KVCacheBuffer {
    torch::Tensor kv_blocks;
    torch::Tensor kv_scale_blocks;
};

}  // namespace rtp_llm
