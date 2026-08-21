#include "rtp_llm/cpp/cache/HybridConfigCreator.h"

#include <algorithm>
#include <numeric>

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

std::vector<std::vector<int>> HybridConfigCreator::splitIntoGroups(const std::vector<int>& ids, int group_layer_num) {
    std::vector<std::vector<int>> groups;
    if (ids.empty()) {
        return groups;
    }
    const int n = static_cast<int>(ids.size());
    const int s = std::max(group_layer_num, 1);
    groups.reserve((n + s - 1) / s);
    for (int i = 0; i < n; i += s) {
        const int end = std::min(i + s, n);
        groups.emplace_back(ids.begin() + i, ids.begin() + end);
    }
    return groups;
}

int HybridConfigCreator::calculateGroupLayerNum(int linear_layer_count, int full_layer_count, int swa_layer_count) {
    // Take the gcd of the non-zero per-kind layer counts; with a single kind this is just
    // that kind's layer count (compatible with the old two-argument behavior)
    std::vector<int> nonzero;
    for (int c : {linear_layer_count, full_layer_count, swa_layer_count}) {
        if (c > 0) {
            nonzero.push_back(c);
        }
    }
    if (nonzero.empty()) {
        return 1;
    }
    int g = nonzero[0];
    for (size_t i = 1; i < nonzero.size(); ++i) {
        g = std::gcd(g, nonzero[i]);
    }
    return std::max(g, 1);
}

HybridConfigCreator::LayerSplit HybridConfigCreator::splitLayersByAttentionType(const ModelConfig& model_config) {
    int64_t layer_num = model_config.num_layers;
    RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "invalid model_config.num_layers=%ld", layer_num);

    LayerSplit out;
    out.linear_layers.reserve(layer_num);
    out.full_layers.reserve(layer_num);
    out.swa_layers.reserve(layer_num);

    const auto& types = model_config.hybrid_attention_config.hybrid_attention_types;
    for (int i = 0; i < static_cast<int>(layer_num); ++i) {
        switch (types[static_cast<size_t>(i)]) {
            case HybridAttentionType::LINEAR:
                out.linear_layers.push_back(i);
                break;
            case HybridAttentionType::SLIDING_WINDOW:
                out.swa_layers.push_back(i);
                break;
            default:
                out.full_layers.push_back(i);
                break;
        }
    }
    return out;
}

CacheConfig HybridConfigCreator::initializeConfig(const ModelConfig& model_config, rtp_llm::DataType dtype) {
    int64_t layer_num = model_config.num_layers;

    CacheConfig config;
    config.layer_num          = static_cast<uint32_t>(layer_num);
    config.layer_all_num      = static_cast<uint32_t>(layer_num);
    config.block_num          = 0;
    config.seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    config.use_mla            = model_config.attn_config.use_mla;
    config.dtype              = dtype;
    config.linear_step        = 1;

    return config;
}

KVCacheSpecPtr HybridConfigCreator::createFullAttentionSpec(const ModelConfig&       model_config,
                                                            const ParallelismConfig& parallelism_config,
                                                            rtp_llm::DataType        dtype) {
    KVCacheSpecPtr full_spec;
    if (model_config.attn_config.use_mla && model_config.mla_ops_type != rtp_llm::MlaOpsType::MHA) {
        full_spec = std::make_shared<MLAKVCacheSpec>(model_config.attn_config, parallelism_config);
    } else {
        full_spec = std::make_shared<MHAKVCacheSpec>(model_config.attn_config, parallelism_config);
    }
    full_spec->dtype = dtype;
    return full_spec;
}

KVCacheSpecPtr HybridConfigCreator::createLinearAttentionSpec(const ModelConfig&       model_config,
                                                              const ParallelismConfig& parallelism_config,
                                                              rtp_llm::DataType        dtype) {
    auto linear_spec = std::make_shared<LinearKVCacheSpec>(
        model_config.attn_config, parallelism_config, model_config.linear_attention_config);
    linear_spec->dtype = dtype;
    return linear_spec;
}

KVCacheSpecPtr HybridConfigCreator::createSwaAttentionSpec(const ModelConfig&       model_config,
                                                           const ParallelismConfig& parallelism_config,
                                                           rtp_llm::DataType        dtype) {
    auto swa_spec   = std::make_shared<MHAKVCacheSpec>(model_config.attn_config, parallelism_config);
    swa_spec->dtype = dtype;
    // attn_config.kv_head_num holds the GA value; the SWA layer head count is taken
    // explicitly from swa_attention_config
    const int swa_kv_head_num = model_config.hybrid_attention_config.swa_attention_config.swa_kv_head_num;
    if (swa_kv_head_num > 0) {
        const int tp                = parallelism_config.get_attn_tp_size();
        swa_spec->local_head_num_kv = static_cast<uint32_t>(
            (swa_kv_head_num % tp == 0) ? swa_kv_head_num / tp : swa_kv_head_num / std::gcd(swa_kv_head_num, tp));
    }
    return swa_spec;
}

bool HybridConfigCreator::usePerLayerRowLayout(const LayerSplit&     split,
                                               const KVCacheSpecPtr& full_spec,
                                               const KVCacheSpecPtr& swa_spec) {
    // Mirrors BlockPoolConfigHelper::useDualStrideLayouts(): no LINEAR group, and at
    // least two groups whose per-block strides differ. With per-type grouping the
    // "more than one group" part means both a full and a SWA class exist.
    if (!split.linear_layers.empty() || split.full_layers.empty() || split.swa_layers.empty()) {
        return false;
    }
    if (full_spec == nullptr || swa_spec == nullptr) {
        return false;
    }
    return full_spec->block_size_bytes() != swa_spec->block_size_bytes();
}

HybridConfigCreator::LayerGroups
HybridConfigCreator::createLayerGroups(const LayerSplit& split, bool per_layer_row_layout, int& group_layer_num) {
    const int linear_cnt = static_cast<int>(split.linear_layers.size());
    const int full_cnt   = static_cast<int>(split.full_layers.size());
    const int swa_cnt    = static_cast<int>(split.swa_layers.size());

    LayerGroups groups;

    if (per_layer_row_layout) {
        // One group per attention type, i.e. do not subdivide a type any further.
        //
        // A cache group is the unit of block-id allocation: every group mallocs its
        // own ids, and one id reserves a cell in *every* row of the pool. The legacy
        // layout gives the pool exactly `group_layer_num` rows shared by all groups,
        // so an id is fully consumed by whoever takes it and the group count is free
        // — smaller groups mean a cheaper id and proportionally more ids. That is why
        // gcd subdivision costs nothing there, and it is only there to satisfy the
        // structural requirement that all groups have equal layer counts.
        //
        // The per-layer-row layout breaks that trade: the pool has one row per model
        // layer (9 + 39 for MiMo V2.5), so an id always reserves all 48 rows while a
        // 3-layer group uses 3 of them — a 48/3 = 16x loss. Nothing here needs equal
        // group sizes either: KVCacheGroup::init() addresses layer tensors by global
        // layer id under an explicit layer mapping, not by index within the group.
        groups.full_groups = HybridConfigCreator::splitIntoGroups(split.full_layers, full_cnt);
        groups.swa_groups  = HybridConfigCreator::splitIntoGroups(split.swa_layers, swa_cnt);
        // Groups are deliberately unequal, so group_layer_num has no single value.
        // Only the legacy branch of BlockPoolConfigHelper::createConfig() reads it and
        // that branch is unreachable here; publish the max so any other consumer that
        // treats it as "rows needed" cannot under-count.
        group_layer_num = std::max(full_cnt, swa_cnt);
        return groups;
    }

    group_layer_num      = HybridConfigCreator::calculateGroupLayerNum(linear_cnt, full_cnt, swa_cnt);
    groups.linear_groups = HybridConfigCreator::splitIntoGroups(split.linear_layers, group_layer_num);
    groups.full_groups   = HybridConfigCreator::splitIntoGroups(split.full_layers, group_layer_num);
    groups.swa_groups    = HybridConfigCreator::splitIntoGroups(split.swa_layers, group_layer_num);
    return groups;
}

void HybridConfigCreator::setupCacheConfigSpecs(CacheConfig&          config,
                                                const LayerGroups&    groups,
                                                const KVCacheSpecPtr& linear_spec,
                                                const KVCacheSpecPtr& full_spec,
                                                const KVCacheSpecPtr& swa_spec,
                                                int                   swa_ring_blocks) {
    config.global_layer_ids.clear();
    config.layer_ids.clear();
    config.cache_specs.clear();
    config.group_types.clear();
    config.group_ring_blocks.clear();

    // Keep order: all full groups first, then swa groups, then linear groups.
    for (const auto& g : groups.full_groups) {
        config.global_layer_ids.push_back(g);
        config.layer_ids.push_back(g);
        config.cache_specs.push_back(full_spec);
        config.group_types.push_back(CacheGroupType::FULL);
        config.group_ring_blocks.push_back(0);
    }
    for (const auto& g : groups.swa_groups) {
        config.global_layer_ids.push_back(g);
        config.layer_ids.push_back(g);
        config.cache_specs.push_back(swa_spec);
        // Still typed FULL: the group keeps ordinary paged semantics and ordinary
        // block-cache behaviour. What bounds its footprint is group_ring_blocks, not
        // the group type — CacheGroupType::SWA selects SWAKVCacheGroup, whose
        // "full-length list with only the tail materialized" policy is a different
        // scheme aimed at DSV4's fixed ring pools and does not fit a 64-token page.
        config.group_types.push_back(CacheGroupType::FULL);
        config.group_ring_blocks.push_back(static_cast<uint32_t>(swa_ring_blocks));
    }
    for (const auto& g : groups.linear_groups) {
        config.global_layer_ids.push_back(g);
        config.layer_ids.push_back(g);
        config.cache_specs.push_back(linear_spec);
        config.group_types.push_back(CacheGroupType::LINEAR);
        config.group_ring_blocks.push_back(0);
    }
    config.linear_group_num = static_cast<int>(groups.linear_groups.size());
    config.swa_group_num    = static_cast<int>(groups.swa_groups.size());
    config.full_group_num   = static_cast<int>(groups.full_groups.size());
}

void HybridConfigCreator::setupPhysicalSizes(CacheConfig& config) {
    // Under a dual layout (e.g. MiMo, where GA/SWA strides differ) each group uses the
    // real stride of its own spec, recorded in group_kv_block_stride_bytes
    // (BlockPoolConfigHelper reads it in preference to the compat field below).
    config.group_kv_block_stride_bytes.clear();
    config.group_kv_scale_stride_bytes.clear();
    config.group_block_size_bytes.clear();
    for (const auto& spec : config.cache_specs) {
        config.group_kv_block_stride_bytes.push_back(spec->block_size_bytes());
        config.group_kv_scale_stride_bytes.push_back(spec->scale_block_size_bytes());
        config.group_block_size_bytes.push_back(spec->block_size_bytes() + spec->scale_block_size_bytes());
    }

    // Compat fields: still fill in one representative value (some legacy code reads them),
    // using the maximum stride. Note the original full >= linear assertion was removed --
    // MiMo's GA stride is smaller than its SWA stride, and under a dual layout each group
    // uses its own real stride instead of padding to a single unified stride.
    size_t max_kv = 0, max_scale = 0;
    for (const auto& s : config.cache_specs) {
        max_kv    = std::max(max_kv, s->block_size_bytes());
        max_scale = std::max(max_scale, s->scale_block_size_bytes());
    }
    config.kv_block_stride_bytes = max_kv;
    config.kv_scale_stride_bytes = max_scale;

    // block_size_bytes = sum over groups of (group layer_num * that group's real stride).
    // With two distinct strides this reflects the true total cost across all layers; with a
    // single stride it is equivalent to total_layers * stride.
    size_t total_kv = 0, total_scale = 0;
    for (size_t i = 0; i < config.cache_specs.size(); ++i) {
        size_t layers_in_group = config.layer_ids[i].size();
        total_kv += layers_in_group * config.group_kv_block_stride_bytes[i];
        total_scale += layers_in_group * config.group_kv_scale_stride_bytes[i];
    }
    config.kv_block_size_bytes = total_kv;
    config.kv_scale_size_bytes = total_scale;
    config.block_size_bytes    = total_kv + total_scale;
}

void HybridConfigCreator::setupLayerToGroupMapping(CacheConfig& config) {
    config.layer_to_group_id.assign(config.layer_num, 0);
    for (size_t gid = 0; gid < config.layer_ids.size(); ++gid) {
        for (int layer_id : config.layer_ids[gid]) {
            if (layer_id >= 0 && static_cast<size_t>(layer_id) < config.layer_num) {
                config.layer_to_group_id[static_cast<size_t>(layer_id)] = static_cast<int32_t>(gid);
            }
        }
    }
}

CacheConfig HybridConfigCreator::createHybridConfig(const ModelConfig&       model_config,
                                                    const ParallelismConfig& parallelism_config,
                                                    bool                     is_mtp) {
    auto dtype = MemoryEvaluationHelper::getDataTypeForCache(model_config);

    // Split layers by attention type (linear / full / swa)
    const LayerSplit split = HybridConfigCreator::splitLayersByAttentionType(model_config);

    // Initialize config
    CacheConfig config = HybridConfigCreator::initializeConfig(model_config, dtype);

    // Create attention specs
    auto full_spec = HybridConfigCreator::createFullAttentionSpec(model_config, parallelism_config, dtype);

    KVCacheSpecPtr linear_spec;
    if (!split.linear_layers.empty()) {
        linear_spec = HybridConfigCreator::createLinearAttentionSpec(model_config, parallelism_config, dtype);
    }

    KVCacheSpecPtr swa_spec;
    if (!split.swa_layers.empty()) {
        // Mixed GA/SWA (e.g. MiMo V2.5):
        // - attn_config.kv_head_num holds the GA value, so full_spec is already correct;
        // - the SWA head count is taken explicitly by createSwaAttentionSpec() from
        //   swa_attention_config.swa_kv_head_num.
        swa_spec = HybridConfigCreator::createSwaAttentionSpec(model_config, parallelism_config, dtype);
    }

    // Create layer groups and calculate group layer number. Grouping granularity
    // depends on which pool layout BlockPoolConfigHelper will pick, so decide that
    // first — see usePerLayerRowLayout().
    const bool  per_layer_row_layout = HybridConfigCreator::usePerLayerRowLayout(split, full_spec, swa_spec);
    int         group_layer_num      = 0;
    LayerGroups groups     = HybridConfigCreator::createLayerGroups(split, per_layer_row_layout, group_layer_num);
    config.group_layer_num = group_layer_num;
    RTP_LLM_LOG_INFO("hybrid cache grouping: per_layer_row_layout=%d group_layer_num=%d "
                     "(linear=%zu full=%zu swa=%zu groups)",
                     static_cast<int>(per_layer_row_layout),
                     group_layer_num,
                     groups.linear_groups.size(),
                     groups.full_groups.size(),
                     groups.swa_groups.size());

    // A sliding-window group's cache is a ring of exactly window/page blocks: the
    // layer only ever reads the last `window` tokens, so position p can live at ring
    // slot p % window and the per-request footprint stops growing. This requires the
    // page size to divide the window — otherwise the wrap point falls inside a page
    // and the ring would hold stale slots at arbitrary positions that `window_left`
    // cannot mask out (it masks by position, not by slot). When it does not divide,
    // fall back to 0 = allocate the whole sequence. Keep in sync with
    // swa_ring_pages() in py_flashinfer_mha.py.
    int swa_ring_blocks = 0;
    if (!split.swa_layers.empty()) {
        const int window = model_config.hybrid_attention_config.swa_attention_config.window_size > 0 ?
                               model_config.hybrid_attention_config.swa_attention_config.window_size :
                               static_cast<int>(model_config.attn_config.sliding_window);
        const int page   = static_cast<int>(config.seq_size_per_block);
        if (window > 0 && page > 0 && window % page == 0) {
            swa_ring_blocks = window / page;
        } else if (window > 0) {
            RTP_LLM_LOG_WARNING("SWA ring disabled: sliding_window=%d is not a multiple of "
                                "seq_size_per_block=%d, falling back to full-length allocation",
                                window,
                                page);
        }
    }

    // Setup cache config specs
    HybridConfigCreator::setupCacheConfigSpecs(config, groups, linear_spec, full_spec, swa_spec, swa_ring_blocks);

    // Setup physical sizes (per-group strides + compat fields)
    HybridConfigCreator::setupPhysicalSizes(config);

    // Setup layer to group mapping
    HybridConfigCreator::setupLayerToGroupMapping(config);

    // Populate region mapping: all hybrid groups use DEFAULT region.
    for (size_t gid = 0; gid < config.cache_specs.size(); ++gid) {
        config.group_region_names.push_back(KVCacheRegionName::DEFAULT);
    }
    const size_t region_count = static_cast<size_t>(KVCacheRegionName::REGION_COUNT);
    config.layer_region_to_group_id.resize(config.layer_num);
    for (size_t l = 0; l < config.layer_num; l++) {
        config.layer_region_to_group_id[l].assign(region_count, -1);
        int gid = config.layer_to_group_id[l];
        config.layer_region_to_group_id[l][static_cast<size_t>(KVCacheRegionName::DEFAULT)] = gid;
    }

    config.layer_group_types.assign(config.layer_num, CacheGroupType::FULL);
    for (size_t layer_id = 0; layer_id < config.layer_to_group_id.size(); ++layer_id) {
        const int gid = config.layer_to_group_id[layer_id];
        if (gid >= 0 && static_cast<size_t>(gid) < config.group_types.size()) {
            config.layer_group_types[layer_id] = config.group_types[static_cast<size_t>(gid)];
        }
    }

    // Per-layer block stride (kv + scale).
    // Under a dual layout, fill in the real stride of the group each layer belongs to;
    // under a single layout (all groups share a stride) this matches the original logic.
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num), 0);
    for (size_t l = 0; l < config.layer_to_group_id.size() && l < config.layer_to_block_stride_bytes.size(); ++l) {
        const int gid = config.layer_to_group_id[l];
        if (gid >= 0 && static_cast<size_t>(gid) < config.group_block_size_bytes.size()) {
            config.layer_to_block_stride_bytes[l] = static_cast<int>(config.group_block_size_bytes[gid]);
        } else {
            config.layer_to_block_stride_bytes[l] =
                static_cast<int>(config.kv_block_stride_bytes + config.kv_scale_stride_bytes);
        }
    }

    return config;
}

}  // namespace rtp_llm