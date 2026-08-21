#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/BlockPoolConfig.h"

#include <string>

namespace rtp_llm {

class BlockPoolConfigHelper {
public:
    /**
     * Create block pool config from CacheConfig.
     * Supports both single model and MTP (1+N models) configuration.
     * Memory layout is [layout0_kv][layout0_scale][layout1_kv][layout1_scale]...[layoutN_kv][layoutN_scale]
     * Generally Memory layout is [main_kv][main_scale][mtp1_kv][mtp1_scale]...[mtpN_kv][mtpN_scale]
     *
     * @param cache_config The CacheConfig containing main model and optional MTP modules
     */
    static BlockPoolConfig createConfig(const CacheConfig& cache_config) {
        RTP_LLM_CHECK_WITH_INFO(!cache_config.cache_specs.empty(), "cache_specs must not be empty");
        BlockPoolConfig config;
        config.pool_name      = "default";
        config.block_num      = cache_config.block_num;
        const bool  is_hybrid = cache_config.groupNums() > 1;
        auto        layer_num = is_hybrid ? cache_config.group_layer_num : cache_config.layer_num;
        const auto& main_spec = cache_config.cache_specs[0];

        size_t current_offset = 0;
        if (useDualStrideLayouts(cache_config)) {
            // GA/SWA dual layout (e.g. MiMo V2.5): one layout per distinct stride, with
            // an explicit mapping table for the interleaved layer distribution. Only
            // reached when there is no LINEAR group.
            current_offset = appendPerStrideLayouts(config, cache_config);
        } else {
            // linear block size is same with full block block size
            MemoryLayoutConfig main_layout = createMemoryLayoutConfig(is_hybrid,
                                                                      layer_num,
                                                                      cache_config.kv_block_stride_bytes,
                                                                      cache_config.kv_scale_stride_bytes,
                                                                      main_spec,
                                                                      cache_config);

            main_layout.kv_cache_offset_bytes = 0;
            main_layout.kv_scale_offset_bytes =
                main_layout.kv_cache_offset_bytes + main_layout.kv_block_pool_size_bytes;
            current_offset = main_layout.kv_scale_offset_bytes + main_layout.kv_scale_pool_size_bytes;
            RTP_LLM_LOG_INFO("main_layout.kv_scale_offset_bytes: %zu", main_layout.kv_scale_offset_bytes);
            RTP_LLM_LOG_INFO("main_layout.kv_scale_pool_size_bytes: %zu", main_layout.kv_scale_pool_size_bytes);

            config.memory_layouts.push_back(main_layout);
        }

        // Create MTP sub-model layouts
        for (size_t i = 0; i < cache_config.mtp_sub_configs.size(); ++i) {
            const auto& mtp_sub_config = cache_config.mtp_sub_configs[i];
            RTP_LLM_CHECK_WITH_INFO(mtp_sub_config != nullptr, "mtp_sub_configs[%zu] is null", i);
            RTP_LLM_CHECK_WITH_INFO(
                !mtp_sub_config->cache_specs.empty(), "MTP module %zu cache_specs must not be empty", i);

            const auto mtp_layer_num = mtp_sub_config->layer_num;

            const auto& mtp_spec = mtp_sub_config->cache_specs[0];
            // mtp block size is not same with main model block size
            MemoryLayoutConfig mtp_layout = createMemoryLayoutConfig(false,
                                                                     mtp_layer_num,
                                                                     mtp_spec->block_size_bytes(),
                                                                     mtp_spec->scale_block_size_bytes(),
                                                                     mtp_spec,
                                                                     cache_config);

            mtp_layout.kv_cache_offset_bytes = current_offset;
            RTP_LLM_LOG_INFO("mtp_layout.kv_block_pool_size_bytes = %ld", mtp_layout.kv_block_pool_size_bytes);
            current_offset += mtp_layout.kv_block_pool_size_bytes;

            if (mtp_layout.hasScale()) {
                mtp_layout.kv_scale_offset_bytes = current_offset;
                RTP_LLM_LOG_INFO("mtp_layout.kv_scale_pool_size_bytes = %ld", mtp_layout.kv_scale_pool_size_bytes);
                current_offset += mtp_layout.kv_scale_pool_size_bytes;
            } else {
                mtp_layout.kv_scale_offset_bytes = current_offset;
            }

            // When the explicit mapping is enabled, MTP layers keep the contiguous cursor
            // semantics; append them to the mapping table to stay consistent
            if (!config.explicit_layer_mapping.empty()) {
                const int layout_idx = static_cast<int>(config.memory_layouts.size());
                for (uint32_t local = 0; local < mtp_layer_num; ++local) {
                    config.explicit_layer_mapping.emplace_back(layout_idx, static_cast<int>(local));
                }
            }

            config.memory_layouts.push_back(mtp_layout);
        }

        config.total_size_bytes = current_offset;

        RTP_LLM_LOG_INFO("BlockPoolConfig(memory_layouts=%zu): total_size=%zu bytes",
                         config.memory_layouts.size(),
                         config.total_size_bytes);
        return config;
    }

    static BlockPoolConfig createConfigForGroup(const CacheConfig& cache_config, size_t group_id) {
        RTP_LLM_CHECK_WITH_INFO(group_id < cache_config.cache_specs.size(),
                                "group_id %zu out of range, cache_specs.size=%zu",
                                group_id,
                                cache_config.cache_specs.size());
        RTP_LLM_CHECK_WITH_INFO(group_id < cache_config.global_layer_ids.size(),
                                "group_id %zu out of range, global_layer_ids.size=%zu",
                                group_id,
                                cache_config.global_layer_ids.size());
        const auto& spec = cache_config.cache_specs[group_id];
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache_specs[%zu] is null", group_id);

        BlockPoolConfig config;
        config.pool_name = "group_" + std::to_string(group_id);
        const bool has_group_blocks =
            group_id < cache_config.group_block_nums.size() && cache_config.group_block_nums[group_id] > 0;
        config.block_num = has_group_blocks ? cache_config.group_block_nums[group_id] : cache_config.block_num;
        RTP_LLM_LOG_INFO("createConfigForGroup: gid=%zu block_num=%d (has_group_blocks=%d, "
                         "group_block_nums.size=%zu, global_block_num=%d)",
                         group_id,
                         config.block_num,
                         has_group_blocks,
                         cache_config.group_block_nums.size(),
                         cache_config.block_num);

        const uint32_t layer_num = static_cast<uint32_t>(cache_config.global_layer_ids[group_id].size());
        RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "group %zu has no layers", group_id);

        const size_t kv_stride    = (group_id < cache_config.group_kv_block_stride_bytes.size()
                                  && cache_config.group_kv_block_stride_bytes[group_id] > 0) ?
                                        cache_config.group_kv_block_stride_bytes[group_id] :
                                        spec->block_size_bytes();
        const size_t scale_stride = (group_id < cache_config.group_kv_scale_stride_bytes.size()) ?
                                        cache_config.group_kv_scale_stride_bytes[group_id] :
                                        spec->scale_block_size_bytes();

        CacheConfig group_cache_config = cache_config;
        group_cache_config.block_num   = config.block_num;
        if (group_id < cache_config.group_seq_size_per_block.size()
            && cache_config.group_seq_size_per_block[group_id] > 0) {
            group_cache_config.seq_size_per_block = cache_config.group_seq_size_per_block[group_id];
        }

        MemoryLayoutConfig layout =
            createMemoryLayoutConfig(false, layer_num, kv_stride, scale_stride, spec, group_cache_config);
        RTP_LLM_CHECK_WITH_INFO(group_id < cache_config.group_types.size(),
                                "missing cache group type for group %zu (group_types.size=%zu)",
                                group_id,
                                cache_config.group_types.size());
        const bool is_full_group          = cache_config.group_types[group_id] == CacheGroupType::FULL;
        layout.kernel_blocks_per_kv_block = is_full_group ? cache_config.kernelBlocksPerKvBlock() : 1;
        layout.kv_cache_offset_bytes      = 0;
        layout.kv_scale_offset_bytes      = layout.kv_cache_offset_bytes + layout.kv_block_pool_size_bytes;

        config.memory_layouts.push_back(layout);
        config.total_size_bytes = layout.kv_block_pool_size_bytes + layout.kv_scale_pool_size_bytes;
        return config;
    }

    // for memory connector
    static BlockPoolConfig
    createConfig(uint32_t layer_num, uint32_t block_num, size_t block_stride_bytes, rtp_llm::DataType dtype) {
        BlockPoolConfig config;
        config.pool_name = "memory_connector";
        config.block_num = block_num;

        MemoryLayoutConfig layout_cfg;
        layout_cfg.layer_num = layer_num;
        layout_cfg.block_num = block_num;

        layout_cfg.kv_block_stride_bytes = block_stride_bytes;
        layout_cfg.dtype                 = dtype;

        layout_cfg.kv_cache_offset_bytes = 0;
        layout_cfg.kv_block_pool_size_bytes =
            static_cast<size_t>(layer_num) * static_cast<size_t>(block_num) * block_stride_bytes;
        layout_cfg.kv_scale_offset_bytes    = layout_cfg.kv_cache_offset_bytes + layout_cfg.kv_block_pool_size_bytes;
        layout_cfg.kv_scale_pool_size_bytes = 0;
        layout_cfg.total_size_bytes         = layout_cfg.kv_block_pool_size_bytes;

        config.memory_layouts   = {layout_cfg};
        config.total_size_bytes = layout_cfg.total_size_bytes;
        return config;
    }

private:
    // Whether to enable "one layout per stride class": hybrid, all groups are attention
    // groups (no LINEAR), and the group strides are not all identical (e.g. MiMo
    // GA 40960B / SWA 81920B). Existing models such as linear+full have a single stride
    // and do not take this branch, so their behavior is unchanged.
    //
    // This branch gives the pool "one row per layer", so a single block id occupies a
    // slot in every layer; the legacy branch allocates only group_layer_num rows shared
    // across groups. The grouping granularity must follow whichever branch is taken,
    // which is why HybridConfigCreator::usePerLayerRowLayout() duplicates the same
    // predicate (it runs before grouping and has no fully-built CacheConfig available).
    // Keep the two in sync when changing the condition here.
    static bool useDualStrideLayouts(const CacheConfig& cache_config) {
        if (cache_config.groupNums() <= 1
            || cache_config.group_kv_block_stride_bytes.size() != cache_config.cache_specs.size()) {
            return false;
        }
        for (const auto t : cache_config.group_types) {
            if (t == CacheGroupType::LINEAR) {
                return false;
            }
        }
        size_t first = cache_config.group_kv_block_stride_bytes[0];
        for (size_t s : cache_config.group_kv_block_stride_bytes) {
            if (s != first) {
                return true;
            }
        }
        return false;
    }

    // Build one layout per stride class (preserving first-appearance order) and generate
    // the explicit layer mapping. Returns the accumulated byte offset so that subsequent
    // MTP layouts can keep appending.
    static size_t appendPerStrideLayouts(BlockPoolConfig& config, const CacheConfig& cache_config) {
        struct StrideClass {
            size_t         kv_stride    = 0;
            size_t         scale_stride = 0;
            KVCacheSpecPtr spec;
            uint32_t       layer_num = 0;
        };
        std::vector<StrideClass> classes;
        std::vector<int>         group_to_class(cache_config.cache_specs.size(), -1);

        for (size_t gid = 0; gid < cache_config.cache_specs.size(); ++gid) {
            const auto&  spec         = cache_config.cache_specs[gid];
            const size_t kv_stride    = cache_config.group_kv_block_stride_bytes[gid];
            const size_t scale_stride = gid < cache_config.group_kv_scale_stride_bytes.size() ?
                                            cache_config.group_kv_scale_stride_bytes[gid] :
                                            spec->scale_block_size_bytes();
            int          ci           = -1;
            for (size_t c = 0; c < classes.size(); ++c) {
                if (classes[c].kv_stride == kv_stride && classes[c].scale_stride == scale_stride) {
                    ci = static_cast<int>(c);
                    break;
                }
            }
            if (ci < 0) {
                ci = static_cast<int>(classes.size());
                classes.push_back({kv_stride, scale_stride, spec, 0});
            }
            group_to_class[gid] = ci;
            classes[static_cast<size_t>(ci)].layer_num +=
                static_cast<uint32_t>(cache_config.global_layer_ids[gid].size());
        }

        size_t current_offset = 0;
        for (size_t c = 0; c < classes.size(); ++c) {
            MemoryLayoutConfig layout    = createMemoryLayoutConfig(/*enable_hybrid_attention=*/true,
                                                                 classes[c].layer_num,
                                                                 classes[c].kv_stride,
                                                                 classes[c].scale_stride,
                                                                 classes[c].spec,
                                                                 cache_config);
            layout.kv_cache_offset_bytes = current_offset;
            current_offset += layout.kv_block_pool_size_bytes;
            layout.kv_scale_offset_bytes = current_offset;
            if (layout.hasScale()) {
                current_offset += layout.kv_scale_pool_size_bytes;
            }
            RTP_LLM_LOG_INFO("per-stride layout[%zu]: layer_num=%u kv_stride=%zu scale_stride=%zu kv_off=%zu",
                             c,
                             classes[c].layer_num,
                             classes[c].kv_stride,
                             classes[c].scale_stride,
                             layout.kv_cache_offset_bytes);
            config.memory_layouts.push_back(layout);
        }

        // Explicit layer mapping: global_layer_id -> {layout_idx, local index within layout}
        config.explicit_layer_mapping.assign(static_cast<size_t>(cache_config.layer_num), {-1, -1});
        std::vector<int> class_cursor(classes.size(), 0);
        for (size_t gid = 0; gid < cache_config.global_layer_ids.size(); ++gid) {
            const int ci = group_to_class[gid];
            for (int layer_id : cache_config.global_layer_ids[gid]) {
                RTP_LLM_CHECK_WITH_INFO(layer_id >= 0
                                            && static_cast<size_t>(layer_id) < config.explicit_layer_mapping.size(),
                                        "layer_id %d out of range for explicit mapping (layer_num=%u)",
                                        layer_id,
                                        cache_config.layer_num);
                config.explicit_layer_mapping[static_cast<size_t>(layer_id)] = {
                    ci, class_cursor[static_cast<size_t>(ci)]++};
            }
        }
        return current_offset;
    }

    static MemoryLayoutConfig createMemoryLayoutConfig(bool           enable_hybrid_attention,
                                                       uint32_t       layer_num,
                                                       size_t         kv_block_stride_bytes,
                                                       size_t         kv_scale_stride_bytes,
                                                       KVCacheSpecPtr spec,
                                                       CacheConfig    cache_config) {
        MemoryLayoutConfig cfg;
        cfg.layer_num             = layer_num;
        cfg.block_num             = cache_config.block_num;
        cfg.kv_block_stride_bytes = kv_block_stride_bytes;
        cfg.k_block_stride_bytes  = spec->k_block_size_bytes();
        cfg.v_block_stride_bytes  = spec->v_block_size_bytes();
        cfg.kv_scale_stride_bytes = kv_scale_stride_bytes;
        cfg.k_scale_stride_bytes  = spec->k_scale_block_size_bytes();
        cfg.v_scale_stride_bytes  = spec->v_scale_block_size_bytes();

        cfg.enable_kv_scale         = cfg.kv_scale_stride_bytes > 0;
        cfg.dtype                   = spec->dtype;
        cfg.local_head_num_kv       = spec->local_head_num_kv;
        cfg.enable_hybrid_attention = enable_hybrid_attention;
        // Scale 3D layout for MLA and indexer; KV 3D only for MLA (concat_and_cache_mla)
        cfg.is_mla             = cache_config.use_mla || cache_config.is_sparse;
        cfg.use_mla            = cache_config.use_mla;
        cfg.seq_size_per_block = static_cast<size_t>(cache_config.seq_size_per_block);

        cfg.kv_block_pool_size_bytes =
            static_cast<size_t>(layer_num) * static_cast<size_t>(cfg.block_num) * cfg.kv_block_stride_bytes;

        cfg.kv_scale_pool_size_bytes =
            static_cast<size_t>(layer_num) * static_cast<size_t>(cfg.block_num) * cfg.kv_scale_stride_bytes;
        cfg.total_size_bytes = cfg.kv_block_pool_size_bytes + cfg.kv_scale_pool_size_bytes;
        return cfg;
    }
};

}  // namespace rtp_llm
