#pragma once

#include <memory>
#include <vector>
#include <utility>
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

class HybridConfigCreator {
public:
    // Layer id sets split by attention type (three kinds: linear / full / sliding-window)
    struct LayerSplit {
        std::vector<int> linear_layers;
        std::vector<int> full_layers;
        std::vector<int> swa_layers;
    };
    // Grouping result for each of the three layer kinds
    struct LayerGroups {
        std::vector<std::vector<int>> linear_groups;
        std::vector<std::vector<int>> full_groups;
        std::vector<std::vector<int>> swa_groups;
    };

    static CacheConfig                   createHybridConfig(const ModelConfig&       model_config,
                                                            const ParallelismConfig& parallelism_config,
                                                            bool                     is_mtp = false);
    static std::vector<std::vector<int>> splitIntoGroups(const std::vector<int>& ids, int group_layer_num);

    // Calculate the number of layers per group based on linear/full/swa layers count
    static int calculateGroupLayerNum(int linear_layer_count, int full_layer_count, int swa_layer_count = 0);

private:
    // Helper functions for creating hybrid config in the order they appear in the main flow
    static LayerSplit     splitLayersByAttentionType(const ModelConfig& model_config);
    static CacheConfig    initializeConfig(const ModelConfig& model_config, rtp_llm::DataType dtype);
    static KVCacheSpecPtr createFullAttentionSpec(const ModelConfig&       model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  rtp_llm::DataType        dtype);
    static KVCacheSpecPtr createLinearAttentionSpec(const ModelConfig&       model_config,
                                                    const ParallelismConfig& parallelism_config,
                                                    rtp_llm::DataType        dtype);
    static KVCacheSpecPtr createSwaAttentionSpec(const ModelConfig&       model_config,
                                                 const ParallelismConfig& parallelism_config,
                                                 rtp_llm::DataType        dtype);
    // True when BlockPoolConfigHelper will lay the pool out with one row per model
    // layer (per-stride layouts + explicit layer mapping) instead of the legacy
    // layout that gives the pool only `group_layer_num` rows shared by every group.
    //
    // Must stay in sync with BlockPoolConfigHelper::useDualStrideLayouts(), which
    // decides the same thing but from an already-grouped CacheConfig — grouping has
    // to be chosen before that exists, so the predicate is duplicated here over
    // (split, specs). The two differ only in how they spell "more than one group".
    static bool
    usePerLayerRowLayout(const LayerSplit& split, const KVCacheSpecPtr& full_spec, const KVCacheSpecPtr& swa_spec);
    static LayerGroups createLayerGroups(const LayerSplit& split, bool per_layer_row_layout, int& group_layer_num);
    static void        setupCacheConfigSpecs(CacheConfig&          config,
                                             const LayerGroups&    groups,
                                             const KVCacheSpecPtr& linear_spec,
                                             const KVCacheSpecPtr& full_spec,
                                             const KVCacheSpecPtr& swa_spec,
                                             int                   swa_ring_blocks);
    static void        setupPhysicalSizes(CacheConfig& config);
    static void        setupLayerToGroupMapping(CacheConfig& config);
};

}  // namespace rtp_llm