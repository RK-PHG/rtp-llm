# MiMo-V2.5 支持说明

本文按推理链路的先后顺序，说明 RTP-LLM 为支持 MiMo-V2.5 在每个环节所做的工作。

---

## 0. 模型事实

MiMo-V2.5 是 48 层 MoE omni 模型。与框架既有模型相比，有 **7 个非标准点**，每一个都对应下面某个环节的专门处理。

### 0.1 超参

| 项 | 值 | 来源 key |
|---|---|---|
| num_hidden_layers | 48 | `num_hidden_layers` |
| hidden_size | 4096 | `hidden_size` |
| vocab_size | 152576 | `vocab_size` |
| num_attention_heads | 64 | `num_attention_heads` |
| **QK head_dim** | **192** | `head_dim` |
| **V head_dim** | **128** | `v_head_dim` |
| GA kv heads | 4 | `num_key_value_heads` |
| SWA kv heads | 8 | `swa_num_key_value_heads` |
| sliding_window | 128 | `sliding_window` |
| GA rope_theta | 1e7 | `rope_theta` |
| SWA rope_theta | 1e4 | `swa_rope_theta` |
| partial_rotary_factor | 0.334 | `partial_rotary_factor` |
| attention_value_scale | 0.707 | `attention_value_scale` |
| layernorm_epsilon | 1e-5 | `layernorm_epsilon` |
| 专家数 / top_k | 256 / 8 | `n_routed_experts` / `num_experts_per_tok` |
| MoE inter / dense inter | 2048 / 16384 | `moe_intermediate_size` / `intermediate_size` |

### 0.2 七个非标准点

1. **逐层混合注意力**。`hybrid_layer_pattern[i] == 1` 表示 SWA，`== 0` 表示 GA（全局注意力）。
   GA 共 9 层：`0, 5, 11, 17, 23, 29, 35, 41, 47`；其余 39 层是 SWA。
   两类层的 **kv 头数不同**（4 / 8），因此 KV cache 每层字节数不同。

2. **K/V head_dim 不相等**。QK=192、V=128。框架原先所有 MHA 路径都假设 K/V 同维。

3. **Partial RoPE**。RoPE 只作用于每个 head 的**前** `int(192 × 0.334) = 64` 维，后 128 维不加位置信息；NEOX 风格（`rotate_half`）。

4. **双 rope base**。GA 用 1e7，SWA 用 1e4。同一模型内两套 cos/sin cache。

5. **Attention sink**。仅 SWA 层带一个 per-head 可学习标量 `attention_sink_bias`（BF16，形状 `[64]`），作为额外一列 logit 参与 softmax 分母、输出时丢弃。GA 层没有。

6. **attention_value_scale**。参考实现对所有 48 层的 V 统一乘 0.707，且发生在写 KV cache **之前**。

7. **FP8 融合 QKV 是 slab-interleaved**。ckpt 里 Q/K/V 是单个 `qkv_proj.weight` 张量，且沿行方向被切成 4 个独立量化的 slab；`o_proj` 是 BF16（在 `quantization_config.ignored_layers` 里）。

---

## 1. 注册与配置解析

**文件**：[rtp_llm/models/mimo_v25.py](rtp_llm/models/mimo_v25.py)、[rtp_llm/model_factory_register.py](rtp_llm/model_factory_register.py)

### 1.1 注册

双注册（与 qwen3_next 相同惯例）：

- [model_factory_register.py:329](rtp_llm/model_factory_register.py#L329) 的 `register_lazy_model("mimo_v25", "rtp_llm.models.mimo_v25", ["MiMoV2ForCausalLM"])`
- [mimo_v25.py](rtp_llm/models/mimo_v25.py) 模块底部的 `register_model("mimo_v25", MiMoV25, ["MiMoV2ForCausalLM"])`

解析入口是 `ModelDict.get_ft_model_type_by_config`，按 `config.json` 的 `architectures[0]`（不是 `model_type`）查表，因此 `MiMoV2ForCausalLM → mimo_v25`。

### 1.2 配置映射

`MiMoV25._create_config()` 分 6 段解析。关键映射：

| ModelConfig 字段 | 值 | 说明 |
|---|---|---|
| `attn_config.size_per_head` | 192 | QK |
| `attn_config.v_size_per_head` | 128 | 新增字段，见 §2 |
| `attn_config.kv_head_num` | **4** | **GA 值**；SWA 的 8 单独放 `swa_attention_config` |
| `attn_config.rope_config.style` | 1 | `RopeStyle::Base`（NEOX） |
| `attn_config.rope_config.base` | 1e7 | GA base；SWA 在前向按层覆盖 |
| `attn_config.rope_config.dim` | 64 | `int(192 × 0.334)`，partial rope |
| `attn_config.sliding_window` | 128 | 全模型标记「本模型有滑窗」 |
| `attn_config.add_sink_bias` | True | 全模型标记「本模型有 sink」 |
| `hybrid_attention_config.enable_hybrid_attention` | True | 走 `HybridConfigCreator` |
| `hybrid_attention_config.hybrid_attention_types` | 48 项 | `NONE`(GA) / `SLIDING_WINDOW`(SWA) |
| `swa_attention_config.window_size` | 128 | |
| `swa_attention_config.swa_kv_head_num` | 8 | |
| `swa_attention_config.ga_kv_head_num` | 4 | |
| `swa_attention_config.swa_rope_theta` | 1e4 | |
| `swa_attention_config.add_sink_bias` | True | |
| `moe_style` / `scoring_func` | 1 / 1 | 无 shared expert / sigmoid |
| `moe_layer_index` | `[1..47]` | layer 0 是 dense |

**`attn_config.kv_head_num` 填 GA 值（4）而非 SWA 值**，是因为 C++ 侧 `createFullAttentionSpec()` 直接用它构造 GA 的 cache spec；SWA 的头数由 `createSwaAttentionSpec()` 从 `swa_attention_config.swa_kv_head_num` 显式取。

`ModelConfig.__setattr__` 有属性白名单（`_python_fields` / `_cpp_members`），未登记的属性名直接抛 `AttributeError`。所以新字段必须挂在 C++ config struct 上，不能直接往 `ModelConfig` 加属性。

### 1.3 入口断言

解析阶段卡住 5 个前提，任何一个不成立就在启动最早期失败，而不是等到精度对拍：

```python
assert cj.get("swa_head_dim", cj["head_dim"]) == cj["head_dim"]
assert cj.get("swa_v_head_dim", cj["v_head_dim"]) == cj["v_head_dim"]
assert cj.get("swa_num_attention_heads", cj["num_attention_heads"]) == cj["num_attention_heads"]
assert cj.get("attention_projection_layout") == "fused_qkv"
assert not cj.get("add_full_attention_sink_bias", False)
assert cj["hidden_act"] == "silu"
assert cj.get("n_shared_experts") is None
assert cj["topk_method"] == "noaux_tc"
```

前三条尤其重要：config schema 允许 `swa_head_dim` / `swa_v_head_dim` / `swa_num_attention_heads` 与非 SWA 值不同（见 ckpt 自带的 `configuration_mimo_v2.py:243-248`），本 ckpt 恰好相同。若将来不同，融合 qkv 的行边界、`o_proj` 输入维、两个 KV spec 的 stride 都得改成逐层量。

### 1.4 停止词

`config.json` 只给单个 `eos_token_id`（151645 `<|im_end|>`），而 `generation_config.json` 给的是三项列表 `[151643, 151645, 151672]`。框架 `eos_token_id` 是标量，`_parse_stop_words()` 从 ckpt 读 `generation_config.json`，把多出来的送进 `stop_words_id_list`，不硬编码任何 token id。

---

## 2. C++ 配置字段与 pybind

### 2.1 新增字段

| 文件 | 新增 |
|---|---|
| [AttentionConfig.h](rtp_llm/cpp/model_utils/AttentionConfig.h) | `size_t v_size_per_head = 0`（0 = 与 `size_per_head` 相同）、`bool add_sink_bias = false`、辅助函数 `vSizePerHead()` |
| [ConfigModules.h](rtp_llm/cpp/config/ConfigModules.h) | `struct SwaAttentionConfig { window_size, swa_kv_head_num, ga_kv_head_num, swa_rope_theta, add_sink_bias }`；`HybridAttentionConfig` 内新增成员 `swa_attention_config` |
| [KVCacheSpecBase.h](rtp_llm/cpp/cache/KVCacheSpecBase.h) | `KVCacheSpecType::SlidingWindowAttention` |
| [CacheConfig.h](rtp_llm/cpp/cache/CacheConfig.h) | `std::vector<uint32_t> group_ring_blocks`（逐组环形容量，见 §4.3） |
| [BufferTypes.h](rtp_llm/cpp/cache/BufferTypes.h) | `std::vector<int> layer_local_kv_head_num`（逐层本 rank kv 头数） |
| [BlockPoolConfig.h](rtp_llm/cpp/cache/BlockPoolConfig.h) | `std::vector<std::pair<int,int>> explicit_layer_mapping` |

`v_size_per_head` 用 `0` 表示「与 `size_per_head` 相同」，所有下游都走 `vSizePerHead()`，对称模型零影响。

### 2.2 pybind 暴露

C++ 加字段不等于 Python 能赋值。绑定集中在 [ConfigInit.cc](rtp_llm/cpp/pybind/ConfigInit.cc)，共四处：

1. `AttentionConfigs` 块补 `v_size_per_head` / `add_sink_bias`
2. 新注册 `SwaAttentionConfig` 类 —— **必须排在 `HybridAttentionConfig` 之前**，否则 `def_readwrite` 按引用暴露该类型时运行期报 `Unable to convert function return value to a Python type`
3. `HybridAttentionConfig` 暴露 `swa_attention_config` 成员
4. Python 侧 [ops/__init__.py](rtp_llm/ops/__init__.py) 的再导出 + [libth_transformer_config.pyi](rtp_llm/ops/libth_transformer_config.pyi) 存根

[OpDefs.cc](rtp_llm/models_py/bindings/OpDefs.cc) 另外暴露了 `LayerKVCache.k_cache` / `.v_cache` 和 `KVCache.v_head_dim` / `.num_kv_heads_by_layer`（见 §5）。

`SwaAttentionConfig::to_string()` 在 [ConfigModules.cc](rtp_llm/cpp/config/ConfigModules.cc) 有实现并接进 `HybridAttentionConfig::to_string()`；只写声明会在 pybind 引用它时链接期报 undefined symbol。

---

## 3. 权重加载

**文件**：[rtp_llm/models/mimo_v25_weight.py](rtp_llm/models/mimo_v25_weight.py)

与通用路径有三处差异，各有专门处理。

### 3.1 融合 QKV 保持融合

ckpt 的 QKV 是单张量 `model.layers.{i}.self_attn.qkv_proj.weight`。通用 FP8 量化包装 `PerBlockFp8Weight._get_qkv_quant_weight()` 假设 Q/K/V 是三个独立张量并硬编码 `merge_te_qkv`，对 MiMo 会解包失败。

分发靠类属性标记 `is_mimo_v25 = True`（作用等同 DSV4 的 `is_v4_weight`，但 DSV4 靠名字前缀、MiMo 用的是标准 W slot 名，所以只能用实例标记）：

- [model_weight.py](rtp_llm/utils/model_weight.py) 新增 `is_mimo_v25_weight()`
- [per_block_fp8_quant_weight.py](rtp_llm/model_loader/per_block_fp8_quant_weight.py) 的 `PerBlockFp8Weight.support()` 里让带标记的权重放行
- `MiMoPerBlockFp8Weight.support()` 接管，且只接管 `W.attn_qkv_w`

形状核对（与 ckpt 实测一致）：

```
GA  qkv_proj.weight  [13568, 4096]   13568 = 64×192 + 4×192 + 4×128
SWA qkv_proj.weight  [14848, 4096]   14848 = 64×192 + 8×192 + 8×128
    o_proj.weight    [4096, 8192]     8192 = 64×128
```

### 3.2 FP8 slab-interleaved 布局与零转换 TP 切分

ckpt 的融合 QKV 沿行方向分成 **4 个 slab**，每个 slab 内部是完整的 `[Q_shard | K_shard | V_shard]`，且**每个 slab 独立做 FP8 block 量化**（block = `[128, 128]`）：

| 层型 | slab 内行数 | 组成 | slab scale 行数 | scale 总行数 |
|---|---|---|---|---|
| GA (kv=4) | 3392 | Q=16头×192, K=1头×192, V=1头×128 | `ceil(3392/128)` = 27 | 27×4 = **108** |
| SWA (kv=8) | 3712 | Q=16头×192, K=2头×192, V=2头×128 | `ceil(3712/128)` = 29 | 29×4 = **116** |

这与 ckpt 里 `weight_scale_inv` 的实际行数完全吻合。反过来也排除了「整体作为一个张量量化」的可能 —— 那样 GA 会是 `ceil(13568/128) = 106` 行，而不是 108。

slab 边界恰好等于 TP=4 的标准 head 切分（rank r 拿 Q 头 `16r..16r+15`、kv 头 `r`），而且 slab 内部已经是 kernel 期望的 section-major `[Q|K|V]`。因此 `MiMoPerBlockFp8Weight._split()` 做的是**零转换切片** —— 每个 rank 直接按行区间切出 FP8 权重和 scale，不 dequant / requant：

```python
new_weight = weight_fp8[tp_rank * slab_rows : (tp_rank + 1) * slab_rows]
new_scale  = scale_inv[tp_rank * slab_scale_rows : (tp_rank + 1) * slab_scale_rows]
```

**TP 必须等于 4**。断言 `tp == QKV_QUANT_SHARDS`：slab 是 ckpt 原生的量化单元，FP8 域里既不能合并也不能再切（K 和 V 共享量化 block 25，`slab_rows = 3392` 也不是 128 的整数倍）。`check_qkv_scale_rows()` 额外校验 scale 总行数 == `slab_scale_rows × 4`，ckpt 换版本时先在这里失败。

### 3.3 attention_value_scale 折进 o_proj

参考实现里 `value_states = value_states * 0.707` 发生在写 KV cache 之前，作用于所有 48 层。因为 attention 对 V 是线性的：

```
0.707 · (softmax(QKᵀ) · V) · W_o  ==  (softmax(QKᵀ) · V) · (0.707 · W_o)
```

所以外提到 `o_proj` 完全等价。`_transpose_with_value_scale()` 在加载时把它折进 `o_proj` 的 BF16 权重。这样做的好处是 qkv 路径完全不涉及 float 域操作，FP8 权重可以原样切片（§3.2）。

> 前提是 `o_proj` 为 BF16。若将来 ckpt 把 `o_proj` 也量化，这个折叠会变成量化误差来源。

### 3.4 o_proj 走非量化路径

`o_proj` 在 `quantization_config.ignored_layers` 里，ckpt 中没有 `o_proj.weight_scale_inv`。`MiMoBf16AtomicWeight` 打上 `is_mimo_v25` 标记让 FP8 包装放行，切分沿用 `W.gpt_style_tp_strategy` 的默认规则。

### 3.5 逐层 kv 头数

`_get_hf_layer_weight_info(layer_id)` 按层型构造各自的 `AttnConfig`：

```python
kv_heads = swa_cfg.ga_kv_head_num if is_ga else swa_cfg.swa_kv_head_num
attn_config = AttnConfig(
    size_per_head=192, v_size_per_head=128, head_num=64, head_num_kv=kv_heads,
)
```

[attn_weight.py](rtp_llm/model_loader/attn_weight.py) 的 `AttnConfig` 为此新增了 `v_size_per_head` 字段。BF16 ckpt 路径下 `MiMoAttnAtomicWeight._split()` 用新增的 [`get_sp_tensor_kv_asym()`](rtp_llm/utils/model_weight.py) 按三段（`q_hidden` / `k_hidden` / `v_hidden`）分别切头再拼接，kv 段按 `gcd(head_num_kv, tp)` 处理。

### 3.6 sink bias

新增 W slot `W.attn_sink_bias = "self_attention_weights.sink_bias"`，只在 SWA 层登记：

```python
if not is_ga:
    weights.append(AtomicWeight(W.attn_sink_bias, [...attention_sink_bias], identity))
```

TP 策略是 `sp_0`（按第 0 维切）—— FlashInfer 用**局部** `qo_head_idx` 索引 sinks，所以每个 rank 只能持有自己那份 head 切片，不能拿整个张量。

### 3.7 MoE 与非文本权重

MoE 结构与 deepseek_v2 同构（sigmoid + `e_score_correction_bias` + `moe_layer_index` 区分 dense/MoE、无共享专家），映射对齐其实现。layer 0 走 `_get_dense_ffn_layer_weight_info()`（inter=16384），layer 1~47 走 `_get_moe_layer_weight_info()`（256 专家，inter=2048）。

非文本权重（`visual.*` / `audio_encoder.*` / `speech_embeddings.*` / `model.mtp.*`）不在本文件登记；加载器按登记清单过滤 ckpt，天然排除。

---

## 4. KV cache 规划

**文件**：[HybridConfigCreator.cc](rtp_llm/cpp/cache/HybridConfigCreator.cc)、[BlockPoolConfigHelper.h](rtp_llm/cpp/cache/BlockPoolConfigHelper.h)、[BlockPool.cc](rtp_llm/cpp/cache/BlockPool.cc)

### 4.1 路由

`enable_hybrid_attention = True` 且未开 `enable_independent_kv_cache_pools`，所以 [CacheConfigCreator.cc](rtp_llm/cpp/cache/CacheConfigCreator.cc) 走 `HybridConfigCreator::createHybridConfig()`（不是 DSV4 的 typed-region 路径 `HybridPoolConfigCreator`）。

### 4.2 层分组：二分类 → 三分类

`splitLayersByAttentionType()` 原先只分 LINEAR / full 两类，现返回 `LayerSplit{ linear_layers, full_layers, swa_layers }` 三类。`SLIDING_WINDOW` 单独成组，GA 用 `createFullAttentionSpec()`（kv=4），SWA 用新增的 `createSwaAttentionSpec()`（kv=8，从 `swa_attention_config` 取）。

两个 spec 的每块字节数不同。TP=4、page=64 tokens、BF16：

```
GA  : local_kv=1, (192+128) × 64 × 2 × 1 = 40960 B
SWA : local_kv=2, (192+128) × 64 × 2 × 2 = 81920 B
```

### 4.3 SWA 环形 cache

滑窗层只会读最后 `window` 个 token，所以绝对位置 `p` 可以住在环形槽 `p % window`，per-request 占用不再随序列长度增长。`group_ring_blocks = window / page = 128 / 64 = 2` 块。

要求 **page 整除 window**：环把位置 `p` 映到槽 `p % W`，页边界必须与回绕点对齐。整除时任意 W 个连续位置双射到 W 个槽，2 页正好装下整窗、无空洞无残留。不整除则退回 0（按整条序列分配）并打 warning —— 把环向上取整到页倍数会在任意环位置留下残留槽，而 `window_left` 按**位置**而非**槽**掩码，掩不掉。

接线：

- [HybridConfigCreator.cc](rtp_llm/cpp/cache/HybridConfigCreator.cc) 计算 `swa_ring_blocks` 填进 `CacheConfig::group_ring_blocks`
- [HybridTypeKVCacheAllocator.cc](rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.cc) 调 `group->setRingBlocks()`
- [KVCacheGroup.h](rtp_llm/cpp/cache/KVCacheGroup.h) 提供 `setRingBlocks()` / `capToRing()`
- [FullKVCacheGroup.cc](rtp_llm/cpp/cache/FullKVCacheGroup.cc) 的 `needBlocksNum()` 用 `capToRing()` 封顶块数

SWA 组仍标 `CacheGroupType::FULL`：它保持普通 paged 语义和普通 block-cache 行为，约束占用的是 `group_ring_blocks` 而非组类型。`CacheGroupType::SWA` 选的是 `SWAKVCacheGroup`，那套「整长列表只物化尾部」的方案面向 DSV4 的固定 ring 池，不适配 64-token 页。

### 4.4 双 memory layout

两类层 stride 不同，不能共用一个 layout。`BlockPoolConfigHelper::useDualStrideLayouts()` 在「hybrid + 全是注意力组（无 LINEAR）+ 各组 stride 不唯一」时启用 `appendPerStrideLayouts()`：按 stride 类各建一个 layout（保持首次出现顺序），各组用自己 spec 的真实 stride，无 padding。

`setupPhysicalSizes()` 相应改为逐组填 `group_kv_block_stride_bytes`，并且：

- 删掉了原有的 `full >= linear` 断言 —— MiMo 的 GA stride 小于 SWA stride，该断言必然失败
- `block_size_bytes` 改为 `Σ(每组 layer_num × 该组真实 stride)`，双 stride 下反映真实总开销

### 4.5 交错层号的显式映射

GA/SWA 层号是交错的（GA 在 0,5,11,...），**无法用连续区间表达**，而原有的 layout 映射是游标式（每个 layout 吃掉连续的一段层号）。

新增 `BlockPoolConfig::explicit_layer_mapping`（`global_layer_id → {layout_idx, layout 内局部序号}`）：

- `appendPerStrideLayouts()` 生成映射表
- [BlockPool.cc](rtp_llm/cpp/cache/BlockPool.cc) 的 `applyExplicitLayerMapping()` 在游标逻辑之后覆盖默认映射；映射为空时（MTP / 普通路径）零影响
- `BlockPool::hasExplicitLayerMapping()` 供调用方判断语义：启用时直接传全局层号，未启用时沿用组内序号（物理槽）
- [KVCacheGroup.cc](rtp_llm/cpp/cache/KVCacheGroup.cc) 和 [HybridTypeKVCacheAllocator.cc](rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.cc) 各自按此分支
- MTP layout 在映射启用时补进映射表，保持连续游标语义

### 4.6 分组粒度

`usePerLayerRowLayout()` 复制了 `useDualStrideLayouts()` 的判据（它在分组之前执行，拿不到成型的 `CacheConfig`；改一处需同步另一处）。走双 layout 时**每种注意力类型只分一组**，不再按 gcd 细分：

- legacy layout 给池子开 `group_layer_num` 行、各组复用，一个 block id 被领取者完全消耗，组数免费 —— 组越小 id 越便宜、id 数按比例变多，所以 gcd 细分零成本，它存在只是为了满足「各组层数相等」的结构要求。
- per-layer-row layout 打破了这个交易：池子每个模型层一行（9 + 39），一个 id 永远占满全部 48 行，而 3 层的组只用其中 3 行 —— 48/3 = **16 倍浪费**。
- 这里也不需要各组层数相等：显式层映射下 `KVCacheGroup::init()` 按全局层号寻址层张量，而非组内下标。

### 4.7 K≠V 的块大小

[MHAKVCacheSpec.h](rtp_llm/cpp/cache/MHAKVCacheSpec.h) 的 `block_size()` 从 `2 × local_head_num_kv × size_per_head × page` 改为 `k_block_size() + v_block_size()` 两段分别计算。对称模型下 `v_size_per_head == size_per_head`，结果与原式完全一致。

---

## 5. 逐层 cache 视图：K/V 分离

**文件**：[OpDefs.h](rtp_llm/models_py/bindings/OpDefs.h)、[PyWrappedModel.h](rtp_llm/cpp/models/PyWrappedModel.h)

物理块内布局是 `[P, H, N, K_dim + V_dim]`（K 在前 V 在后）。`KVCache::getLayerCache()` 在 `v_head_dim > 0 && v_head_dim != head_dim` 时 reshape 后按最后一维 narrow 出两个视图：

```cpp
auto pool = base.reshape({kernel_block_num, heads, page, head_dim + v_head_dim});
layer_cache.k_cache = pool.narrow(3, 0, head_dim);
layer_cache.v_cache = pool.narrow(3, head_dim, v_head_dim);
```

用**交错布局 + narrow** 而不是「K 池 / V 池」两段，是因为这样 `k_cache` 和 `v_cache` 的 stride 完全一致，满足 FlashInfer paged kernel 对 K/V stride 相同的要求。

两个配套点：

- **逐层 kv 头数**。`heads` 从 `num_kv_heads_by_layer[layer_idx]` 取（GA=4/tp、SWA=8/tp），为空时回落到全局 `num_kv_heads`。该数组由 [HybridTypeKVCacheAllocator.cc](rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.cc) 按层所属组的 spec 填进 `CacheLayerLayout::layer_local_kv_head_num`，[KVCacheManager.cc](rtp_llm/cpp/cache/KVCacheManager.cc) 透传，[PyWrappedModel.h](rtp_llm/cpp/models/PyWrappedModel.h) 写进 `KVCache`。
- **kernel-block 粒度保护**。物理块存 `[H][N][K+V]`，把 N 拆成 kernel block 会让 token 轴跑到 head 轴之上，只有 `heads == 1` 时 reshape 才仍与内存一致。`kernel_blocks_per_kv_block != 1 && heads != 1` 时抛异常，而不是静默读出看似合理的垃圾。

`PyWrappedModel.h` 里 `kv_cache.v_head_dim` 用 `vSizePerHead() == size_per_head ? 0 : vSizePerHead()`，对称模型传 0，走原来的路径。

---

## 6. 前向：双 fmha 实例 + 按层路由

**文件**：[rtp_llm/models_py/model_desc/mimo_v25.py](rtp_llm/models_py/model_desc/mimo_v25.py)

### 6.1 逐层参数覆盖

`MiMoV25DecoderLayer.__init__` 先取 `config.getAttentionConfigs(tp_size)`，再按层型覆盖四项差异：

| | GA | SWA |
|---|---|---|
| `kv_head_num` | `ga_kv_head_num // tp` | `swa_kv_head_num // tp` |
| `sliding_window` | 0（不限窗） | 128 |
| `rope_config.base` | 1e7 | 1e4 |
| `add_sink_bias` | False | True |

`rope_config.dim = 64`（partial rope）两类层相同，沿用配置解析结果。

`MiMoV25Model.__init__` 断言两个 kv 头数都能被 `tp_size` 整除 —— 这正是上面整除写法与 C++ 侧 `MHAKVCacheSpec` / `createSwaAttentionSpec` 结果一致的充要条件；TP≠4 时权重加载器本来就会拒绝（§3.2），这里让它在更早、错误信息更清楚的地方失败。

### 6.2 双 fmha 实例

工厂原先只接受 `model_config`，无法为同一模型的不同层型传不同参数。[attn_factory.py](rtp_llm/models_py/modules/factory/attention/attn_factory.py) 新增 `AttnImplFactory.get_fmha_impl_with_configs()`，直接按 `AttentionConfigs` 选实现，调用方可为不同层型传各自覆盖过的 configs，而不需要构造多份 `model_config`。

`_prepare_dual_fmha()` 各建一个实例，并断言 SWA 实例具备 `set_sink_bias` —— 只有 PyFlashinfer 系实现它，而那也是唯一把 `sliding_window` 透传给 kernel 的系列，一个能力探测同时覆盖窗口和 sink 两项要求。

### 6.3 逐层 KV cache 组切换

一个 cache 组自有 block id 空间，且只持有某一注意力类型的一部分层。MiMo 的 9 个 GA 层跨 3 组、39 个 SWA 层跨 13 组，所以构造时那一次 `plan()` 只对它当时看到的组有效。

`forward()` 逐层：

```python
gid = select_block_map_for_layer(inputs.attention_inputs, i)
impl = ga_impl if self.is_ga_layer[i] else swa_impl
if gid is not None and planned_gid.get(id(impl)) != gid:
    impl.sync_layer_block_table(inputs.attention_inputs)
    planned_gid[id(impl)] = gid
if hasattr(impl, "set_sink_bias"):
    impl.set_sink_bias(decoder_layer.sink_bias)
```

`plan()` 的调度元数据只依赖序列长度和页数（跨组相同），而 `plan()` 持有的是 `page_indice_d` 的**引用**而非拷贝，所以原地重填该 buffer 就够，不必重新 plan。两个实例各有独立的 `fmha_params` buffer，因此 GA→SWA→GA 交错序列不会互相污染。

sink bias 逐层设置：SWA 层有，GA 层显式设 `None`，防止上一层的 bias 泄漏到下一层。

---

## 7. RoPE 与 KV 写入

### 7.1 双 rope base 的 cos/sin cache

**文件**：[base_rotary_embedding_op.py](rtp_llm/models_py/modules/factory/attention/cuda_impl/base_rotary_embedding_op.py)

`get_rope_cache_once` 用 `std::call_once` 维护每个 interleave 模式一份**进程级单例**，之后不论传入什么 config 都返回同一份。逐层 rope 参数不同的模型会全部拿到「最先申请的那个 config」—— MiMo 的 39 个 SWA 层会被错误地用 GA 的 theta 定位。

`_resolve_cos_sin_cache()` 只在单例确实匹配时才用它（`check_rope_cache` 加一个该辅助函数不覆盖的长度检查），否则单独构建并按 config 记忆化（key 由 style/dim/base/scale/max_pos/factor/mscale/max_position_embeddings/interleave 组成）。

### 7.2 非对称 QKV 切分

**文件**：[flashinfer_rotary_emb.py](rtp_llm/models_py/modules/factory/attention/cuda_impl/flashinfer_rotary_emb.py)

`MhaRotaryEmbeddingOp` 的三段切分从「K/V 同维」改为按各自维度：

```python
split_sizes = [
    head_size * num_heads,          # 64 × 192
    head_size * num_kv_heads,       # kv × 192
    v_head_size * num_kv_heads,     # kv × 128
]
value = v.reshape(v.shape[0], num_kv_heads, v_head_size)
```

### 7.3 KV cache 写入

**文件**：[kv_cache_write_op.py](rtp_llm/models_py/modules/factory/attention/cuda_impl/kv_cache_write_op.py)

FlashInfer 的 `append_paged_kv_cache` 要求 K/V head_dim 相等，非对称时不可用。新增一条显式 scatter 路径（`k_cache.size(-1) != v_cache.size(-1)` 时启用）：

```python
page_ids_local = pos // page_size
slot_in_page   = pos % page_size
physical_page  = block_table[batch_idx, page_ids_local]
k_cache[physical_page, :, slot_in_page, :] = key
v_cache[physical_page, :, slot_in_page, :] = value
```

两个参数配合环形 cache：

- **`block_table`**：优先用 `select_block_map_for_layer` 设的逐层 block table（并已被 FMHA op 做过环形循环），而不是 `decode_page_indptr_d` —— 后者在 prepare 时算好，可能是过期的。
- **`ring_window_tokens`**：环形层丢弃比窗口更早的 token。位置 `p` 和 `p + window` 共享同一环形槽，同一次调用里后者本来就会覆盖前者，而 PyTorch scatter 未定义谁赢。用 `scatter_reduce_(reduce="amax")` 求每个请求的最新位置再筛尾部窗口，全程不需要 host 同步。

非对称路径同时覆盖 warmup（`kv_cache is None`）分支，dummy V cache 用 `value.size(-1)` 而不是 `head_size`。

---

## 8. Attention kernel 接线

**文件**：[py_flashinfer_mha.py](rtp_llm/models_py/modules/factory/attention/cuda_impl/py_flashinfer_mha.py)

### 8.1 window_left

`attn_configs.sliding_window` 原先在框架里只写不读（唯一消费者是 DSV4 自研链路）。现在 prefill / decode 的 `plan()` 都传：

```python
window_left=window_left_from_sliding_window(self.attn_configs.sliding_window)
```

**要减 1**。HF 掩码是 `kv_idx > q_idx - sliding_window`，即含自身共 `sliding_window` 个 key；FlashInfer 是 `kv_idx + qo_len + window_left >= kv_len + qo_idx`，即含自身共 `window_left + 1` 个。直接透传会把窗口放宽一个 key。

### 8.2 环形 block table

`swa_ring_pages(attn_configs)` = `window / kernel_tokens_per_block`（不整除返回 0），与 C++ 侧 `HybridConfigCreator` 的计算保持一致。

`ring_block_table()` 把环形组的短 block 列表在整表宽度上循环重复：

```python
block_table[:, :ring_pages].repeat(1, reps)[:, :width]
```

paged-KV planner 和 KV cache 写入都按绝对页号 `pos // page_size` 索引 block table，而第 j 项填环形页 `j % ring_pages` 后，`(pos // page_size) % ring_pages` **就是** `pos` 的环形页 —— 两侧都不需要知道有环存在。分配器留在行尾的 0（block 0 是保留位，0 读作「未分配」）被循环后的环覆盖。

### 8.3 sink bias

FlashInfer 的 wrapper 接受 `sinks=` 参数，但只有 `prefill.py` 里 `paged_run` 的 `trtllm-gen` 分支真的转发给 kernel；fa2 / fa3 分支静默丢弃，它们默认的 `DefaultAttention` variant 根本没声明 sink 张量。原生支持要用 JIT-only 的 `BatchAttentionWithAttentionSinkWrapper`，其 AOT 覆盖只有 `(64, 64)`。

所以改为从 LSE 精确折算，不需要额外 kernel。sink 贡献一个额外 logit `b_h` 到 softmax 分母、不贡献 value，令 `D = Σ_j exp(s_j)`：

```
out_sink = (Σ_j exp(s_j) v_j) / (D + exp(b_h))
         = out · D / (D + exp(b_h))
         = out · sigmoid(log(D) - b_h)
```

`apply_attention_sink()` 实现为 `out * sigmoid(lse × ln2 - sink)`。两个后端报的都是 `lse = log2(D)`（fa2 写 `log2(d) + m`，`m` 已缩放进 log2 域；fa3 写 `row_max × sm_scale_log2 + log2(sum)`），故有 ln2 因子。

`normalize_sink_bias()` 把 ckpt 里的 BF16 上转 fp32（校正在 fp32 域跑，上转精确）。

### 8.4 backend 选择

`resolve_paged_backend()`：FlashInfer 的 SM90（fa3）**paged** prefill 只在 `HEAD_DIM_QK == HEAD_DIM_VO` 时 dispatch，其他情况落到 `return cudaErrorNotSupported`。`head_dim_qk_192_head_dim_vo_128 ... sm90` 的 AOT 模块确实编译了，所以失败只在 launch 时以 `BatchPrefillWithPagedKVCacheSM90Run failed with error: operation not supported` 暴露。fa3 的 **ragged** 路径没有这个限制（所以非对称 head dim 在那边一直能跑，只有层移到 paged wrapper 后才踩到）。fa2 的 paged dispatch 没有 head-dim 相等检查，因此非对称时钉在 fa2。

### 8.5 环形层的 prefill 必须走 ragged

`PyFlashinferPagedPrefillImpl.support()` 在 `swa_ring_pages > 0` 时返回 False。环只持有一个窗口的 KV，而 paged prefill 按整条序列长度 plan，会读到从未写入的环形槽。ragged impl 从 QKV 张量读 K/V、完全不碰 cache，是环形层唯一正确的 prefill。在这里显式拒绝，是为了把将来的路由变化（比如前缀复用使 `prefix_lengths` 非零，从而让 ragged impl 拒绝）变成 `cannot find mha type` 报错，而不是静默算错。

### 8.6 不支持非对称 K/V 的实现自我排除

[xqa.py](rtp_llm/models_py/modules/factory/attention/cuda_impl/xqa.py) 靠既有的 `size_per_head in [64, 128, 256]` 天然拒绝 192，无需改动。其余三个在 `support()` 里加形状检查：

```python
if attn_configs.v_size_per_head and attn_configs.v_size_per_head != attn_configs.size_per_head:
    return False
```

覆盖 [trt.py](rtp_llm/models_py/modules/factory/attention/cuda_impl/trt.py)（2 处）、[flash_infer.py](rtp_llm/models_py/modules/factory/attention/cuda_impl/flash_infer.py)（2 处）、[trtllm_gen.py](rtp_llm/models_py/modules/factory/attention/cuda_impl/trtllm_gen.py)（3 处，含 `FlashInferTRTLLMSpecDecodeImpl` —— 它注册在 `PREFILL_MHA_IMPS` 里排在 PyFlashinfer 之前）。

判据只用 `v_size_per_head`（纯形状事实、全仓只有 MiMo 会设），不读 `sliding_window` / `add_sink_bias` 这类模型标记字段 —— DSV4 也会设 `attn_config.sliding_window`（[deepseek_v4.py:605](rtp_llm/models/deepseek_v4.py#L605)）且它是 MHA 路由（`use_mla = False`），按标记字段 gate 会改变 DSV4 的实现选择。窗口/sink 的能力要求由 §6.2 在 MiMo 侧断言。

### 8.7 plan 阶段的 block table 越界保护

**文件**：[FlashInferMlaParams.cc](rtp_llm/models_py/bindings/cuda/FlashInferMlaParams.cc)

plan kernel 算 `pages_self = ceil((input_len + prefix_len) / page_size)`，必须不超过 block table 列数。SWA 层的 block table 只有 `sliding_window / page_size` 宽，未裁剪的 `input_len` 会让 `pages_self` 远超表宽。`fillParamsMhaDevice()` 为 plan 单独裁一份 `input_lengths`（有 prefix 时按 per-sample 上限 `max_seq_for_plan - prefix_len`），plan 之后再用原始未裁剪长度重填 `positions_d` —— RoPE 需要完整序列位置。

---

## 9. MoE

**文件**：[generic_moe.py](rtp_llm/models_py/model_desc/generic_moe.py)

MoE 门控与 deepseek_v2 同构，复用 `GenericMoeLayer`，无需新增专用实现：

- `scoring_func = 1`（sigmoid）经 `MoeConfigs` 传到 op
- `moe_style = 1` → `add_shared_expert = False`
- `correction_bias` 从 `W.e_score_correction_b` 读，`noaux_tc` 路由下 bias 只参与选专家，topk 权重用不含 bias 的 sigmoid 分数
- `n_group = 1` / `topk_group = 1` 使分组掩码退化为恒等
- `routed_scaling_factor` 为 null，取 1.0
- `has_moe_norm = True`（`norm_topk_prob`）

layer 0 是 dense FFN，走 `DenseMLP`（inter=16384）；layer 1~47 走 `GenericMoeLayer`（256 专家，inter=2048）。分派依据是 `layer_idx in config.moe_layer_index`。

---

## 10. 显存账目

[model_config.py](rtp_llm/config/model_config.py) 的 `_eval_kv_cache_mem_size()` 原先假设全模型同构（`2 × num_layers × kv_head_num × size_per_head`）。hybrid 时改为按层型分别累加，并区分 K/V 维度：

```python
ga_kv  = ga_layers  * kv_head_num     * (size_per_head + v_head_size)
swa_kv = swa_layers * swa_kv_head_num * (size_per_head + v_head_size)
kv_cache_size = (ga_kv + swa_kv) * kv_cache_bytes * max_seq_len
```

对 MiMo（TP 前，K+V 合计 320 维）：GA 9 层 × 4 头、SWA 39 层 × 8 头。

---

## 11. 约束与边界

| 约束 | 原因 | 检查位置 |
|---|---|---|
| **TP 必须 = 4** | FP8 ckpt 的 4 slab 是原生量化单元，FP8 域不可合并/再切 | `MiMoPerBlockFp8Weight._split()` |
| `swa_head_dim` / `swa_v_head_dim` / `swa_num_attention_heads` 须等于非 SWA 值 | 骨架用同一套 head 维度推导 qkv 行边界、o_proj 输入维、两个 KV spec 的 stride | `_parse_basic_config()` |
| `attention_projection_layout == "fused_qkv"` | 整套权重映射按单融合张量切分 | `_parse_basic_config()` |
| `add_full_attention_sink_bias` 必须为 false | GA 层不走 sink 路径 | `_parse_swa_config()` |
| 无 shared expert、`topk_method == "noaux_tc"` | FFN 权重映射与门控实现按此假设 | `_parse_moe_config()` |
| `sliding_window % page == 0` 才启用环形 cache | 环按位置取模，页边界须与回绕点对齐 | `swa_ring_pages()` / `HybridConfigCreator` |
| 非对称 K/V 下 `kernel_seq_size_per_block == seq_size_per_block`（除非 `heads == 1`） | 拆 kernel block 会调换 head/token stride | `KVCache::getLayerCache()` 抛异常 |
| SWA 层 prefill 必须走 ragged impl | 环只有一窗 KV，paged prefill 会读未写入的槽 | `PyFlashinferPagedPrefillImpl.support()` |
| SWA fmha 实现须支持窗口 + sink | 否则窗口不生效、sink 丢失 | `_prepare_dual_fmha()` 断言 |

**不在支持范围内**：视觉 / 音频塔（`visual.*`、`audio_encoder.*`、`speech_embeddings.*`）与 MTP 头（`model.mtp.*`）—— 纯文本链路，这些权重不登记。

---

## 12. 测试

[rtp_llm/test/model_test/test_mimo_v25.py](rtp_llm/test/model_test/test_mimo_v25.py)（target 见 [BUILD](rtp_llm/test/model_test/BUILD)）通过 `MagaServerManager` 起真实 server 后发 HTTP 请求，四个用例：

| 用例 | 覆盖点 |
|---|---|
| `test_text_generation_basic` | 基础生成通路 |
| `test_openai_chat_completion` | OpenAI 端点 + prompt 模板 |
| `test_long_context_needle_recall` | 长上下文，跨滑窗与 KV 页边界；只有 9 个 GA 层能承载信息，GA 与 SWA 必须行为不同 |
| `test_long_prompt_decode_crosses_kv_page` | 长 prompt 的 decode 跨 KV 页，触发新物理块分配（每个 cache 组各一个新 block id） |

`tp_size` 默认 4（`TP_SIZE` 可覆盖），与 §11 的 TP 约束一致。

---

## 13. 改动文件清单

### Python — 模型层

| 文件 | 作用 |
|---|---|
| [models/mimo_v25.py](rtp_llm/models/mimo_v25.py) | 模型类、配置解析、注册 |
| [models/mimo_v25_weight.py](rtp_llm/models/mimo_v25_weight.py) | 权重映射、FP8 slab 切分、value_scale 折叠 |
| [models_py/model_desc/mimo_v25.py](rtp_llm/models_py/model_desc/mimo_v25.py) | 前向、双 fmha、按层路由 |
| [model_factory_register.py](rtp_llm/model_factory_register.py) | 懒加载注册 |

### Python — 框架

| 文件 | 作用 |
|---|---|
| [config/model_config.py](rtp_llm/config/model_config.py) | hybrid 的 KV cache 显存估算 |
| [model_loader/attn_weight.py](rtp_llm/model_loader/attn_weight.py) | `AttnConfig.v_size_per_head` |
| [model_loader/per_block_fp8_quant_weight.py](rtp_llm/model_loader/per_block_fp8_quant_weight.py) | 量化包装分发放行 |
| [utils/model_weight.py](rtp_llm/utils/model_weight.py) | `get_sp_tensor_kv_asym`、`W.attn_sink_bias`、`is_mimo_v25_weight` |
| [ops/__init__.py](rtp_llm/ops/__init__.py) / [libth_transformer_config.pyi](rtp_llm/ops/libth_transformer_config.pyi) | `SwaAttentionConfig` 再导出与存根 |

### Python — attention

| 文件 | 作用 |
|---|---|
| [attn_factory.py](rtp_llm/models_py/modules/factory/attention/attn_factory.py) | `get_fmha_impl_with_configs` |
| [py_flashinfer_mha.py](rtp_llm/models_py/modules/factory/attention/cuda_impl/py_flashinfer_mha.py) | window_left、sink、环形 block table、backend 选择 |
| [base_rotary_embedding_op.py](rtp_llm/models_py/modules/factory/attention/cuda_impl/base_rotary_embedding_op.py) | 按 config 记忆化 cos/sin cache |
| [flashinfer_rotary_emb.py](rtp_llm/models_py/modules/factory/attention/cuda_impl/flashinfer_rotary_emb.py) | 非对称 QKV 切分 |
| [kv_cache_write_op.py](rtp_llm/models_py/modules/factory/attention/cuda_impl/kv_cache_write_op.py) | 非对称 K/V 写入、环形丢弃 |
| [xqa.py](rtp_llm/models_py/modules/factory/attention/cuda_impl/xqa.py) / [trt.py](rtp_llm/models_py/modules/factory/attention/cuda_impl/trt.py) / [trtllm_gen.py](rtp_llm/models_py/modules/factory/attention/cuda_impl/trtllm_gen.py) / [flash_infer.py](rtp_llm/models_py/modules/factory/attention/cuda_impl/flash_infer.py) | 非对称 K/V 自我排除 |

### C++ — 配置

| 文件 | 作用 |
|---|---|
| [model_utils/AttentionConfig.h](rtp_llm/cpp/model_utils/AttentionConfig.h) / [.cc](rtp_llm/cpp/model_utils/AttentionConfig.cc) | `v_size_per_head`、`add_sink_bias`、`vSizePerHead()` |
| [config/ConfigModules.h](rtp_llm/cpp/config/ConfigModules.h) / [.cc](rtp_llm/cpp/config/ConfigModules.cc) | `SwaAttentionConfig` |
| [pybind/ConfigInit.cc](rtp_llm/cpp/pybind/ConfigInit.cc) | 四处绑定 |

### C++ — KV cache

| 文件 | 作用 |
|---|---|
| [cache/HybridConfigCreator.h](rtp_llm/cpp/cache/HybridConfigCreator.h) / [.cc](rtp_llm/cpp/cache/HybridConfigCreator.cc) | 三分类分组、SWA spec、环形块数、逐组 stride |
| [cache/CacheConfig.h](rtp_llm/cpp/cache/CacheConfig.h) | `group_ring_blocks` |
| [cache/KVCacheGroup.h](rtp_llm/cpp/cache/KVCacheGroup.h) / [.cc](rtp_llm/cpp/cache/KVCacheGroup.cc) | `setRingBlocks` / `capToRing`、全局层号寻址 |
| [cache/FullKVCacheGroup.cc](rtp_llm/cpp/cache/FullKVCacheGroup.cc) | 块数封顶 |
| [cache/MHAKVCacheSpec.h](rtp_llm/cpp/cache/MHAKVCacheSpec.h) | K≠V 的块大小 |
| [cache/BlockPoolConfig.h](rtp_llm/cpp/cache/BlockPoolConfig.h) / [BlockPoolConfigHelper.h](rtp_llm/cpp/cache/BlockPoolConfigHelper.h) | 显式层映射、按 stride 建多 layout |
| [cache/BlockPool.h](rtp_llm/cpp/cache/BlockPool.h) / [.cc](rtp_llm/cpp/cache/BlockPool.cc) | `applyExplicitLayerMapping` |
| [cache/BufferTypes.h](rtp_llm/cpp/cache/BufferTypes.h) / [HybridTypeKVCacheAllocator.cc](rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.cc) / [KVCacheManager.cc](rtp_llm/cpp/cache/KVCacheManager.cc) | 逐层 kv 头数、环形接线 |
| [cache/KVCacheSpecBase.h](rtp_llm/cpp/cache/KVCacheSpecBase.h) / [HybridPoolConfigCreator.cc](rtp_llm/cpp/cache/HybridPoolConfigCreator.cc) | spec 类型枚举、typed-region 路径的 SWA spec |

### C++ — 模型与绑定

| 文件 | 作用 |
|---|---|
| [models/PyWrappedModel.h](rtp_llm/cpp/models/PyWrappedModel.h) | `v_head_dim` / `num_kv_heads_by_layer` 填充 |
| [models_py/bindings/OpDefs.h](rtp_llm/models_py/bindings/OpDefs.h) / [.cc](rtp_llm/models_py/bindings/OpDefs.cc) | K/V 分离视图与绑定 |
| [models_py/bindings/cuda/FlashInferMlaParams.cc](rtp_llm/models_py/bindings/cuda/FlashInferMlaParams.cc) | plan 阶段长度裁剪 |
