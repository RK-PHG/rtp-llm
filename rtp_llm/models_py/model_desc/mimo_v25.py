# rtp_llm/models_py/model_desc/mimo_v25.py
# MiMo V2.5 forward implementation.
# The core difference from the qwen3 / generic_moe templates is dual fmha instances plus
# per-layer routing: the GA (global attention) and SWA (sliding-window attention) layer
# kinds differ in kv head count, window, rope theta and sink bias, so one fmha instance is
# built for each and selected per layer according to the layer kind.
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    RMSNorm,
)
from rtp_llm.models_py.modules.factory.attention.attn_factory import AttnImplFactory
from rtp_llm.ops import (
    HWKernelConfig,
    HybridAttentionType,
    MoeConfig,
    ParallelismConfig,
)
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class MiMoV25DecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        layer_idx: int,
        weights: Dict[str, torch.Tensor],
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        is_ga: bool = False,
    ):
        super().__init__()
        self.is_ga = is_ga
        swa_cfg = config.hybrid_attention_config.swa_attention_config
        tp_size = parallelism_config.get_attn_tp_size()
        attn_configs = config.getAttentionConfigs(tp_size)
        # Per-layer overrides for the parameters that differ: kv_head / window /
        # rope theta / sink. rope_config.dim = 64 (partial rope) is the same for both layer
        # kinds, so the value parsed at config time is reused as-is.
        if is_ga:
            attn_configs.kv_head_num = swa_cfg.ga_kv_head_num // tp_size  # 4 / tp
            attn_configs.sliding_window = 0  # GA is not windowed
            attn_configs.rope_config.base = int(
                config.attn_config.rope_config.base
            )  # 1e7
            attn_configs.add_sink_bias = False
        else:
            attn_configs.kv_head_num = swa_cfg.swa_kv_head_num // tp_size  # 8 / tp
            attn_configs.sliding_window = swa_cfg.window_size  # 128
            attn_configs.rope_config.base = int(swa_cfg.swa_rope_theta)  # 1e4
            attn_configs.add_sink_bias = swa_cfg.add_sink_bias  # True
        self.attn_configs = attn_configs

        self.self_attn = CausalAttention(
            attn_configs,
            parallelism_config,
            weights,
            config.layernorm_eps,
            quant_config,
            hw_kernel_config,
            layer_idx,
        )
        # SWA layers carry a per-head sink bias; GA layers do not.
        self.sink_bias: Optional[torch.Tensor] = weights.get(W.attn_sink_bias, None)
        # Layer 0 is a dense FFN (inter=16384); the rest are 256-expert MoE layers
        # (noaux_tc routing, with the correction_bias logic handled inside GenericMoeLayer)
        if layer_idx in config.moe_layer_index:
            self.mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=enable_cuda_graph,
                hw_kernel_config=hw_kernel_config,
            )
        else:
            self.mlp = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                quant_config,
                hw_kernel_config,
            )
        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states, fmha_impl=fmha_impl, kv_cache=kv_cache
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class MiMoV25Model(GptModelBase):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        quant_config: Optional[object] = None,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        types = config.hybrid_attention_config.hybrid_attention_types
        self.is_ga_layer: List[bool] = [
            t != HybridAttentionType.SLIDING_WINDOW for t in types
        ]

        # The fused-qkv FP8 ckpt is natively sliced into 4 slabs, so MiMoPerBlockFp8Weight
        # only accepts tp_size == 4 (see models/mimo_v25_weight.py). Both kv head counts
        # divide evenly at that size, which is what lets the per-layer override below use
        # plain floor division. Assert the precondition rather than carrying a
        # non-divisible fallback that the weight loader would reject anyway.
        swa_cfg = config.hybrid_attention_config.swa_attention_config
        tp_size = parallelism_config.get_attn_tp_size()
        for name, kv_head_num in (
            ("ga_kv_head_num", swa_cfg.ga_kv_head_num),
            ("swa_kv_head_num", swa_cfg.swa_kv_head_num),
        ):
            assert (
                kv_head_num % tp_size == 0
            ), f"{name}={kv_head_num} must be divisible by attn tp_size={tp_size}"

        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )

        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.layers = nn.ModuleList(
            [
                MiMoV25DecoderLayer(
                    config,
                    parallelism_config,
                    idx,
                    weights.weights[idx],
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    quant_config=quant_config,
                    hw_kernel_config=py_hw_kernel_config,
                    is_ga=self.is_ga_layer[idx],
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

    def _prepare_dual_fmha(self, inputs: PyModelInputs):
        """Build the GA and SWA fmha instances (each with its own kv head count, window,
        rope theta and sink).

        Returns (ga_impl, swa_impl, planned_gid). planned_gid records the KV cache group id
        that plan() was called with when each instance was created; during per-layer
        routing in forward() it tells us whether the page table has to be switched to the
        group the current layer belongs to -- see forward().
        """
        ga_idx = self.is_ga_layer.index(True)
        swa_idx = self.is_ga_layer.index(False)
        common_kwargs = dict(
            fmha_config=self.fmha_config,
            quant_config=self.config.quant_config,
            max_seq_len=self.config.max_seq_len,
            headwise_config=getattr(self.config, "headwise_config", None),
            is_cuda_graph=False,
        )
        ga_gid = select_block_map_for_layer(inputs.attention_inputs, ga_idx)
        ga_impl = AttnImplFactory.get_fmha_impl_with_configs(
            self.layers[ga_idx].attn_configs,
            self.parallelism_config,
            self.weight,
            inputs.attention_inputs,
            **common_kwargs,
        )
        swa_gid = select_block_map_for_layer(inputs.attention_inputs, swa_idx)
        swa_impl = AttnImplFactory.get_fmha_impl_with_configs(
            self.layers[swa_idx].attn_configs,
            self.parallelism_config,
            self.weight,
            inputs.attention_inputs,
            **common_kwargs,
        )
        # The SWA layers need an impl that both honours window_left and takes a per-head
        # sink bias. Rather than having the shared impls opt out on MiMo's config fields
        # (which would also change dispatch for any other model that sets them), check the
        # capability here: set_sink_bias is implemented only by the PyFlashinfer family,
        # which is also the only family that forwards sliding_window to the kernel.
        assert hasattr(swa_impl, "set_sink_bias"), (
            f"MiMo SWA layers require a sliding-window + attention-sink capable fmha impl, "
            f"got {type(swa_impl).__name__}"
        )

        planned_gid = {id(ga_impl): ga_gid, id(swa_impl): swa_gid}
        return ga_impl, swa_impl, planned_gid

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        hidden_states = self.embed_tokens(input_ids)
        ga_impl, swa_impl, planned_gid = self._prepare_dual_fmha(inputs)

        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            gid = select_block_map_for_layer(inputs.attention_inputs, i)
            impl = ga_impl if self.is_ga_layer[i] else swa_impl
            # A KV cache group owns its own block-id space and holds only a slice of
            # the layers of one attention type (MiMo's 9 GA layers span 3 groups, its
            # 39 SWA layers span 13), so the single plan() this impl did at
            # construction is only valid for the group it saw. Re-point it whenever
            # the group changes; the plan's schedule depends on sequence lengths, not
            # on page ids, so refilling the page table in place is enough.
            if gid is not None and planned_gid.get(id(impl)) != gid:
                impl.sync_layer_block_table(inputs.attention_inputs)
                planned_gid[id(impl)] = gid
            # Per-layer sink bias: SWA layers carry one, GA layers must get None so a
            # previous layer's bias cannot leak forward.
            if hasattr(impl, "set_sink_bias"):
                impl.set_sink_bias(decoder_layer.sink_bias)
            hidden_states = decoder_layer(
                hidden_states,
                impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, ga_impl.fmha_params)


__all__ = [
    "MiMoV25Model",
]
