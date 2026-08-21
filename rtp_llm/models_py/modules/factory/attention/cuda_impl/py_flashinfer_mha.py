import math
from typing import Any, Optional

import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    MhaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
    KVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.utils import is_sm_100
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
    check_attention_inputs,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import (
    AttentionConfigs,
    FMHAType,
    KvCacheDataType,
    ParallelismConfig,
    RopeStyle,
)
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    LayerKVCache,
    ParamsBase,
    PyAttentionInputs,
    fill_mla_params,
    get_scalar_type,
    rtp_llm_ops,
)

# Constants
DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_MB = 128


def window_left_from_sliding_window(sliding_window: int) -> int:
    """Translate a HuggingFace ``sliding_window`` into FlashInfer's ``window_left``.

    HF masks with ``kv_idx > q_idx - sliding_window``, i.e. a query attends to
    ``sliding_window`` keys including itself. FlashInfer masks with
    ``kv_idx + qo_len + window_left >= kv_len + qo_idx``, i.e. ``window_left + 1``
    keys including itself. Passing ``sliding_window`` straight through widens the
    window by one key.
    """
    if sliding_window <= 0:
        return -1
    return sliding_window - 1


def resolve_paged_backend(backend: str, head_dim_qk: int, head_dim_vo: int) -> str:
    """Pick a paged-KV backend that can handle this head-dim pair.

    FlashInfer's SM90 (fa3) paged prefill only dispatches when
    ``HEAD_DIM_QK == HEAD_DIM_VO``; anything else falls through to
    ``return cudaErrorNotSupported`` (``hopper/prefill_sm90.cuh``,
    ``BatchPrefillWithPagedKVCacheDispatched``). The AOT module for
    ``head_dim_qk_192_head_dim_vo_128 ... sm90`` does get compiled, so the failure only
    surfaces at launch as "BatchPrefillWithPagedKVCacheSM90Run failed with error:
    operation not supported".

    The fa3 *ragged* path has no such restriction, which is why asymmetric head dims
    worked there and this only bites once a layer moves to the paged wrapper. fa2's
    paged dispatch carries no head-dim equality check, so pin it — the same reason
    ``PyFlashinferDecodeAttnOp`` hardcodes ``backend="fa2"``.
    """
    if backend == "auto" and head_dim_qk != head_dim_vo:
        return "fa2"
    return backend


def normalize_sink_bias(bias: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Put a per-head attention sink bias into the dtype the correction needs.

    The checkpoint stores the bias in the model dtype (bf16 for MiMo V2.5) and the
    correction below runs in fp32; the upcast is exact, so this only fixes the type.
    """
    if bias is None:
        return None
    if bias.dtype != torch.float32:
        bias = bias.to(torch.float32)
    return bias.contiguous()


# ln 2, converting FlashInfer's base-2 LSE into a natural log.
_LN2 = math.log(2.0)


def apply_attention_sink(
    out: torch.Tensor, lse: torch.Tensor, sink: torch.Tensor
) -> torch.Tensor:
    """Fold a per-head attention sink into an output computed without one.

    FlashInfer's wrappers accept a ``sinks=`` argument but only the ``trtllm-gen``
    branch of ``prefill.py``'s ``paged_run`` shim forwards it to the kernel; the fa2
    and fa3 branches drop it silently, and their default ``DefaultAttention`` variant
    declares no sink tensor at all. Applying sinks natively would mean the JIT-only
    ``BatchAttentionWithAttentionSinkWrapper``, whose AOT coverage is (64, 64) only.
    Doing it here instead is exact and needs no extra kernel.

    A sink contributes one extra logit ``b_h`` to the softmax denominator and no
    value, so with ``D = sum_j exp(s_j)`` over the visible keys:

        out_sink = (sum_j exp(s_j) v_j) / (D + exp(b_h))
                 = out * D / (D + exp(b_h))
                 = out * sigmoid(log(D) - b_h)

    Both backends report ``lse = log2(D)`` — fa2 writes ``log2(d) + m`` with ``m``
    already scaled into the log2 domain (``prefill.cuh``), fa3 writes
    ``row_max * sm_scale_log2 + log2(sum)`` (``hopper/attention_updater.cuh``) —
    hence the ln2 factor.

    Args:
        out: attention output, ``[tokens, num_qo_heads, head_dim_vo]``
        lse: base-2 log-sum-exp, ``[tokens, num_qo_heads]``
        sink: per-head sink bias, ``[num_qo_heads]``, fp32
    """
    scale = torch.sigmoid(lse.float() * _LN2 - sink.view(1, -1))
    return out * scale.unsqueeze(-1).to(out.dtype)


def swa_ring_pages(attn_configs: AttentionConfigs) -> int:
    """Number of KV pages a sliding-window layer needs, or 0 for "not a ring".

    A window of W tokens over pages of P tokens needs exactly R = W/P pages, and only
    if P divides W: the ring maps absolute position p to slot ``p % W``, so page
    boundaries must line up with the wrap point. When they do, any W consecutive
    positions map bijectively onto the W slots, which is why R pages hold exactly the
    window with no gaps and nothing stale — see ring_block_table().

    Returns 0 when W is not a multiple of P, which falls back to allocating the whole
    sequence. Rounding the ring up to a page multiple would leave stale slots at
    arbitrary ring positions, and ``window_left`` cannot mask those out because it
    masks by position, not by slot.
    """
    window = attn_configs.sliding_window
    page = attn_configs.kernel_tokens_per_block
    if window <= 0 or page <= 0 or window % page != 0:
        return 0
    return window // page


def ring_block_table(
    block_table: Optional[torch.Tensor], ring_pages: int
) -> Optional[torch.Tensor]:
    """Repeat a ring group's short block list across the full table width.

    A windowed layer holds only ``ring_pages`` physical blocks, but both the paged-KV
    planner and the KV-cache write index the table by absolute page number
    ``pos // page_size``. Filling entry j with ring page ``j % ring_pages`` makes that
    index land on the correct ring page without either side needing to know a ring is
    involved, because ``(pos // page_size) % ring_pages`` *is* the ring page of pos.

    The allocator leaves the tail of the row at 0 (block 0 is reserved, so 0 reads as
    "unallocated"); this overwrites it with the cycled ring.
    """
    if ring_pages <= 0 or block_table is None or block_table.dim() != 2:
        return block_table
    width = block_table.size(1)
    if width <= ring_pages:
        return block_table
    reps = (width + ring_pages - 1) // ring_pages
    return block_table[:, :ring_pages].repeat(1, reps)[:, :width].contiguous()


def has_paged_kv_cache(attn_inputs: PyAttentionInputs) -> bool:
    """Whether a real paged KV cache backs this forward.

    False during ``NormalEngine::prefillWarmUp()``, where the cache manager does not
    exist yet and every layer is handed ``kv_cache=None``. Implementations that read
    K/V from the cache cannot run in that state and must decline the layer.
    """
    block_id = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
    return block_id is not None and block_id.numel() > 0


# Global workspace buffer pool
_g_py_flashinfer_workspace_pool: list[torch.Tensor] = []
_g_py_flashinfer_pool_lock = __import__("threading").Lock()


def get_py_flashinfer_workspace_buffer(device: str = "cuda") -> torch.Tensor:
    """Get a PyFlashInfer workspace buffer from the pool.

    This function manages workspace buffers to support multiple concurrent instances.
    """
    with _g_py_flashinfer_pool_lock:
        if _g_py_flashinfer_workspace_pool:
            return _g_py_flashinfer_workspace_pool.pop()
    return torch.zeros(
        DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_MB * 1024 * 1024,
        dtype=torch.uint8,
        device=device,
    )


def release_py_flashinfer_workspace_buffer(buffer: torch.Tensor) -> None:
    """Release a PyFlashInfer workspace buffer back to the pool."""
    with _g_py_flashinfer_pool_lock:
        _g_py_flashinfer_workspace_pool.append(buffer)


class PyFlashinferPrefillPagedAttnOp(object):
    """FlashInfer Prefill Attention Op with Paged KV Cache support"""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        backend: str = "auto",
    ) -> None:
        self.g_workspace_buffer = get_py_flashinfer_workspace_buffer()
        self.attn_configs = attn_configs
        self.sink_bias = None
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.head_dim_vo = attn_configs.v_size_per_head or attn_configs.size_per_head
        self.page_size = attn_configs.kernel_tokens_per_block
        self.datatype = attn_configs.dtype
        self.max_seq_len = attn_configs.max_seq_len
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.enable_cuda_graph = attn_inputs.is_cuda_graph
        self.prefill_cuda_graph_copy_params = None
        self.ring_pages = swa_ring_pages(attn_configs)
        self.window_tokens = max(attn_configs.sliding_window, 0)
        self.block_table: Optional[torch.Tensor] = None
        # Pre-allocated buffers for CUDA graph copy path (avoid per-forward allocation)
        self._aligned_q_buf = None
        self._compact_out_buf = None
        # Use Paged KV Cache wrapper
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            "HND",
            backend=resolve_paged_backend(backend, self.head_dim_qk, self.head_dim_vo),
        )

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_sink_bias(self, bias):
        """Set per-layer sink bias for attention sink support."""
        self.sink_bias = normalize_sink_bias(bias)

    def set_params(self, params: Any):
        """Set the params object to be used by this op."""
        self.fmha_params = params

    def _fill_params(
        self, attn_inputs: PyAttentionInputs, forbid_realloc: bool = False
    ) -> None:
        """Fill FlashInfer paged-KV plus batch/position metadata on device."""
        kv_block_id = ring_block_table(
            attn_inputs.kv_cache_kernel_block_id_device, self.ring_pages
        )
        if kv_block_id is None or kv_block_id.numel() == 0:
            batch_size = attn_inputs.input_lengths.size(0)
            kv_block_id = torch.zeros(
                (batch_size, 1),
                dtype=torch.int32,
                device=attn_inputs.input_lengths.device,
            )
        self.block_table = kv_block_id
        self.fmha_params.fill_params_mha_device(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            kv_block_id,
            self.page_size,
            forbid_realloc,
        )

    def sync_layer_block_table(self, attn_inputs: PyAttentionInputs) -> None:
        """Re-point the planned page table at the current layer's cache group.

        Hybrid models hand out one block-id table per KV cache group, and a group
        holds only a subset of the layers of its attention type, so a single
        ``plan()`` cannot serve every layer that shares this op. The plan's
        scheduling metadata only depends on sequence lengths and page counts,
        which are identical across groups, and ``plan()`` keeps a *reference* to
        ``page_indice_d`` rather than a copy — so refilling that buffer in place is
        enough, no re-plan required.
        """
        self._fill_params(attn_inputs)

    def prepare(
        self,
        attn_inputs: PyAttentionInputs,
        forbid_realloc: bool = False,
    ) -> ParamsBase:
        """
        Prepare the prefill wrapper with paged KV cache parameters.

        forbid_realloc: True only when called from prepare_cuda_graph (replay); forbids buffer realloc.
        """
        check_attention_inputs(attn_inputs)
        self._fill_params(attn_inputs, forbid_realloc)
        # Store CUDA graph copy parameters
        # Define qo_indptr early for CUDA graph initialization
        if attn_inputs.prefill_cuda_graph_copy_params is not None:
            # For CUDA graph mode, create a buffer that will be filled later
            self.input_lengths = attn_inputs.input_lengths
            self.cu_seq_lens = attn_inputs.cu_seqlens
            qo_indptr = attn_inputs.cu_seqlens.clone()
        else:
            qo_indptr = attn_inputs.cu_seqlens[: attn_inputs.input_lengths.size(0) + 1]

        if self.enable_cuda_graph and self.prefill_wrapper._qo_indptr_buf is None:
            self.prefill_wrapper._use_cuda_graph = True
            self.prefill_wrapper._qo_indptr_buf = qo_indptr
            self.prefill_wrapper._paged_kv_indptr_buf = (
                self.fmha_params.decode_page_indptr_d
            )
            self.prefill_wrapper._paged_kv_last_page_len_buf = (
                self.fmha_params.paged_kv_last_page_len_d
            )
            self.prefill_wrapper._paged_kv_indices_buf = self.fmha_params.page_indice_d
            self.prefill_wrapper._fixed_batch_size = len(attn_inputs.cu_seqlens) - 1
            if attn_inputs.prefill_cuda_graph_copy_params is not None:
                self.prefill_cuda_graph_copy_params = (
                    attn_inputs.prefill_cuda_graph_copy_params
                )
                # input_lengths and cu_seq_lens were already set above
                self.qo_indptr = qo_indptr
                # Fill with cumulative sequence: [0, max_seq_len, 2*max_seq_len, ...]
                self.qo_indptr.copy_(
                    torch.arange(
                        self.qo_indptr.size(0),
                        device=self.qo_indptr.device,
                        dtype=self.qo_indptr.dtype,
                    )
                    * self.prefill_cuda_graph_copy_params.max_seq_len
                )

        # Update buffers for subsequent calls if in CUDA graph mode
        if self.prefill_cuda_graph_copy_params is not None:
            assert attn_inputs.prefill_cuda_graph_copy_params is not None
            assert self.input_lengths is not None
            assert self.cu_seq_lens is not None
            self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size[0] = (
                attn_inputs.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size
            )
            self.input_lengths[: attn_inputs.input_lengths.size(0)] = (
                attn_inputs.input_lengths
            )
            self.cu_seq_lens[: attn_inputs.cu_seqlens.size(0)] = attn_inputs.cu_seqlens
            # Build qo_indptr matching the padded Q layout produced by small2large copy.
            # Each batch's Q tokens sit at [i*max_seq_len, i*max_seq_len + input_len_i)
            # in the padded buffer, so qo_indptr[i] = i*max_seq_len, but we set
            # qo_indptr[i+1] = i*max_seq_len + input_len_i to tell FlashInfer the
            # exact number of real tokens per batch (avoiding padding token processing
            # which causes numerical differences).
            batch_size = attn_inputs.input_lengths.size(0)
            max_sl = self.prefill_cuda_graph_copy_params.max_seq_len
            offsets = (
                torch.arange(
                    batch_size, device=self.qo_indptr.device, dtype=self.qo_indptr.dtype
                )
                * max_sl
            )
            self.qo_indptr[0] = 0
            self.qo_indptr[1 : batch_size + 1] = offsets + attn_inputs.input_lengths.to(
                self.qo_indptr.device
            )
            qo_indptr = self.qo_indptr

        self.prefill_wrapper.plan(
            qo_indptr,
            self.fmha_params.decode_page_indptr_d,
            self.fmha_params.page_indice_d,
            self.fmha_params.paged_kv_last_page_len_d,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.page_size,
            causal=True,
            q_data_type=self.datatype,
            kv_data_type=self.datatype,
            head_dim_vo=self.head_dim_vo,
            window_left=window_left_from_sliding_window(
                self.attn_configs.sliding_window
            ),
        )
        return self.fmha_params

    @staticmethod
    def support(attn_inputs: PyAttentionInputs) -> bool:
        return True

    def _get_paged_kv_cache(self, kv_cache: "LayerKVCache"):
        """Get paged KV cache in the correct format for FlashInfer run().

        For asymmetric K/V (head_dim_qk != head_dim_vo), returns a tuple
        (k_cache, v_cache) so FlashInfer reads K and V with their respective
        head dimensions. For symmetric K/V, returns the merged 5D tensor.
        """
        if self.head_dim_vo != self.head_dim_qk:
            # Asymmetric path: use pre-split k_cache/v_cache from C++ layer
            k_cache_sep = getattr(kv_cache, "k_cache", None)
            if k_cache_sep is not None and k_cache_sep.numel() > 0:
                return (k_cache_sep, kv_cache.v_cache)
        # Symmetric fallback: merged 5D tensor
        paged_kv_cache = kv_cache.kv_cache_base
        if paged_kv_cache.dim() == 2:
            paged_kv_cache = common.reshape_paged_kv_cache(
                paged_kv_cache, self.local_kv_head_num, self.page_size, self.head_dim_qk
            )
        return paged_kv_cache

    def forward(
        self, q: torch.Tensor, kv_cache: Optional[LayerKVCache]
    ) -> torch.Tensor:
        """
        Forward pass with paged KV cache

        Args:
            q: Query tensor [total_tokens, num_heads, head_dim]
            kv_cache: Paged KV cache [num_pages, 2, page_size, kv_heads, head_dim]
            params: Parameters (not used currently)

        Returns:
            output: [total_tokens, num_heads, head_dim]
        """
        from rtp_llm.ops.compute_ops import (
            cuda_graph_copy_large2small,
            cuda_graph_copy_small2large,
        )

        assert kv_cache is not None, "kv_cache is required for paged attention"
        assert (
            q.dim() == 3
        ), f"Expected q to be 3D tensor [total_tokens, num_heads, head_dim], got {q.dim()}D"

        paged_kv_cache = self._get_paged_kv_cache(kv_cache)
        # CUDA graph copy logic for prefill
        if self.prefill_cuda_graph_copy_params:
            assert (
                self.input_lengths is not None
            ), "input_lengths is required for CUDA graph copy"
            assert (
                self.cu_seq_lens is not None
            ), "cu_seq_lens is required for CUDA graph copy"

            # Reshape from 3D [token_num, head_num, head_size] to 2D [token_num, hidden_size]
            token_num, head_num, head_size = q.shape
            hidden_size = head_num * head_size
            # Output hidden size uses head_dim_vo (may differ from head_dim_qk)
            out_head_size = self.head_dim_vo
            out_hidden_size = head_num * out_head_size

            # Pre-allocate buffers on first use (avoid per-forward GPU allocation)
            total_len = (
                self.prefill_cuda_graph_copy_params.max_seq_len
                * self.prefill_cuda_graph_copy_params.max_batch_size
            )
            if self._aligned_q_buf is None or self._aligned_q_buf.shape != (
                total_len,
                hidden_size,
            ):
                self._aligned_q_buf = torch.zeros(
                    (total_len, hidden_size), dtype=q.dtype, device=q.device
                )
            if self._compact_out_buf is None or self._compact_out_buf.shape != (
                token_num,
                out_hidden_size,
            ):
                self._compact_out_buf = torch.zeros(
                    (token_num, out_hidden_size), dtype=q.dtype, device=q.device
                )

            q_2d = q.view(token_num, hidden_size).contiguous()
            self._aligned_q_buf.zero_()

            # Copy small to large (compact -> aligned)
            cuda_graph_copy_small2large(
                q_2d,
                self._aligned_q_buf,
                self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size,
                self.prefill_cuda_graph_copy_params.max_batch_size,
                self.prefill_cuda_graph_copy_params.max_seq_len,
                self.input_lengths,
                hidden_size,
                self.cu_seq_lens,
            )

            # Reshape back to 3D for FlashInfer
            q_aligned = self._aligned_q_buf.view(total_len, head_num, head_size)

            result = self._run_with_sink(q_aligned, paged_kv_cache)

            # Reshape result to 2D for copy back (ensure contiguous)
            result_2d = result.view(total_len, out_hidden_size).contiguous()
            self._compact_out_buf.zero_()

            # Copy large to small (aligned -> compact)
            cuda_graph_copy_large2small(
                result_2d,
                self._compact_out_buf,
                self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size,
                self.prefill_cuda_graph_copy_params.max_batch_size,
                self.prefill_cuda_graph_copy_params.max_seq_len,
                self.input_lengths,
                out_hidden_size,
                self.cu_seq_lens,
            )

            # Reshape back to 3D
            result = self._compact_out_buf.view(token_num, head_num, out_head_size)
        else:
            # No CUDA graph copy, direct execution
            result = self._run_with_sink(q, paged_kv_cache)

        return result

    def _run_with_sink(self, q: torch.Tensor, paged_kv_cache: Any) -> torch.Tensor:
        """Run the wrapper, folding in the attention sink when this layer has one.

        Not passing ``sinks=`` to run(): fa2/fa3 accept and discard it, see
        apply_attention_sink().
        """
        if self.sink_bias is None:
            return self.prefill_wrapper.run(q, paged_kv_cache)
        out, lse = self.prefill_wrapper.run(q, paged_kv_cache, return_lse=True)
        return apply_attention_sink(out, lse, self.sink_bias)


class PyFlashinferPrefillAttnOp(object):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        backend: str = "auto",
    ) -> None:
        self.g_workspace_buffer = get_py_flashinfer_workspace_buffer()
        self.attn_configs = attn_configs
        # attn_configs.head_num and kv_head_num are already divided by tp_size in ModelConfig::getAttentionConfigs
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head  # 192 for MiMo
        self.page_size = attn_configs.kernel_tokens_per_block
        self.head_dim_vo = (
            attn_configs.v_size_per_head or attn_configs.size_per_head
        )  # 128 for MiMo
        self.prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self.g_workspace_buffer,
            backend=backend,
        )
        self.datatype = attn_configs.dtype
        self.params = None
        self.sink_bias = None
        self.ring_pages = swa_ring_pages(attn_configs)
        self.window_tokens = max(attn_configs.sliding_window, 0)
        self.block_table: Optional[torch.Tensor] = None

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_sink_bias(self, bias):
        """Set per-layer sink bias for attention sink support.

        The ragged wrapper's ``run(..., return_lse=True)`` returns the same
        ``(out, lse)`` pair the paged one does, and apply_attention_sink() only needs
        those two plus the bias — it does not care whether K/V came from a cache or
        from a dense tensor. So sinks work here exactly as on the paged path.
        """
        self.sink_bias = normalize_sink_bias(bias)

    def set_params(self, params: ParamsBase):
        """Set the params object to be used by this op."""
        self.params = params

    def _fill_params(self, attn_inputs: PyAttentionInputs) -> None:
        batch_size = attn_inputs.input_lengths.size(0)
        # Ragged prefill also uses the device metadata planner. Attention itself reads
        # K/V from the QKV tensor, so this metadata only drives RoPE positions and the
        # KV-cache write — which is why a windowed layer's prefill never needs the
        # cache to hold more than a window.
        kv_block_id = ring_block_table(
            attn_inputs.kv_cache_kernel_block_id_device, self.ring_pages
        )
        if kv_block_id is None or kv_block_id.numel() == 0:
            kv_block_id = torch.zeros(
                (batch_size, 1),
                dtype=torch.int32,
                device=attn_inputs.input_lengths.device,
            )
        self.block_table = kv_block_id
        self.params.fill_params_mha_device(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            kv_block_id,
            self.page_size,
        )

    def sync_layer_block_table(self, attn_inputs: PyAttentionInputs) -> None:
        """Refill device metadata for the current layer's cache group.

        Ragged attention reads K/V from the QKV tensor, so the page table only
        matters to the KV-cache write that runs alongside it.
        """
        self._fill_params(attn_inputs)

    def prepare(self, attn_inputs: PyAttentionInputs) -> ParamsBase:
        """
        Prepare the prefill wrapper

        Args:
            attn_inputs: Attention inputs containing sequence information
        """
        batch_size = attn_inputs.input_lengths.size(0)
        cu_seqlens = attn_inputs.cu_seqlens[: batch_size + 1]

        self._fill_params(attn_inputs)

        self.prefill_wrapper.plan(
            cu_seqlens,
            cu_seqlens,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.head_dim_vo,
            causal=True,
            q_data_type=get_scalar_type(attn_inputs.dtype),
            window_left=window_left_from_sliding_window(
                self.attn_configs.sliding_window
            ),
        )
        return self.params if self.params is not None else ParamsBase()

    @staticmethod
    def support(attn_inputs: PyAttentionInputs) -> bool:
        return (
            attn_inputs.prefix_lengths.numel() <= 0
            or attn_inputs.prefix_lengths.sum().item() == 0
        )

    ## 1. pure prefill attn: qkv contains q and k,v
    ## 2. paged attn: qkv is only q, and kv is in kv_cache
    def forward(
        self, qkv: torch.Tensor, kv_cache: Optional[LayerKVCache]
    ) -> torch.Tensor:
        qkv = qkv.reshape(qkv.shape[0], -1)
        q, k, v = torch.split(
            qkv,
            [
                self.head_dim_qk * self.local_head_num,
                self.head_dim_qk * self.local_kv_head_num,
                self.head_dim_vo * self.local_kv_head_num,
            ],
            dim=-1,
        )
        q = q.reshape(q.shape[0], self.local_head_num, self.head_dim_qk)
        k = k.reshape(k.shape[0], self.local_kv_head_num, self.head_dim_qk)
        v = v.reshape(v.shape[0], self.local_kv_head_num, self.head_dim_vo)
        if self.sink_bias is None:
            return self.prefill_wrapper.run(q, k, v)
        out, lse = self.prefill_wrapper.run(q, k, v, return_lse=True)
        return apply_attention_sink(out, lse, self.sink_bias)


class PyFlashinferPrefillImplBase(FMHAImplBase):
    """Base class for FlashInfer prefill implementations (Ragged and Paged)."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        """Initialize prefill implementation with common setup.

        Args:
            attn_configs: Attention configuration
            attn_inputs: Attention inputs
        """
        # Store configs and inputs
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs

        self.fmha_impl = self._create_fmha_impl(attn_configs, attn_inputs)
        self.rope_impl = self._create_rope_impl(attn_configs)
        # Create KV cache write op
        self.kv_cache_write_op = KVCacheWriteOp(
            num_kv_heads=attn_configs.kv_head_num,
            head_size=attn_configs.size_per_head,
            token_per_block=attn_configs.kernel_tokens_per_block,
        )
        self.create_params(attn_inputs)
        self.fmha_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        self.fmha_impl.prepare(attn_inputs, forbid_realloc=True)

    def set_sink_bias(self, bias):
        """Set per-layer sink bias for attention sink support."""
        self.fmha_impl.set_sink_bias(bias)

    def sync_layer_block_table(self, attn_inputs: PyAttentionInputs) -> None:
        """Point this impl at the block table currently selected on attn_inputs."""
        self.fmha_impl.sync_layer_block_table(attn_inputs)

    def create_params(self, attn_inputs: PyAttentionInputs):
        """Create FlashInfer MLA attention parameters.

        Similar to MLA implementation, this creates and initializes the params
        that will be used for both FMHA and RoPE operations.
        """
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.rope_params = self.fmha_params
        # Pass the shared params to all ops
        self.fmha_impl.set_params(self.fmha_params)
        if self.rope_impl is not None:
            self.rope_impl.set_params(self.rope_params)
        # KV cache write always needs params (even without RoPE)
        self.kv_cache_write_op.set_params(self.rope_params)

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        """Create FMHA implementation. To be overridden by subclasses."""
        raise NotImplementedError("Subclass must implement _create_fmha_impl")

    def _create_rope_impl(self, attn_configs: AttentionConfigs) -> Any:
        """Create RoPE implementation. To be overridden by subclasses."""
        raise NotImplementedError("Subclass must implement _create_rope_impl")

    def _split_qkv(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split QKV tensor into query, key, value.

        Args:
            qkv: QKV tensor [total_tokens, (num_heads + 2*num_kv_heads) * head_dim]

        Returns:
            Tuple of (query, key, value) tensors
        """
        qkv = qkv.reshape(qkv.shape[0], -1)
        num_heads = self.attn_configs.head_num
        num_kv_heads = self.attn_configs.kv_head_num
        head_dim = self.attn_configs.size_per_head
        head_dim_vo = self.attn_configs.v_size_per_head or head_dim

        q, k, v = torch.split(
            qkv,
            [
                head_dim * num_heads,
                head_dim * num_kv_heads,
                head_dim_vo * num_kv_heads,
            ],
            dim=-1,
        )

        query = q.reshape(q.shape[0], num_heads, head_dim)
        key = k.reshape(k.shape[0], num_kv_heads, head_dim)
        value = v.reshape(v.shape[0], num_kv_heads, head_dim_vo)

        return query, key, value

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Common forward implementation for all prefill implementations."""
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            if self.rope_impl is not None:
                # Apply RoPE and get Q, K, V
                query, key, value = self.rope_impl.forward(qkv)
            else:
                # No RoPE, just split QKV
                query, key, value = self._split_qkv(qkv)

            # Write KV to cache through the block table the FMHA op resolved — for a
            # windowed layer that is the ring-cycled view, so absolute page indices
            # land on ring pages. ring_window_tokens makes the writer drop tokens
            # older than the window, which both saves bandwidth and avoids two
            # positions colliding on one ring slot within a single prefill.
            self.kv_cache_write_op.forward(
                key,
                value,
                kv_cache,
                block_table=getattr(self.fmha_impl, "block_table", None),
                ring_window_tokens=(
                    getattr(self.fmha_impl, "window_tokens", 0)
                    if getattr(self.fmha_impl, "ring_pages", 0) > 0
                    else 0
                ),
            )

            # Pass query to FMHA (for paged) or reconstruct qkv (for ragged)
            qkv = self._prepare_fmha_input(query, key, value)

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(qkv, kv_cache)

    def _prepare_fmha_input(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Prepare input for FMHA. To be overridden by subclasses if needed."""
        # Default: just return query (for paged layout)
        return query


class PyFlashinferPagedPrefillImpl(PyFlashinferPrefillImplBase):
    """FlashInfer prefill implementation with paged KV cache layout using MhaRotaryEmbeddingOp."""

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        """Create paged FMHA implementation."""
        return PyFlashinferPrefillPagedAttnOp(attn_configs, attn_inputs)

    def _create_rope_impl(self, attn_configs: AttentionConfigs) -> Any:
        """Create RoPE implementation for paged layout."""
        if attn_configs.rope_config.style == RopeStyle.No:
            return None
        return MhaRotaryEmbeddingOp(attn_configs)

    def _prepare_fmha_input(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """For paged layout, only return query (KV is already in cache)."""
        return query

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """Check if paged prefill implementation is supported.

        Returns True if:
        1. Not running on SM 10.0 (Blackwell) architecture
        2. The layer's cache is not a sliding-window ring
        3. A paged KV cache exists — attention reads K/V from it, so this impl
           cannot serve the warmup forward, which runs with kv_cache=None
        4. The underlying paged FMHA op supports the inputs
        5. MhaRotaryEmbeddingOp supports the inputs

        On (2): a ring holds only a window's worth of KV, but this impl plans over the
        full sequence length, so it would read ring slots that were never written. The
        ragged impl is the only correct prefill for a ring layer — it reads K/V from
        the QKV tensor and never touches the cache. Refusing here turns a future
        routing change (e.g. prefix reuse making prefix_lengths non-zero, which makes
        the ragged impl decline) into "cannot find mha type" rather than silently
        wrong attention.
        """
        if swa_ring_pages(attn_configs) > 0:
            return False
        return (
            not is_sm_100()
            and has_paged_kv_cache(attn_inputs)
            and PyFlashinferPrefillPagedAttnOp.support(attn_inputs)
        )

    def support_cuda_graph(self) -> bool:
        return True


class PyFlashinferPrefillImpl(PyFlashinferPrefillImplBase):
    """FlashInfer prefill implementation with ragged KV cache layout using MhaRotaryEmbeddingOp."""

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        """Create ragged FMHA implementation."""
        return PyFlashinferPrefillAttnOp(attn_configs)

    def _create_rope_impl(self, attn_configs: AttentionConfigs) -> Any:
        """Create RoPE implementation for ragged layout."""
        if attn_configs.rope_config.style == RopeStyle.No:
            return None
        return MhaRotaryEmbeddingOp(attn_configs)

    def _prepare_fmha_input(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """For ragged layout, reconstruct full qkv tensor from q, k, v."""
        # query: [total_tokens, num_heads, head_dim]
        # key: [total_tokens, num_kv_heads, head_dim]
        # value: [total_tokens, num_kv_heads, head_dim]

        # Flatten to 2D and concatenate
        q_flat = query.reshape(
            query.shape[0], -1
        )  # [total_tokens, num_heads * head_dim]
        k_flat = key.reshape(
            key.shape[0], -1
        )  # [total_tokens, num_kv_heads * head_dim]
        v_flat = value.reshape(
            value.shape[0], -1
        )  # [total_tokens, num_kv_heads * head_dim]

        # Concatenate along feature dimension
        qkv = torch.cat(
            [q_flat, k_flat, v_flat], dim=-1
        )  # [total_tokens, (num_heads + 2*num_kv_heads) * head_dim]

        return qkv

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """Check if ragged prefill implementation is supported.

        Returns True if:
        1. Not running on SM 10.0 (Blackwell) architecture
        2. The underlying ragged FMHA op supports the inputs
           (requires prefix_lengths to be empty or zero)
        3. MhaRotaryEmbeddingOp supports the inputs

        Sliding-window and sink layers stay here rather than moving to the paged impl.
        Ragged reads K/V from the QKV tensor the layer just produced, so a windowed
        layer's prefill never touches the KV cache at all — which is what lets the
        cache hold only a window's worth per request (see KVCacheWriteOp's ring
        write). ``window_left`` comes from plan(), and the sink is folded in from the
        LSE, both of which the ragged wrapper supports.
        """
        return not is_sm_100() and PyFlashinferPrefillAttnOp.support(attn_inputs)


def determine_use_tensor_core_from_configs(attn_configs: AttentionConfigs) -> bool:
    """Determine whether to use tensor cores based on attention configs."""
    # Use tensor cores for larger head dimensions and when kv_head_num matches requirements
    return attn_configs.head_num // attn_configs.kv_head_num >= 4


class PyFlashinferDecodeAttnOp(object):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
    ) -> None:
        self.g_workspace_buffer = get_py_flashinfer_workspace_buffer()
        self.attn_configs = attn_configs
        self.sink_bias = None
        # attn_configs already has head_num and kv_head_num divided by tp_size
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.head_dim_vo = attn_configs.v_size_per_head or attn_configs.size_per_head
        self.seq_size_per_block = attn_configs.kernel_tokens_per_block
        self.use_tensor_core = determine_use_tensor_core_from_configs(attn_configs)
        # FlashInfer 0.6.6 BatchDecodeWithPagedKVCacheWrapper.plan() takes a single
        # `head_dim` and no `head_dim_vo`. For asymmetric head_dim
        # (head_dim_qk != head_dim_vo), use BatchPrefillWithPagedKVCacheWrapper which
        # supports both. Decode is functionally equivalent to prefill with 1 token per
        # batch.
        self.use_prefill_for_decode = self.head_dim_qk != self.head_dim_vo
        if self.use_prefill_for_decode:
            # fa2, not "auto": fa3's paged dispatch bails out with
            # cudaErrorNotSupported unless head_dim_qk == head_dim_vo. See
            # resolve_paged_backend().
            self.decode_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.g_workspace_buffer,
                "HND",
                backend="fa2",
            )
        else:
            self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self.g_workspace_buffer,
                "HND",
                use_tensor_cores=self.use_tensor_core,
            )
        self.kv_cache_dtype = attn_configs.kv_cache_dtype
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.ring_pages = swa_ring_pages(attn_configs)
        self.window_tokens = max(attn_configs.sliding_window, 0)
        self.block_table: Optional[torch.Tensor] = None

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_sink_bias(self, bias):
        """Set per-layer sink bias for attention sink support."""
        self.sink_bias = normalize_sink_bias(bias)

    def _fill_params(self, attn_inputs: PyAttentionInputs) -> None:
        # Steady-state decode drops the host metadata loop and H2D copy.
        #
        # For a ring layer the page list still spans the whole sequence
        # (ceil(seq_len / page)), but every entry resolves to one of the ring pages,
        # and the kernel only ever reads the last `window_left + 1` keys — its
        # kv_start_idx is the window lower bound — so the pages it touches are exactly
        # the ring slots holding the window. See ring_block_table().
        self.block_table = ring_block_table(
            attn_inputs.kv_cache_kernel_block_id_device, self.ring_pages
        )
        self.fmha_params.fill_params_mha_device(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            self.block_table,
            self.seq_size_per_block,
        )

    def sync_layer_block_table(self, attn_inputs: PyAttentionInputs) -> None:
        """Re-point the planned page table at the current layer's cache group.

        See PyFlashinferPrefillPagedAttnOp.sync_layer_block_table for why an
        in-place refill is sufficient and no re-plan is needed.
        """
        self._fill_params(attn_inputs)

    def prepare(self, attn_inputs: PyAttentionInputs):
        # from rtp_llm.models_py.utils.debug import set_trace_on_tty
        # set_trace_on_tty()
        # Convert kv_cache_dtype to torch dtype
        if self.kv_cache_dtype == KvCacheDataType.INT8:
            kv_datatype = torch.int8
        elif self.kv_cache_dtype == KvCacheDataType.FP8:
            kv_datatype = torch.float8_e4m3fn
        else:  # BASE
            kv_datatype = get_scalar_type(attn_inputs.dtype)

        self._fill_params(attn_inputs)
        # Get torch.dtype from attention configs
        if self.use_prefill_for_decode:
            # Prefill wrapper plan: decode = prefill with 1 token per batch
            batch_size = attn_inputs.input_lengths.size(0)
            qo_indptr = torch.arange(
                batch_size + 1,
                dtype=torch.int32,
                device=attn_inputs.input_lengths.device,
            )
            self.decode_wrapper.plan(
                qo_indptr,
                self.fmha_params.decode_page_indptr_d,
                self.fmha_params.page_indice_d,
                self.fmha_params.paged_kv_last_page_len_d,
                self.local_head_num,
                self.local_kv_head_num,
                self.head_dim_qk,
                self.seq_size_per_block,
                causal=False,
                q_data_type=get_scalar_type(attn_inputs.dtype),
                kv_data_type=kv_datatype,
                head_dim_vo=self.head_dim_vo,
                window_left=window_left_from_sliding_window(
                    self.attn_configs.sliding_window
                ),
            )
        else:
            # BatchDecodeWithPagedKVCacheWrapper.plan() has no head_dim_vo; this
            # branch only runs when head_dim_qk == head_dim_vo anyway.
            self.decode_wrapper.plan(
                self.fmha_params.decode_page_indptr_d,
                self.fmha_params.page_indice_d,
                self.fmha_params.paged_kv_last_page_len_d,
                self.local_head_num,
                self.local_kv_head_num,
                self.head_dim_qk,
                self.seq_size_per_block,
                q_data_type=get_scalar_type(attn_inputs.dtype),
                kv_data_type=kv_datatype,
                window_left=window_left_from_sliding_window(
                    self.attn_configs.sliding_window
                ),
            )
        return self.fmha_params

    def prepare_for_cuda_graph_replay(self, attn_inputs: PyAttentionInputs) -> None:
        """Update CUDA graph replay buffers without calling plan().

        Replay refreshes decode metadata in-place via fill_decode_cuda_graph_params,
        falling back to the device MHA planner when the decode-only input is absent.
        """
        fill_decode = getattr(self.fmha_params, "fill_decode_cuda_graph_params", None)
        if (
            callable(fill_decode)
            and attn_inputs.sequence_lengths_plus_1_d is not None
            and attn_inputs.sequence_lengths_plus_1_d.numel() > 0
        ):
            fill_decode(
                attn_inputs.sequence_lengths_plus_1_d,
                attn_inputs.kv_cache_kernel_block_id_device,
                self.seq_size_per_block,
            )
            return

        # CUDA graph replay fallback when sequence_lengths_plus_1_d is absent.
        self.fmha_params.fill_params_mha_device(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_device,
            self.seq_size_per_block,
            forbid_realloc=True,
        )

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def _get_paged_kv_cache(self, kv_cache: "LayerKVCache"):
        """Get paged KV cache in the correct format for FlashInfer run().

        For asymmetric K/V (head_dim_qk != head_dim_vo), returns a tuple
        (k_cache, v_cache) so FlashInfer reads K and V with their respective
        head dimensions. For symmetric K/V, returns the merged 5D tensor.
        """
        if self.head_dim_vo != self.head_dim_qk:
            k_cache_sep = getattr(kv_cache, "k_cache", None)
            if k_cache_sep is not None and k_cache_sep.numel() > 0:
                return (k_cache_sep, kv_cache.v_cache)
        paged_kv_cache = kv_cache.kv_cache_base
        if paged_kv_cache is not None and paged_kv_cache.dim() == 2:
            paged_kv_cache = common.reshape_paged_kv_cache(
                paged_kv_cache,
                self.local_kv_head_num,
                self.seq_size_per_block,
                self.head_dim_qk,
            )
        return paged_kv_cache

    def forward(
        self, q: torch.Tensor, kv_cache: Optional[LayerKVCache], params: ParamsBase
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required"
        q = q.reshape(q.shape[0], self.local_head_num, self.head_dim_qk)
        paged_kv_cache = self._get_paged_kv_cache(kv_cache)

        # Not passing sinks= to run(): fa2/fa3 accept and discard it, so the sink is
        # folded in here from the LSE. See apply_attention_sink().
        if self.sink_bias is None:
            return self.decode_wrapper.run(q, paged_kv_cache)
        output, lse = self.decode_wrapper.run(q, paged_kv_cache, return_lse=True)
        return apply_attention_sink(output, lse, self.sink_bias)


class PyFlashinferDecodeImpl(FMHAImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = PyFlashinferDecodeAttnOp(attn_configs)
        self.rope_impl = FusedRopeKVCacheDecodeOp(attn_configs)
        self.attn_configs = attn_configs

        # Detect asymmetric head_dim (e.g. MiMo V2.5: QK=192, V=128)
        head_dim_qk = attn_configs.size_per_head
        head_dim_vo = attn_configs.v_size_per_head or head_dim_qk
        self.asymmetric_kv = head_dim_qk != head_dim_vo

        if self.asymmetric_kv:
            # For asymmetric models, the C++ fused rope+kv_cache kernel cannot
            # write K/V with different dimensions. Use Python RoPE + explicit
            # KVCacheWriteOp instead (same approach as the prefill path).
            self.py_rope_impl = MhaRotaryEmbeddingOp(attn_configs)
            self.kv_cache_write_op = KVCacheWriteOp(
                num_kv_heads=attn_configs.kv_head_num,
                head_size=attn_configs.size_per_head,
                token_per_block=attn_configs.kernel_tokens_per_block,
            )

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

        if self.asymmetric_kv:
            # Share fmha_params with py_rope_impl and kv_cache_write_op
            # (fmha_params has positions_d, batch_indice_d, page metadata)
            self.py_rope_impl.set_params(self.fmha_params)
            self.kv_cache_write_op.set_params(self.fmha_params)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return not attn_configs.use_mla

    def set_sink_bias(self, bias):
        """Set per-layer sink bias for attention sink support."""
        self.fmha_impl.set_sink_bias(bias)

    def sync_layer_block_table(self, attn_inputs: PyAttentionInputs) -> None:
        """Point this impl at the block table currently selected on attn_inputs."""
        self.fmha_impl.sync_layer_block_table(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            if self.asymmetric_kv:
                # Asymmetric head_dim path: C++ fused kernel cannot write K/V
                # with different dimensions, so we split QKV, apply RoPE via
                # Python, write K/V to cache explicitly, then run attention on Q.
                query, key, value = self.py_rope_impl.forward(qkv)
                self.kv_cache_write_op.forward(
                    key,
                    value,
                    kv_cache,
                    block_table=getattr(self.fmha_impl, "block_table", None),
                    ring_window_tokens=(
                        getattr(self.fmha_impl, "window_tokens", 0)
                        if getattr(self.fmha_impl, "ring_pages", 0) > 0
                        else 0
                    ),
                )
                qkv = query  # only Q goes to attention
            else:
                # Symmetric path: fused kernel does RoPE + KV write in one shot
                qkv = self.rope_impl.forward(qkv, kv_cache, self.rope_params)

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(qkv, kv_cache, self.fmha_params)
