"""KV Cache Write Operation for paged KV cache."""

import logging
from typing import Any, Optional, Tuple

import flashinfer.page as page
import torch

from rtp_llm.ops.compute_ops import LayerKVCache

logger = logging.getLogger(__name__)


class KVCacheWriteOp:
    """Operator for writing key-value pairs to paged KV cache."""

    def __init__(
        self,
        num_kv_heads: int,
        head_size: int,
        token_per_block: int,
    ) -> None:
        """
        Initialize KV Cache Write operator.

        Args:
            num_kv_heads: Number of key-value heads
            head_size: Dimension of each attention head
            token_per_block: Number of tokens per KV cache block (page size)
        """
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.token_per_block = token_per_block
        self.params = None

    def set_params(self, params: Any):
        """Set the params object to be used by this op."""
        self.params = params

    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        block_table: Optional[torch.Tensor] = None,
        ring_window_tokens: int = 0,
    ) -> None:
        """
        Write key and value tensors to paged KV cache.

        Args:
            key: Key tensor [total_tokens, num_kv_heads, head_dim]
            value: Value tensor [total_tokens, num_kv_heads, head_dim]
            kv_cache: KV cache [num_pages, 2, num_kv_heads, page_size, head_dim] (HND layout)
            block_table: Optional [batch_size, max_blocks_per_batch] block table mapping
                         logical page indices to physical page IDs. When provided, used
                         directly for asymmetric write (bypasses stale decode_page_indptr_d).
            ring_window_tokens: >0 for a sliding-window layer whose cache is a ring of
                         this many tokens. Tokens further back than the window are
                         dropped: they would be overwritten by a later token in the same
                         call anyway (position p and p + window share a ring slot), and
                         PyTorch's scatter leaves which one wins undefined.
        """
        if kv_cache is not None:
            # For real execution - use provided KV cache
            # KV cache has shape [num_pages, 2, num_kv_heads, page_size, head_dim] (HND layout)
            k_cache_separate = getattr(kv_cache, "k_cache", None)
            if k_cache_separate is not None and k_cache_separate.numel() > 0:
                # Asymmetric K/V path (e.g. MiMo V2.5: K head_dim=192, V head_dim=128)
                k_cache = k_cache_separate
                v_cache = kv_cache.v_cache
            else:
                # Symmetric path: interleaved [num_pages, 2, num_kv_heads, page_size, head_dim]
                k_cache = kv_cache.kv_cache_base[
                    :, 0, :, :, :
                ]  # [num_pages, num_kv_heads, page_size, head_dim]
                v_cache = kv_cache.kv_cache_base[
                    :, 1, :, :, :
                ]  # [num_pages, num_kv_heads, page_size, head_dim]

            # FlashInfer requires batch_indices/positions size == nnz.
            # Device planner leaves buffers oversized, so narrow without a host sync.
            nnz = key.size(0)
            batch_indices = self.params.batch_indice_d.narrow(0, 0, nnz)
            positions = self.params.positions_d.narrow(0, 0, nnz)

            if k_cache.size(-1) != v_cache.size(-1):
                # Asymmetric K/V write: flashinfer kernel requires equal K/V head_dim
                page_size = k_cache.size(2)
                pos = positions.long()
                batch_idx = batch_indices.long()
                page_ids_local = pos // page_size
                slot_in_page = pos % page_size

                bt = (
                    block_table
                    if block_table is not None
                    and block_table.dim() == 2
                    and block_table.numel() > 0
                    else None
                )
                if bt is not None:
                    # The per-layer block table set by select_block_map_for_layer (and
                    # ring-cycled by the FMHA op for windowed layers). Preferred over
                    # decode_page_indptr_d, which is computed at prepare time and can
                    # be stale.
                    kv_indptr = None
                    kv_indices = None
                    keep = page_ids_local < bt.size(1)
                    num_reqs = bt.size(0)
                else:
                    kv_indptr = self.params.decode_page_indptr_d
                    kv_indices = self.params.page_indice_d
                    pages_per_req = kv_indptr[batch_idx + 1] - kv_indptr[batch_idx]
                    keep = page_ids_local < pages_per_req.long()
                    num_reqs = max(int(kv_indptr.numel()) - 1, 1)

                if ring_window_tokens > 0:
                    # Newest position per request, then keep only the window's tail.
                    # num_reqs comes from a shape, so this needs no host sync.
                    seq_end = torch.full(
                        (num_reqs,), -1, dtype=pos.dtype, device=pos.device
                    )
                    seq_end.scatter_reduce_(
                        0, batch_idx, pos, reduce="amax", include_self=False
                    )
                    keep = keep & (pos > seq_end[batch_idx] - ring_window_tokens)

                if bool(keep.all()):
                    sel_page, sel_batch, sel_slot = (
                        page_ids_local,
                        batch_idx,
                        slot_in_page,
                    )
                    k_src, v_src = key, value
                else:
                    sel = keep.nonzero(as_tuple=True)[0]
                    sel_page, sel_batch, sel_slot = (
                        page_ids_local[sel],
                        batch_idx[sel],
                        slot_in_page[sel],
                    )
                    k_src, v_src = key[sel], value[sel]

                if bt is not None:
                    physical_page = bt[sel_batch, sel_page].long()
                else:
                    assert kv_indptr is not None and kv_indices is not None
                    physical_page = kv_indices[
                        kv_indptr[sel_batch].long() + sel_page
                    ].long()

                k_cache[physical_page, :, sel_slot, :] = k_src
                v_cache[physical_page, :, sel_slot, :] = v_src
            else:
                # Append K and V to paged cache using HND layout
                page.append_paged_kv_cache(  # type: ignore
                    key,  # append_key: [total_tokens, num_kv_heads, head_dim]
                    value,  # append_value: [total_tokens, num_kv_heads, head_dim]
                    batch_indices,
                    positions,
                    (k_cache, v_cache),  # paged_kv_cache: tuple of K and V caches
                    self.params.page_indice_d,
                    self.params.decode_page_indptr_d,
                    self.params.paged_kv_last_page_len_d,
                    "HND",  # kv_layout: HND layout (num_pages, num_kv_heads, page_size, head_dim)
                )
        else:
            # For warmup/JIT compilation - create dummy KV cache
            (
                batch_indices,
                positions,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_len,
                max_num_pages,
            ) = self._prepare_warmup_cache_indices(value.size(0), value.device)

            # Create MHA KV cache: [num_pages, num_kv_heads, page_size, head_dim] (HND layout)
            k_cache = torch.empty(
                (
                    max_num_pages,
                    self.num_kv_heads,
                    self.token_per_block,
                    self.head_size,
                ),
                dtype=value.dtype,
                device=value.device,
            )
            v_cache = torch.empty(
                (
                    max_num_pages,
                    self.num_kv_heads,
                    self.token_per_block,
                    value.size(-1),  # use actual V head_dim (may differ from K)
                ),
                dtype=value.dtype,
                device=value.device,
            )

            if k_cache.size(-1) != v_cache.size(-1):
                # Asymmetric K/V warmup write
                page_size = k_cache.size(2)
                pos = positions.long()
                page_ids_local = pos // page_size
                slot_in_page = pos % page_size
                physical_page = kv_page_indices[
                    kv_page_indptr[batch_indices.long()].long() + page_ids_local
                ].long()
                k_cache[physical_page, :, slot_in_page, :] = key
                v_cache[physical_page, :, slot_in_page, :] = value
            else:
                # Append K and V to paged cache using HND layout
                page.append_paged_kv_cache(  # type: ignore
                    key,
                    value,
                    batch_indices,
                    positions,
                    (k_cache, v_cache),  # paged_kv_cache: tuple of K and V caches
                    kv_page_indices,
                    kv_page_indptr,
                    kv_last_page_len,
                    "HND",  # kv_layout: HND layout (num_pages, num_kv_heads, page_size, head_dim)
                )

    def _prepare_warmup_cache_indices(
        self, num_tokens: int, device: torch.device
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int
    ]:
        """
        Prepare dummy cache indices for warmup/JIT compilation.

        Args:
            num_tokens: Number of tokens to process
            device: Device to create tensors on

        Returns:
            Tuple of (batch_indices, positions, kv_page_indices, kv_page_indptr, kv_last_page_len, max_num_pages)
        """
        # Assume 1 batch, sequential tokens
        batch_indices = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        positions = torch.arange(num_tokens, dtype=torch.int32, device=device)

        # Calculate required pages
        max_num_pages = (num_tokens + self.token_per_block - 1) // self.token_per_block

        # Page indices: [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2, ...]
        kv_page_indices = (
            torch.arange(num_tokens, dtype=torch.int32, device=device)
            // self.token_per_block
        )

        # Page indptr: [0, max_num_pages] for single batch
        kv_page_indptr = torch.tensor(
            [0, max_num_pages], dtype=torch.int32, device=device
        )

        # Last page length
        last_page_len = num_tokens % self.token_per_block
        if last_page_len == 0:
            last_page_len = self.token_per_block
        kv_last_page_len = torch.tensor(
            [last_page_len], dtype=torch.int32, device=device
        )

        return (
            batch_indices,
            positions,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len,
            max_num_pages,
        )
