# MLA / DSA KV support in the Ascend MP transfer path

Module: [`lmcache_ascend/v1/multiprocess/npu_gather.py`](../../../lmcache_ascend/v1/multiprocess/npu_gather.py)

## Why this was needed

The Ascend multiprocess (MP) gather/scatter override (`npu_gather.py`) only
supported `SEPARATE_KV`. MLA_KV (DeepSeek-V2/V3) and DSA_KV (DeepSeek-V3.2
sparse) fell back to the upstream per-layer PyTorch path — never reaching the
fused `multi_layer_kv_transfer` kernel the in-process connector already uses.

The fused kernel, the Ascend `KVCacheFormat` detection, and the in-process
connector (`_V2KVTransferMixin`) **already** handle MLA/DSA. The only gap was
the MP module, blocked by one subtle contract mismatch.

## The blocker: the SHM chunk-shape contract

The MP SHM server allocates per-chunk buffers from a single
upstream-negotiated `hidden_dim_size` + `use_mla` flag. For an Ascend
`(K, V)` tuple, `compute_kv_layout` reports **`num_kv_heads * kv_lora_rank`**
(the K-plane product) and a format `is_mla()` does **not** recognise, so the
server allocated `[2, num_layers, tokens, num_kv_heads*kv_lora_rank]` — wrong
in both the leading and trailing dims.

The fused MLA/DSA kernel instead needs the planes **concatenated** into one
flat block `[num_layers, tokens, kv_lora_rank + qk_rope_head_dim (+dsa)]`
(mirrors the in-process connector's `get_shape`, `npu_connectors.py:2397`).

## The fix

1. **`_derive_plane_geometry`** (pure helper, CPU-testable): derives per-format
   plane widths and staging layout. SEPARATE_KV → two equal planes
   `[2, L, tokens, nh*hs]`; MLA/DSA → one flat block `[1, L, tokens, sum]`.
2. **`_compute_kv_layout_wrapper`** (patches `compute_kv_layout` on `base` and
   `worker_transfer`): for a supported NPU MLA/DSA layout, reports the summed
   hidden dim (from the descriptor) and the `NL_X_NB_BS_HS` format flag (the
   value `is_mla()` accepts), so the server allocates the rank-3
   `[num_layers, tokens, k+v(+dsa)]` buffer. SEPARATE_KV / CPU / CUDA pass
   through unchanged.
3. **`view_as` reconciliation**: gather does `out[out_idx].view_as(staging)`,
   which is identity for SEPARATE_KV's rank-4 slot and adds a leading-1 dim for
   MLA/DSA's rank-3 slot (zero-copy, same element count). Scatter canonicalises
   a retrieved MLA/DSA slot back to `[1, L, tokens, hidden]` before slicing.

The MLA format flag is a signalling mechanism for the server's shape branch
only — the kernel call is driven by the descriptor's real Ascend
`KVCacheFormat` (MLA_KV/DSA_KV), so the stored `_engine_kv_format` is never
used to interpret Ascend tensors.

## Support matrix

| Format | Supported | Staging shape | Plane extras (k, v, dsa, scale) |
|---|---|---|---|
| SEPARATE_KV | yes (unchanged) | `[2, L, tokens, nh*hs]` | `(0, 0, 0, 0)` |
| MLA_KV | **yes (new)** | `[1, L, tokens, k+v]` | `(kv_lora_rank, qk_rope_head_dim, 0, 0)` |
| DSA_KV | **yes (new)** | `[1, L, tokens, k+v+dsa]` | `(k, v, dsa_head_dim, 0)` |
| DSA_C8_KV | no | — | needs `multi_layer_kv_transfer_multi_plane` (follow-up) |
| MULTI_PLANE_KV | no | — | needs multi-plane kernel (follow-up) |

## Verification

- `tests/v1/test_npu_mp_gather_scatter.py`: plane-geometry derivation, the
  layout-negotiation patch (MLA/DSA override + SEPARATE_KV/CPU passthrough),
  and the `view_as` rank reconciliation run CPU-only.
- On Ascend NPU: MLA and DSA descriptor fields, and bit-exact
  gather→scatter round-trips for every stored block; the patched
  `compute_kv_layout` reports `is_mla(fmt) is True` with `hidden = k+v`.

## Risk notes

- `num_kv_heads` must be 1 for MLA (the latent KV head) — the staging per-token
  width is `kv_lora_rank`, with no `num_kv_heads` factor. Real DeepSeek MLA
  satisfies this.
- `view_as` requires the SHM slot to be contiguous (true for
  `torch.frombuffer(...).view(...)` slots) and to match the staging element
  count — if the layout patch did not take effect, `view_as` fails loudly
  rather than corrupting silently.
