# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from typing import Optional

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.ascend.testing import do_bench_npu

NUM_HEADS = [(8, 2), (16, 2)]
HEAD_SIZES = [128]
# only support 32 currently
BLOCK_SIZES = [32]

DTYPES = [torch.float16]
QDTYPES = [None]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]


def _experiment_compile_options():
    dynamic = os.getenv("EXPERIMENT_DYNAMIC_CV", "0") == "1"
    options = {"enable_dynamic_cv_pipeline": dynamic}
    if dynamic:
        options["set_workspace_multibuffer"] = 0
        options["inter_cache_num"] = 1
        options["load_cache_num"] = 1
        intra = os.getenv("EXPERIMENT_INTRA_CACHE_NUM")
        if intra is not None:
            options["intra_cache_num"] = int(intra)
    else:
        depth = os.getenv("EXPERIMENT_DEPTH")
        if depth is not None:
            options["set_workspace_multibuffer"] = int(depth)
    multibuffer_num = os.getenv("EXPERIMENT_MULTIBUFFER_NUM")
    if multibuffer_num is not None:
        options["multibuffer_num"] = int(multibuffer_num)
    merge = os.getenv("EXPERIMENT_VF_MERGE_LEVEL")
    if merge is not None:
        options["vf_merge_level"] = int(merge)
    return options


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def kernel_unified_attention_2d(output_ptr,  # [num_tokens, num_query_heads, head_size]
                                query_ptr,  # [num_tokens, num_query_heads, head_size]
                                key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
                                value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
                                block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
                                seq_lens_ptr,  # [num_seqs]
                                alibi_slopes_ptr,  # [num_query_heads]
                                scale,  # float32
                                k_scale,  # float32
                                v_scale,  # float32
                                softcap,  # float32
                                num_queries_per_kv: tl.constexpr,  # int
                                block_table_stride: tl.int64,  # int
                                query_stride_0: tl.int64,  # int
                                query_stride_1: tl.int64,  # int, should be equal to head_size
                                output_stride_0: tl.int64,  # int
                                output_stride_1: tl.int64,  # int, should be equal to head_size
                                BLOCK_SIZE: tl.constexpr,  # int
                                HEAD_SIZE: tl.constexpr,  # int
                                HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
                                USE_ALIBI_SLOPES: tl.constexpr,  # bool
                                USE_SOFTCAP: tl.constexpr,  # bool
                                SLIDING_WINDOW: tl.constexpr,  # int
                                stride_k_cache_0: tl.int64,  # int
                                stride_k_cache_1: tl.int64,  # int
                                stride_k_cache_2: tl.int64,  # int
                                stride_k_cache_3: tl.constexpr,  # int
                                stride_v_cache_0: tl.int64,  # int
                                stride_v_cache_1: tl.int64,  # int
                                stride_v_cache_2: tl.int64,  # int
                                stride_v_cache_3: tl.constexpr,  # int
                                query_start_len_ptr,  # [num_seqs+1]
                                num_seqs: tl.int32, BLOCK_M: tl.constexpr,  # int
                                ):

    q_block_global_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    kv_head_idx = query_head_idx // num_queries_per_kv

    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        mid_val = tl.load(query_start_len_ptr + mid) // BLOCK_M + mid
        if mid_val <= q_block_global_idx:
            left = mid + 1
        else:
            right = mid

    seq_idx = left - 1
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_M + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index \
        - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_M >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    query_pos = q_block_local_idx * BLOCK_M + offs_m

    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)

    Q_block_ptr = tl.make_block_ptr(
        base=query_ptr + cur_batch_in_all_start_index * query_stride_0 + query_head_idx * query_stride_1,
        shape=(cur_batch_query_len, HEAD_SIZE),
        strides=(query_stride_0, 1),
        offsets=(q_block_local_idx * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_SIZE_PADDED),
        order=(1, 0),
    )
    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_head_idx)

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

    # iterate through tiles
    for j in range(0, num_blocks):

        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

        offs_n = tl.arange(0, BLOCK_SIZE)

        K_block_ptr = tl.make_block_ptr(
            base=key_cache_ptr + physical_block_idx * stride_k_cache_0 + kv_head_idx * stride_k_cache_2,
            shape=(BLOCK_SIZE, HEAD_SIZE),
            strides=(stride_k_cache_1, stride_k_cache_3),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
            order=(1, 0),
        )

        K_load = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")

        V_block_ptr = tl.make_block_ptr(
            base=value_cache_ptr + physical_block_idx * stride_v_cache_0 + kv_head_idx * stride_v_cache_2,
            shape=(BLOCK_SIZE, HEAD_SIZE),
            strides=(stride_v_cache_1, stride_v_cache_3),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
            order=(1, 0),
        )

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        V_load = tl.load(V_block_ptr, boundary_check=(1,), padding_option="zero")

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        seq_offset = j * BLOCK_SIZE + offs_n

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        S = tl.zeros(shape=(BLOCK_M, BLOCK_SIZE), dtype=tl.float32)

        # Load K in its physical [token, head_size] layout and transpose it
        # explicitly for Q @ K^T. The Ascend lowering recognizes this layout;
        # constructing a transposed tensor through pointer offsets can produce
        # a non-terminating kernel with the custom compiler.
        S += scale * tl.dot(Q, tl.trans(K))

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(query_mask_0[:, None] & seq_mask, S, float("-inf"))

        if SLIDING_WINDOW > 0:
            S = tl.where((context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW, S, float("-inf"))

        if USE_ALIBI_SLOPES:
            S += alibi_slope * (seq_offset - context_len)

        # compute running maximum
        m_j = tl.maximum(M, tl.max(S, axis=1))
        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        P = tl.exp(S - m_j[:, None])

        l_j = tl.sum(P, axis=1)

        alpha = tl.exp(M - m_j)

        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    acc = acc / L[:, None]

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr + cur_batch_in_all_start_index * output_stride_0 + query_head_idx * output_stride_1,
        shape=(cur_batch_query_len, HEAD_SIZE),
        strides=(output_stride_0, 1),
        offsets=(q_block_local_idx * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_SIZE_PADDED),
        order=(1, 0),
    )
    tl.store(output_block_ptr, acc.to(output_ptr.dtype.element_ty), boundary_check=(0, 1))


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    block_size = v.shape[1]
    assert q.element_size() >= 2 or block_size >= 32, \
        "Block size must be at least 32 for fp8"

    use_alibi_slopes = alibi_slopes is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = 16

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_M)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_M)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_M) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_M)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_M) + num_seqs
    #    = floor(q.shape[0] / BLOCK_M) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_M + num_seqs

    kernel_unified_attention_2d[(
        total_num_q_blocks,
        num_query_heads,
    )](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        scale=softmax_scale,
        k_scale=k_descale,
        v_scale=v_descale,
        softcap=softcap,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_SOFTCAP=(softcap > 0),
        SLIDING_WINDOW=(1 + window_size[0]),
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        **_experiment_compile_options(),
    )


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx:start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = torch.triu(empty_mask,
                                             diagonal=kv_len - (query_len + sliding_window) + 1).bool().logical_not()
            mask |= sliding_window_mask
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.perf(repeat=17)
@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("q_dtype", QDTYPES)
@torch.inference_mode()
def test_triton_unified_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
    q_dtype: Optional[torch.dtype],
) -> None:
    torch.set_default_device("npu")

    if q_dtype is not None and q_dtype.itemsize < 2 and block_size < 32:
        pytest.skip("block size must be at least 32 for fp8")

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = ((sliding_window - 1, 0) if sliding_window is not None else (-1, -1))
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)

    output = torch.empty_like(query)

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None
    k_descale = None
    v_descale = None
    if q_dtype is not None:
        # QKV are drawn from N(0, 1): no need for a fp8 scaling factor
        maybe_quantized_query = query.to(q_dtype)
        maybe_quantized_key_cache = key_cache.to(q_dtype)
        maybe_quantized_value_cache = value_cache.to(q_dtype)

        scale_shape = (num_seqs, num_kv_heads)
        q_descale = None  # Not yet supported
        k_descale = torch.rand(scale_shape, dtype=torch.float32)
        v_descale = torch.rand(scale_shape, dtype=torch.float32)

    unified_attention(
        q=maybe_quantized_query,
        k=maybe_quantized_key_cache,
        v=maybe_quantized_value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )
    atol, rtol = 1.5e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol), \
        f"{torch.max(torch.abs(output - ref_output))}"


@torch.inference_mode()
def benchmark_unified_attention(warmup=5, active=30):
    torch.set_default_device("npu")
    torch.manual_seed(0)
    seq_lens = [(1, 1328), (5, 18), (129, 463)]
    query_lens = [x[0] for x in seq_lens]
    kv_lens_list = [x[1] for x in seq_lens]
    num_query_heads, num_kv_heads = 8, 2
    head_size, block_size, num_blocks = 128, 32, 2048
    dtype = torch.float16
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens_list)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0, num_blocks, (len(seq_lens), max_num_blocks_per_seq), dtype=torch.int32)
    output = torch.empty_like(query)

    fn = lambda: unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    fn()
    torch.npu.synchronize()
    latency_ms = do_bench_npu(
        fn,
        warmup=warmup,
        active=active,
        target_kernel_name="kernel_unified_attention_2d",
    )
    print(f"BENCHMARK operator=unified_attention latency_ms={latency_ms:.6f} warmup={warmup} active={active}")
    return latency_ms


if __name__ == "__main__":
    print("[EXPERIMENT] operator_parameters=" + json.dumps({
        "sequence_lengths": [[1, 1328], [5, 18], [129, 463]],
        "num_query_heads": 8,
        "num_kv_heads": 2,
        "head_size": 128,
        "block_size": 32,
        "num_blocks": 2048,
        "dtype": "float16",
        "causal": True,
        "sliding_window": None,
        "soft_cap": None,
        "q_dtype": None,
    }, sort_keys=True))
    # A small, deterministic case used by the operator-screening smoke test.
    # Keep the full parametrized pytest cases above for later coverage runs.
    test_triton_unified_attn(
        seq_lens=[(1, 1328), (5, 18), (129, 463)],
        num_heads=(8, 2),
        head_size=128,
        sliding_window=None,
        dtype=torch.float16,
        block_size=32,
        soft_cap=None,
        num_blocks=2048,
        q_dtype=None,
    )
    print("======Unified Attention Test Passed!======")
    benchmark_unified_attention(
        warmup=int(os.getenv("EXPERIMENT_WARMUP", "5")),
        active=int(os.getenv("EXPERIMENT_ACTIVE", "30")),
    )
