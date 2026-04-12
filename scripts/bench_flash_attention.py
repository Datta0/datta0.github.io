#!/usr/bin/env python3
"""
Benchmark a simple Triton FlashAttention-style forward kernel against naive PyTorch attention.

Assumptions:
- forward pass only
- non-causal attention
- CUDA + Triton available
- BF16 inputs

By default, this benchmarks sequence lengths:
1024, 2048, 4096, 8192, 16384, 32768

It writes:
- a CSV with timing and peak-memory numbers
- a PNG with time-vs-sequence-length and memory-vs-sequence-length plots
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import torch
import triton
import triton.language as tl


@dataclass
class ModelSpec:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


def qwen3_8b_spec() -> ModelSpec:
    # From the public Qwen/Qwen3-8B config on Hugging Face.
    return ModelSpec(hidden_size=4096, num_heads=32, num_kv_heads=8, head_dim=128)


def load_model_spec(args: argparse.Namespace) -> ModelSpec:
    if args.preset == "qwen3-8b":
        return qwen3_8b_spec()

    if args.config_json is not None:
        with args.config_json.open() as f:
            cfg = json.load(f)
        num_heads = int(cfg["num_attention_heads"])
        num_kv_heads = int(cfg.get("num_key_value_heads", num_heads))
        head_dim = int(cfg.get("head_dim", cfg["hidden_size"] // num_heads))
        return ModelSpec(
            hidden_size=int(cfg["hidden_size"]),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

    return ModelSpec(
        hidden_size=args.num_heads * args.head_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
    )


@triton.jit
def flash_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stride_qm,
    stride_qk,
    stride_km,
    stride_kk,
    stride_vm,
    stride_vk,
    stride_om,
    stride_ok,
    n_ctx,
    d,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    row_mask = offs_m < n_ctx

    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = row_mask[:, None] & (offs_d[None, :] < d)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in tl.range(0, n_ctx, BLOCK_N):
        col_offsets = start_n + offs_n
        col_mask = col_offsets < n_ctx
        kv_mask = col_mask[:, None] & (offs_d[None, :] < d)

        k_ptrs = k_ptr + col_offsets[:, None] * stride_km + offs_d[None, :] * stride_kk
        v_ptrs = v_ptr + col_offsets[:, None] * stride_vm + offs_d[None, :] * stride_vk

        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(row_mask[:, None] & col_mask[None, :], qk, float("-inf"))

        block_max = tl.max(qk, axis=1)
        new_m_i = tl.maximum(m_i, block_max)

        p = tl.exp(qk - new_m_i[:, None])
        alpha = tl.exp(m_i - new_m_i)

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = alpha[:, None] * acc + tl.dot(p, v)
        m_i = new_m_i

    out = acc / l_i[:, None]
    out = tl.where(row_mask[:, None] & (offs_d[None, :] < d), out, 0.0)

    o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    o_mask = row_mask[:, None] & (offs_d[None, :] < d)
    tl.store(o_ptrs, out, mask=o_mask)


def flash_attn_triton_single_head(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    assert q.shape == k.shape == v.shape
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16

    n_ctx, d = q.shape
    block_d = triton.next_power_of_2(d)
    o = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(d)

    grid = (triton.cdiv(n_ctx, block_m),)
    flash_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        n_ctx,
        d,
        sm_scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return o


def flash_attn_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    if q.ndim == 2:
        return flash_attn_triton_single_head(q, k, v, block_m=block_m, block_n=block_n)

    assert q.ndim == k.ndim == v.ndim == 3
    outs = []
    for h in range(q.shape[0]):
        outs.append(flash_attn_triton_single_head(q[h], k[h], v[h], block_m=block_m, block_n=block_n))
    return torch.stack(outs, dim=0)


def naive_attention_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


@dataclass
class BenchResult:
    status: str
    avg_ms: float | None
    peak_bytes: int | None
    max_abs_diff: float | None = None


def benchmark_runner(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup: int,
    iters: int,
    check_output: torch.Tensor | None = None,
) -> BenchResult:
    times_ms: list[float] = []
    peaks: list[int] = []

    try:
        for _ in range(warmup):
            out = fn(q, k, v)
            torch.cuda.synchronize()
            del out

        for _ in range(iters):
            torch.cuda.synchronize()
            baseline_alloc = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(q, k, v)
            end.record()
            end.synchronize()

            elapsed_ms = start.elapsed_time(end)
            peak_alloc = torch.cuda.max_memory_allocated() - baseline_alloc

            times_ms.append(elapsed_ms)
            peaks.append(max(0, int(peak_alloc)))
            del out

        max_abs_diff = None
        if check_output is not None:
            out = fn(q, k, v)
            torch.cuda.synchronize()
            max_abs_diff = (out.float().cpu() - check_output.float().cpu()).abs().max().item()
            del out

        return BenchResult(
            status="ok",
            avg_ms=sum(times_ms) / len(times_ms),
            peak_bytes=max(peaks) if peaks else None,
            max_abs_diff=max_abs_diff,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            clear_cuda()
            return BenchResult(status="oom", avg_ms=None, peak_bytes=None, max_abs_diff=None)
        raise
    finally:
        clear_cuda()


def repeat_kv_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if x.shape[0] == num_heads:
        return x
    assert num_heads % x.shape[0] == 0
    groups = num_heads // x.shape[0]
    return x.repeat_interleave(groups, dim=0)


def make_inputs(
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + seq_len)
    q = torch.randn((num_heads, seq_len, head_dim), device="cuda", dtype=torch.bfloat16, generator=generator)
    k_base = torch.randn((num_kv_heads, seq_len, head_dim), device="cuda", dtype=torch.bfloat16, generator=generator)
    v_base = torch.randn((num_kv_heads, seq_len, head_dim), device="cuda", dtype=torch.bfloat16, generator=generator)
    k = repeat_kv_heads(k_base, num_heads)
    v = repeat_kv_heads(v_base, num_heads)
    return q.contiguous(), k.contiguous(), v.contiguous()


def seq_lengths_pow2(min_k: int, max_k: int) -> list[int]:
    seqs = []
    cur = min_k * 1024
    end = max_k * 1024
    while cur <= end:
        seqs.append(cur)
        cur *= 2
    return seqs


def bytes_to_gib(num_bytes: int | None) -> float | None:
    if num_bytes is None:
        return None
    return num_bytes / (1024 ** 3)


def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seqs = [int(row["seq_len"]) for row in rows]
    xlabels = [f"{seq // 1024}K" for seq in seqs]

    def numeric_or_nan(value: object) -> float:
        return float("nan") if value is None else float(value)

    triton_time = [numeric_or_nan(row["triton_time_ms"]) for row in rows]
    pytorch_time = [numeric_or_nan(row["pytorch_time_ms"]) for row in rows]
    triton_mem = [numeric_or_nan(row["triton_peak_gib"]) for row in rows]
    pytorch_mem = [numeric_or_nan(row["pytorch_peak_gib"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(seqs, triton_time, marker="o", linewidth=2, label="Triton FlashAttention sketch")
    axes[0].plot(seqs, pytorch_time, marker="o", linewidth=2, label="Naive PyTorch attention")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(seqs)
    axes[0].set_xticklabels(xlabels)
    axes[0].set_xlabel("Sequence length")
    axes[0].set_ylabel("Average forward time (ms)")
    axes[0].set_title("Forward Time vs Sequence Length")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(seqs, triton_mem, marker="o", linewidth=2, label="Triton FlashAttention sketch")
    axes[1].plot(seqs, pytorch_mem, marker="o", linewidth=2, label="Naive PyTorch attention")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(seqs)
    axes[1].set_xticklabels(xlabels)
    axes[1].set_xlabel("Sequence length")
    axes[1].set_ylabel("Peak extra memory (GiB)")
    axes[1].set_title("Peak Memory vs Sequence Length")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("BF16 Attention Benchmark")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def plot_time_results(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seqs = [int(row["seq_len"]) for row in rows]
    xlabels = [f"{seq // 1024}K" for seq in seqs]

    def numeric_or_nan(value: object) -> float:
        return float("nan") if value is None else float(value)

    triton_time = [numeric_or_nan(row["triton_time_ms"]) for row in rows]
    pytorch_time = [numeric_or_nan(row["pytorch_time_ms"]) for row in rows]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(seqs, triton_time, marker="o", linewidth=2, label="Triton FlashAttention sketch")
    ax.plot(seqs, pytorch_time, marker="o", linewidth=2, label="Naive PyTorch attention")
    ax.set_xscale("log", base=2)
    ax.set_xticks(seqs)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Average forward time (ms)")
    ax.set_title("Forward Time vs Sequence Length")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def plot_memory_results(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seqs = [int(row["seq_len"]) for row in rows]
    xlabels = [f"{seq // 1024}K" for seq in seqs]

    def numeric_or_nan(value: object) -> float:
        return float("nan") if value is None else float(value)

    triton_mem = [numeric_or_nan(row["triton_peak_gib"]) for row in rows]
    pytorch_mem = [numeric_or_nan(row["pytorch_peak_gib"]) for row in rows]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(seqs, triton_mem, marker="o", linewidth=2, label="Triton FlashAttention sketch")
    ax.plot(seqs, pytorch_mem, marker="o", linewidth=2, label="Naive PyTorch attention")
    ax.set_xscale("log", base=2)
    ax.set_xticks(seqs)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Peak extra memory (GiB)")
    ax.set_title("Peak Memory vs Sequence Length")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def plot_memory_log_results(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seqs = [int(row["seq_len"]) for row in rows]
    xlabels = [f"{seq // 1024}K" for seq in seqs]

    def numeric_or_nan(value: object) -> float:
        return float("nan") if value is None else float(value)

    triton_mem = [numeric_or_nan(row["triton_peak_gib"]) for row in rows]
    pytorch_mem = [numeric_or_nan(row["pytorch_peak_gib"]) for row in rows]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(seqs, triton_mem, marker="o", linewidth=2, label="Triton FlashAttention sketch")
    ax.plot(seqs, pytorch_mem, marker="o", linewidth=2, label="Naive PyTorch attention")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(seqs)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Peak extra memory (GiB, log scale)")
    ax.set_title("Peak Memory vs Sequence Length (Log-Log)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["qwen3-8b"], default=None, help="Known model preset.")
    parser.add_argument("--config-json", type=Path, default=None, help="Optional HF-style model config.json.")
    parser.add_argument("--head-dim", type=int, default=64, help="Per-head hidden dimension.")
    parser.add_argument("--num-heads", type=int, default=1, help="Number of query heads.")
    parser.add_argument("--num-kv-heads", type=int, default=1, help="Number of key/value heads.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per backend.")
    parser.add_argument("--iters", type=int, default=10, help="Timed iterations per backend.")
    parser.add_argument("--min-k", type=int, default=1, help="Smallest sequence length in K tokens.")
    parser.add_argument("--max-k", type=int, default=32, help="Largest sequence length in K tokens.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--block-m", type=int, default=64, help="Triton query tile size.")
    parser.add_argument("--block-n", type=int, default=64, help="Triton KV tile size.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "flash_attention_bench",
        help="Directory to store the CSV and PNG outputs.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    spec = load_model_spec(args)
    seq_lengths = seq_lengths_pow2(args.min_k, args.max_k)
    rows: list[dict[str, object]] = []

    print(
        f"Using hidden_size={spec.hidden_size}, num_heads={spec.num_heads}, "
        f"num_kv_heads={spec.num_kv_heads}, head_dim={spec.head_dim}, dtype=bf16"
    )
    print(f"{'seq_len':>8} | {'triton_ms':>10} | {'triton_gib':>10} | {'torch_ms':>10} | {'torch_gib':>10} | {'max_abs_diff':>12}")
    print("-" * 78)

    for seq_len in seq_lengths:
        clear_cuda()
        q, k, v = make_inputs(seq_len, spec.num_heads, spec.num_kv_heads, spec.head_dim, args.seed)

        triton_fn = lambda q_, k_, v_: flash_attn_triton(q_, k_, v_, block_m=args.block_m, block_n=args.block_n)
        reference_out = None
        pytorch_result = BenchResult(status="skipped", avg_ms=None, peak_bytes=None, max_abs_diff=None)
        try:
            reference_out = naive_attention_torch(q, k, v).float().cpu()
            torch.cuda.synchronize()
            pytorch_result = benchmark_runner(
                naive_attention_torch,
                q,
                k,
                v,
                warmup=args.warmup,
                iters=args.iters,
                check_output=None,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                clear_cuda()
                pytorch_result = BenchResult(status="oom", avg_ms=None, peak_bytes=None, max_abs_diff=None)
            else:
                raise

        try:
            triton_result = benchmark_runner(
                triton_fn,
                q,
                k,
                v,
                warmup=args.warmup,
                iters=args.iters,
                check_output=reference_out if reference_out is not None else None,
            )
        finally:
            if reference_out is not None:
                del reference_out
            del q, k, v
            clear_cuda()

        row = {
            "seq_len": seq_len,
            "dtype": "bfloat16",
            "hidden_size": spec.hidden_size,
            "num_heads": spec.num_heads,
            "num_kv_heads": spec.num_kv_heads,
            "head_dim": spec.head_dim,
            "triton_status": triton_result.status,
            "triton_time_ms": triton_result.avg_ms,
            "triton_peak_bytes": triton_result.peak_bytes,
            "triton_peak_gib": bytes_to_gib(triton_result.peak_bytes),
            "pytorch_status": pytorch_result.status,
            "pytorch_time_ms": pytorch_result.avg_ms,
            "pytorch_peak_bytes": pytorch_result.peak_bytes,
            "pytorch_peak_gib": bytes_to_gib(pytorch_result.peak_bytes),
            "max_abs_diff": triton_result.max_abs_diff,
            "projected_scores_gib": (spec.num_heads * seq_len * seq_len * 2) / (1024 ** 3),
            "projected_scores_plus_probs_gib": (2 * spec.num_heads * seq_len * seq_len * 2) / (1024 ** 3),
        }
        rows.append(row)

        def fmt(x: object) -> str:
            if isinstance(x, float):
                return f"{x:.3f}"
            if x is None:
                return "OOM"
            return str(x)

        print(
            f"{seq_len:8d} | "
            f"{fmt(row['triton_time_ms']):>10} | "
            f"{fmt(row['triton_peak_gib']):>10} | "
            f"{fmt(row['pytorch_time_ms']):>10} | "
            f"{fmt(row['pytorch_peak_gib']):>10} | "
            f"{fmt(row['max_abs_diff']):>12}"
        )

    csv_path = args.out_dir / "flash_attention_benchmark.csv"
    png_path = args.out_dir / "flash_attention_benchmark.png"
    time_png_path = args.out_dir / "flash_attention_time.png"
    memory_png_path = args.out_dir / "flash_attention_memory.png"
    memory_log_png_path = args.out_dir / "flash_attention_memory_log.png"

    save_csv(rows, csv_path)
    plot_results(rows, png_path)
    plot_time_results(rows, time_png_path)
    plot_memory_results(rows, memory_png_path)
    plot_memory_log_results(rows, memory_log_png_path)

    print()
    print(f"Saved CSV to {csv_path}")
    print(f"Saved plot to {png_path}")
    print(f"Saved time plot to {time_png_path}")
    print(f"Saved memory plot to {memory_png_path}")
    print(f"Saved log-memory plot to {memory_log_png_path}")


if __name__ == "__main__":
    main()
