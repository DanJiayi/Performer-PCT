"""
Micro-benchmark: sweep token length N comparing BOTH attentions:

  - offset_ms: PCT baseline `OffsetAttention` (full N×N, no Performer)
  - performer_ms: `PerformerOffsetAttention` (linear-cost kernel approx.)

Use this to see whether Performer only wins past some N* (depends on GPU, M, batch).

Note: model attention forwards still call cuda.synchronize for attn timing, so absolute
ms are inflated; comparing performer/offset at each N remains meaningful for relative speed.

Run (from repo root or this directory):

  python bench_attention_n_sweep.py
  python bench_attention_n_sweep.py --n_list 256 512 1024 2048 4096 8192
"""
import argparse
import time

import torch

from model import OffsetAttention, PerformerOffsetAttention


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench_forward_ms(mod, x: torch.Tensor, repeats: int, warmup: int, device: torch.device) -> float:
    mod.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = mod(x)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = mod(x)
        sync(device)
        t1 = time.perf_counter()
    return (t1 - t0) / repeats * 1000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--channels", type=int, default=256)
    p.add_argument("--performer_nb_features", type=int, default=32)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument(
        "--n_list",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024, 2048, 4096],
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    offset_attn = OffsetAttention(args.channels).to(device)
    performer_attn = PerformerOffsetAttention(
        args.channels, nb_features=args.performer_nb_features
    ).to(device)

    print(
        f"device={device} B={args.batch_size} C={args.channels} "
        f"performer_M={args.performer_nb_features} warmup={args.warmup} repeats={args.repeats}"
    )
    print(
        f"{'N':>8} {'offset_ms':>14} {'performer_ms':>14} {'performer/offset':>18}"
    )

    for n in args.n_list:
        x = torch.randn(args.batch_size, args.channels, n, device=device)
        t_offset = bench_forward_ms(
            offset_attn, x, args.repeats, args.warmup, device
        )
        t_performer = bench_forward_ms(
            performer_attn, x, args.repeats, args.warmup, device
        )
        ratio = t_performer / max(t_offset, 1e-12)
        print(
            f"{n:>8} {t_offset:>14.4f} {t_performer:>14.4f} {ratio:>18.3f}"
        )


if __name__ == "__main__":
    main()
