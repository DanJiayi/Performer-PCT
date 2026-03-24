import argparse
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import ModelNet10
from model import PCTClassifier


class Logger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._fp = open(log_path, "a", encoding="utf-8")

    def log(self, msg: str):
        print(msg)
        self._fp.write(msg + "\n")
        self._fp.flush()

    def close(self):
        self._fp.close()


@dataclass
class Meter:
    correct: int = 0
    total: int = 0
    loss_sum: float = 0.0

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor):
        pred = logits.argmax(dim=1)
        self.correct += (pred == labels).sum().item()
        self.total += labels.numel()
        self.loss_sum += loss.item() * labels.size(0)

    @property
    def acc(self) -> float:
        return 100.0 * self.correct / max(1, self.total)

    @property
    def loss(self) -> float:
        return self.loss_sum / max(1, self.total)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    vv = v.lower()
    if vv in ("yes", "true", "t", "1"):
        return True
    if vv in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_epoch(model, loader, criterion, optimizer, device, redraw_interval=0, global_step=0):
    model.train()
    meter = Meter()
    step_times = []
    epoch_start = time.perf_counter()
    sample_count = 0
    for pts, labels in loader:
        if redraw_interval > 0 and hasattr(model, "redraw_projection_matrices"):
            if global_step > 0 and global_step % redraw_interval == 0:
                model.redraw_projection_matrices()
        sync_if_cuda(device)
        step_start = time.perf_counter()
        pts = pts.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(pts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        meter.update(logits.detach(), labels, loss.detach())
        sample_count += labels.size(0)
        sync_if_cuda(device)
        step_times.append(time.perf_counter() - step_start)
        global_step += 1
    epoch_time = time.perf_counter() - epoch_start
    speed = {
        "epoch_time_s": epoch_time,
        "avg_step_time_s": float(np.mean(step_times)) if step_times else 0.0,
        "samples_per_s": sample_count / max(epoch_time, 1e-9),
    }
    return meter, speed, global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    meter = Meter()
    for pts, labels in loader:
        pts = pts.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(pts)
        loss = criterion(logits, labels)
        meter.update(logits, labels, loss)
    return meter


@torch.no_grad()
def benchmark_inference(model, loader, device, warmup_steps=10, measure_steps=50):
    model.eval()
    batches = []
    for i, batch in enumerate(loader):
        batches.append(batch)
        if i >= (warmup_steps + measure_steps - 1):
            break
    if not batches:
        return {"avg_latency_ms": 0.0, "throughput_samples_s": 0.0}

    for i in range(min(warmup_steps, len(batches))):
        pts, _ = batches[i]
        pts = pts.to(device, non_blocking=True)
        _ = model(pts)
    sync_if_cuda(device)

    timings = []
    sample_total = 0
    start_idx = min(warmup_steps, len(batches))
    for i in range(start_idx, len(batches)):
        pts, labels = batches[i]
        pts = pts.to(device, non_blocking=True)
        sync_if_cuda(device)
        t0 = time.perf_counter()
        _ = model(pts)
        sync_if_cuda(device)
        dt = time.perf_counter() - t0
        timings.append(dt)
        sample_total += labels.size(0)

    if not timings:
        return {"avg_latency_ms": 0.0, "throughput_samples_s": 0.0}

    total_time = sum(timings)
    return {
        "avg_latency_ms": 1000.0 * float(np.mean(timings)),
        "throughput_samples_s": sample_total / max(total_time, 1e-9),
    }


def main():
    parser = argparse.ArgumentParser("Train PCT on ModelNet10")
    parser.add_argument("--data_root", type=str, default="data/ModelNet10")
    # Paper (PCT classification) uses 250 epochs, bs=32, lr=0.01, 1024 points.
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--npoints", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="checkpoints/PCT_best.pt")
    parser.add_argument("--performer", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--performer_nb_features", type=int, default=64)
    parser.add_argument("--performer_redraw_interval", type=int, default=1000)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.log_path:
        log_path = args.log_path
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"logs/train_{'performer' if args.performer else 'pct'}_{stamp}.log"
    logger = Logger(log_path)

    train_set = ModelNet10(root=args.data_root, split="train", npoints=args.npoints, augment=True)
    test_set = ModelNet10(root=args.data_root, split="test", npoints=args.npoints, augment=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = PCTClassifier(
        num_classes=len(train_set.classes),
        npoints=args.npoints,
        performer=args.performer,
        performer_nb_features=args.performer_nb_features,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_acc = 0.0
    final_test_acc = 0.0
    last_train_speed = {"epoch_time_s": 0.0, "avg_step_time_s": 0.0, "samples_per_s": 0.0}
    global_step = 0
    logger.log(f"Args: {vars(args)}")
    logger.log(f"Device: {device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        redraw_interval = args.performer_redraw_interval if args.performer else 0
        train_m, train_speed, global_step = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            redraw_interval=redraw_interval,
            global_step=global_step,
        )
        test_m = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        final_test_acc = test_m.acc
        last_train_speed = train_speed

        if test_m.acc > best_acc:
            best_acc = test_m.acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": train_set.classes,
                    "acc": best_acc,
                    "args": vars(args),
                },
                args.save_path,
            )
        epoch_msg = (
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss {train_m.loss:.4f} train_acc {train_m.acc:.2f}% | "
            f"test_loss {test_m.loss:.4f} test_acc {test_m.acc:.2f}% | best {best_acc:.2f}%"
        )
        speed_msg = (
            f"TrainSpeed epoch_time {train_speed['epoch_time_s']:.3f}s | "
            f"step_time {train_speed['avg_step_time_s']*1000:.3f}ms | "
            f"throughput {train_speed['samples_per_s']:.2f} samples/s"
        )
        logger.log(epoch_msg)
        logger.log(speed_msg)
        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            logger.log(f"GPU peak memory so far: {peak_mem:.2f} MB")

    infer_speed = benchmark_inference(model, test_loader, device)
    logger.log(f"Final test accuracy: {final_test_acc:.2f}%")
    logger.log(f"Best test accuracy: {best_acc:.2f}%")
    logger.log(
        f"Training summary | last_epoch_time {last_train_speed['epoch_time_s']:.3f}s | "
        f"last_step_time {last_train_speed['avg_step_time_s']*1000:.3f}ms | "
        f"last_throughput {last_train_speed['samples_per_s']:.2f} samples/s"
    )
    logger.log(
        f"Inference summary | latency {infer_speed['avg_latency_ms']:.3f}ms/batch | "
        f"throughput {infer_speed['throughput_samples_s']:.2f} samples/s"
    )
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        logger.log(f"GPU peak memory (total): {peak_mem:.2f} MB")
    logger.log(f"Best checkpoint: {args.save_path}")
    logger.log(f"Log file: {log_path}")
    logger.close()


if __name__ == "__main__":
    main()
