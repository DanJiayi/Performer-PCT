import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from PCT.dataset import ModelNet10
from PCT.model import PCTClassifier


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


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    meter = Meter()
    for pts, labels in loader:
        pts = pts.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(pts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        meter.update(logits.detach(), labels, loss.detach())
    return meter


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


def main():
    parser = argparse.ArgumentParser("Train PCT on ModelNet10")
    parser.add_argument("--data_root", type=str, default="data/ModelNet10")
    # Paper (PCT classification on ModelNet40) uses 250 epochs, bs=32, lr=0.01, 1024 points.
    # We keep these as defaults for best alignment, while training on ModelNet10 here.
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--npoints", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="checkpoints/PCT_best.pt")
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = PCTClassifier(num_classes=len(train_set.classes), npoints=args.npoints).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, optimizer, device)
        test_m = evaluate(model, test_loader, criterion, device)
        scheduler.step()

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
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss {train_m.loss:.4f} train_acc {train_m.acc:.2f}% | "
            f"test_loss {test_m.loss:.4f} test_acc {test_m.acc:.2f}% | best {best_acc:.2f}%"
        )

    print(f"Final best test accuracy: {best_acc:.2f}%")
    print(f"Best checkpoint: {args.save_path}")


if __name__ == "__main__":
    main()
