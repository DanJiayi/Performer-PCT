import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def read_off_vertices(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        if header != "OFF":
            raise ValueError(f"Invalid OFF header in {path}: {header}")
        n_verts, _, _ = map(int, f.readline().strip().split())
        verts = []
        for _ in range(n_verts):
            x, y, z = map(float, f.readline().strip().split()[:3])
            verts.append([x, y, z])
    return np.asarray(verts, dtype=np.float32)


def pc_normalize(pc: np.ndarray) -> np.ndarray:
    centroid = np.mean(pc, axis=0, keepdims=True)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / (m + 1e-8)
    return pc


def random_scale_translate(points: np.ndarray) -> np.ndarray:
    scales = np.random.uniform(2.0 / 3.0, 1.5, size=(1, 3)).astype(np.float32)
    shifts = np.random.uniform(-0.2, 0.2, size=(1, 3)).astype(np.float32)
    return points * scales + shifts


def random_point_dropout(points: np.ndarray, max_dropout_ratio: float = 0.875) -> np.ndarray:
    dropout_ratio = np.random.random() * max_dropout_ratio
    drop_idx = np.where(np.random.random((points.shape[0],)) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        points[drop_idx, :] = points[0, :]
    return points


class ModelNet10(Dataset):
    def __init__(self, root: str, split: str, npoints: int = 1024, augment: bool = False):
        super().__init__()
        assert split in ["train", "test"]
        self.root = root
        self.split = split
        self.npoints = npoints
        self.augment = augment

        self.classes = sorted(
            [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: List[Tuple[str, int]] = []
        for c in self.classes:
            split_dir = os.path.join(root, c, split)
            if not os.path.isdir(split_dir):
                continue
            files = sorted([f for f in os.listdir(split_dir) if f.endswith(".off")])
            for f in files:
                self.samples.append((os.path.join(split_dir, f), self.class_to_idx[c]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        pts = read_off_vertices(path)

        if pts.shape[0] >= self.npoints:
            choice = np.random.choice(pts.shape[0], self.npoints, replace=False)
        else:
            choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice, :]
        pts = pc_normalize(pts)

        if self.augment:
            pts = random_scale_translate(pts)
            pts = random_point_dropout(pts)

        return torch.from_numpy(pts).float(), torch.tensor(label, dtype=torch.long)
