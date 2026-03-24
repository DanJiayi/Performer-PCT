import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    # x: [B, N, C]
    dist = torch.cdist(x, x)
    idx = dist.topk(k=k, largest=False, dim=-1)[1]
    return idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # points: [B, N, C], idx: [B, S] or [B, S, K]
    b = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(b, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    # xyz: [B, N, 3], return [B, npoint]
    device = xyz.device
    b, n, _ = xyz.shape
    centroids = torch.zeros(b, npoint, dtype=torch.long, device=device)
    distance = torch.ones(b, n, device=device) * 1e10
    farthest = torch.randint(0, n, (b,), dtype=torch.long, device=device)
    batch_indices = torch.arange(b, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(b, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


class LBR1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LBR2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SGModule(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, npoint: int, k: int = 32):
        super().__init__()
        self.npoint = npoint
        self.k = k
        self.lbr1 = LBR2d(in_ch * 2, out_ch)
        self.lbr2 = LBR2d(out_ch, out_ch)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor):
        # xyz: [B, N, 3], feat: [B, N, C]
        fps_idx = farthest_point_sample(xyz, self.npoint)
        xyz_s = index_points(xyz, fps_idx)          # [B, S, 3]
        feat_s = index_points(feat, fps_idx)        # [B, S, C]

        dist = torch.cdist(xyz_s, xyz)              # [B, S, N]
        knn_idx = dist.topk(k=self.k, largest=False, dim=-1)[1]  # [B, S, K]
        neigh_feat = index_points(feat, knn_idx)    # [B, S, K, C]

        feat_center = feat_s.unsqueeze(2).expand_as(neigh_feat)
        edge_feat = torch.cat([neigh_feat - feat_center, feat_center], dim=-1)  # [B,S,K,2C]
        edge_feat = edge_feat.permute(0, 3, 1, 2).contiguous()  # [B,2C,S,K]
        edge_feat = self.lbr1(edge_feat)
        edge_feat = self.lbr2(edge_feat)
        feat_out = torch.max(edge_feat, dim=-1)[0].transpose(1, 2).contiguous()  # [B,S,out_ch]
        return xyz_s, feat_out


class NeighborEmbedding(nn.Module):
    def __init__(self, npoints: int):
        super().__init__()
        self.base1 = LBR1d(3, 64)
        self.base2 = LBR1d(64, 64)
        self.sg1 = SGModule(in_ch=64, out_ch=128, npoint=npoints // 2, k=32)
        self.sg2 = SGModule(in_ch=128, out_ch=256, npoint=npoints // 4, k=32)

    def forward(self, x: torch.Tensor):
        # x: [B, N, 3]
        feat = x.transpose(1, 2).contiguous()
        feat = self.base1(feat)
        feat = self.base2(feat)
        feat = feat.transpose(1, 2).contiguous()  # [B,N,64]
        xyz1, feat1 = self.sg1(x, feat)
        xyz2, feat2 = self.sg2(xyz1, feat1)
        return xyz2, feat2


class OffsetAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        da = channels // 4
        self.q = nn.Conv1d(channels, da, 1, bias=False)
        self.k = nn.Conv1d(channels, da, 1, bias=False)
        self.v = nn.Conv1d(channels, channels, 1, bias=False)
        self.proj = LBR1d(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N]
        q = self.q(x).transpose(1, 2)      # [B,N,Da]
        k = self.k(x)                       # [B,Da,N]
        v = self.v(x).transpose(1, 2)      # [B,N,C]
        energy = torch.bmm(q, k)           # [B,N,N]

        attn = F.softmax(energy, dim=1)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        fsa = torch.bmm(attn, v).transpose(1, 2).contiguous()  # [B,C,N]

        out = self.proj(x - fsa) + x
        return out


class PCTClassifier(nn.Module):
    def __init__(self, num_classes: int, npoints: int = 1024, dropout: float = 0.5):
        super().__init__()
        self.embed = NeighborEmbedding(npoints=npoints)
        self.att1 = OffsetAttention(256)
        self.att2 = OffsetAttention(256)
        self.att3 = OffsetAttention(256)
        self.att4 = OffsetAttention(256)
        self.linear_fuse = nn.Sequential(
            nn.Conv1d(256 * 4, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, 3]
        _, feat = self.embed(x)         # [B, N/4, 256]
        f = feat.transpose(1, 2).contiguous()  # [B,256,N']

        f1 = self.att1(f)
        f2 = self.att2(f1)
        f3 = self.att3(f2)
        f4 = self.att4(f3)

        out = torch.cat([f1, f2, f3, f4], dim=1)
        out = self.linear_fuse(out)
        out_max = F.adaptive_max_pool1d(out, 1).squeeze(-1)
        out_avg = F.adaptive_avg_pool1d(out, 1).squeeze(-1)
        global_feat = torch.cat([out_max, out_avg], dim=1)
        return self.cls_head(global_feat)
