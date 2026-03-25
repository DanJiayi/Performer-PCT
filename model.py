import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math


def _create_orthogonal_random_matrix(d: int, m: int, device, dtype) -> torch.Tensor:
    blocks = []
    full_blocks = m // d
    remainder = m % d

    for _ in range(full_blocks):
        q, _ = torch.linalg.qr(torch.randn(d, d, device=device, dtype=dtype), mode="reduced")
        blocks.append(q)
    if remainder > 0:
        q, _ = torch.linalg.qr(torch.randn(d, d, device=device, dtype=dtype), mode="reduced")
        blocks.append(q[:, :remainder])

    projection = torch.cat(blocks, dim=1) if blocks else torch.empty(d, 0, device=device, dtype=dtype)
    scales = torch.randn(m, d, device=device, dtype=dtype).norm(dim=1)
    projection = projection * scales.unsqueeze(0)
    return projection


def _softmax_positive_feature_map_hyp(
    x: torch.Tensor, projection: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    # x: [B, N, D], projection: [D, M] -> [B, N, 2M]
    projected = torch.matmul(x, projection)  # [B,N,M]
    x_norm = (x ** 2).sum(dim=-1, keepdim=True) * 0.5
    pos = projected - x_norm
    neg = -projected - x_norm

    # Use a single stabilization baseline for both halves to preserve relative scaling.
    m = torch.cat([pos, neg], dim=-1).max(dim=-1, keepdim=True)[0]
    pos = pos - m
    neg = neg - m
    features = torch.cat([torch.exp(pos), torch.exp(neg)], dim=-1)
    return features * ((2.0 * projection.shape[1]) ** -0.5) + eps


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


class GeoRFF(nn.Module):
    """
    Learnable Gaussian/RBF RFF features from token coordinates.

    It approximates a positive kernel of the form:
      K(i,j) ~ exp(-tau * ||h(pos_i) - h(pos_j)||^2)
    using random Fourier features:
      phi(pos) = [cos(z @ omega), sin(z @ omega)] / sqrt(r)
    """

    def __init__(self, d_geo: int = 16, r_geo: int = 16):
        super().__init__()
        self.d_geo = d_geo
        self.r_geo = r_geo
        self.geo_mlp = nn.Sequential(
            nn.Linear(3, d_geo),
            nn.ReLU(inplace=True),
            nn.Linear(d_geo, d_geo),
        )
        # Learnable temperature tau. Kept scalar to match the screenshot-style design.
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        omega = _create_orthogonal_random_matrix(d_geo, r_geo, device="cpu", dtype=torch.float32)
        self.register_buffer("omega_geo", omega)
        self.register_buffer("_inv_sqrt_r", torch.tensor(1.0 / math.sqrt(r_geo), dtype=torch.float32))

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        # pos: [B, N, 3]
        g = self.geo_mlp(pos)  # [B, N, d_geo]
        tau = F.softplus(self.log_tau) + 1e-6
        z = torch.sqrt(2.0 * tau) * g  # [B, N, d_geo]
        proj = torch.matmul(z, self.omega_geo)  # [B, N, r_geo]
        # Positive feature map (non-negative), matching the "distance modulation" intent:
        # phi_geo(z) = exp(proj - ||z||^2 / 2) / sqrt(r_geo)
        z_norm = (z ** 2).sum(dim=-1, keepdim=True) * 0.5  # [B, N, 1]
        phi = torch.exp(proj - z_norm) * self._inv_sqrt_r  # [B, N, r_geo]
        return phi


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
        self.attn_time_s = 0.0

    @staticmethod
    def _sync_if_cuda(x: torch.Tensor):
        if x.is_cuda:
            torch.cuda.synchronize(x.device)

    def reset_timing(self):
        self.attn_time_s = 0.0

    def get_timing(self) -> float:
        return float(self.attn_time_s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N]
        self._sync_if_cuda(x)
        t0 = time.perf_counter()
        q = self.q(x).transpose(1, 2)      # [B,N,Da]
        k = self.k(x)                       # [B,Da,N]
        v = self.v(x).transpose(1, 2)      # [B,N,C]
        energy = torch.bmm(q, k)           # [B,N,N]

        attn = F.softmax(energy, dim=1)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        fsa = torch.bmm(attn, v).transpose(1, 2).contiguous()  # [B,C,N]

        out = self.proj(x - fsa) + x
        self._sync_if_cuda(x)
        self.attn_time_s += time.perf_counter() - t0
        return out


class PerformerOffsetAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        nb_features: int = 64,
        add_dist: bool = False,
        geo_d_geo: int = 16,
        geo_r_geo: int = 16,
    ):
        super().__init__()
        da = channels // 4
        self.q = nn.Conv1d(channels, da, 1, bias=False)
        self.k = nn.Conv1d(channels, da, 1, bias=False)
        self.v = nn.Conv1d(channels, channels, 1, bias=False)
        self.proj = LBR1d(channels, channels)
        self.nb_features = nb_features
        self.eps = 1e-6
        projection = _create_orthogonal_random_matrix(da, nb_features, device="cpu", dtype=torch.float32)
        self.register_buffer("projection_matrix", projection)
        self.attn_time_s = 0.0
        self.geo_rff = GeoRFF(d_geo=geo_d_geo, r_geo=geo_r_geo) if add_dist else None

    @staticmethod
    def _sync_if_cuda(x: torch.Tensor):
        if x.is_cuda:
            torch.cuda.synchronize(x.device)

    def reset_timing(self):
        self.attn_time_s = 0.0

    def get_timing(self) -> float:
        return float(self.attn_time_s)

    @torch.no_grad()
    def redraw_projection_matrix(self):
        d = self.projection_matrix.shape[0]
        m = self.nb_features
        self.projection_matrix.copy_(
            _create_orthogonal_random_matrix(
                d=d,
                m=m,
                device=self.projection_matrix.device,
                dtype=self.projection_matrix.dtype,
            )
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, C, N]
        self._sync_if_cuda(x)
        t0 = time.perf_counter()
        q = self.q(x).transpose(1, 2).contiguous()  # [B,N,Da]
        k = self.k(x).transpose(1, 2).contiguous()  # [B,N,Da]
        v = self.v(x).transpose(1, 2).contiguous()  # [B,N,C]
        scale = q.shape[-1] ** -0.25
        q = q * scale
        k = k * scale

        q_prime = _softmax_positive_feature_map_hyp(q, self.projection_matrix, eps=self.eps)  # [B,N,2M]
        k_prime = _softmax_positive_feature_map_hyp(k, self.projection_matrix, eps=self.eps)  # [B,N,2M]

        if self.geo_rff is not None and pos is not None:
            # Outer-product feature composition:
            #   Phi_q(i) = phi_sm(q_i) ⊗ phi_geo(pos_i)
            # Then linear attention works on the composed feature dimension.
            phi_geo = self.geo_rff(pos)  # [B, N, r_geo]
            B, N, r_sm = q_prime.shape
            r_geo_out = phi_geo.shape[-1]
            Phi_q = (q_prime.unsqueeze(-1) * phi_geo.unsqueeze(-2)).reshape(B, N, r_sm * r_geo_out)
            Phi_k = (k_prime.unsqueeze(-1) * phi_geo.unsqueeze(-2)).reshape(B, N, r_sm * r_geo_out)

            kv = torch.bmm(Phi_k.transpose(1, 2), v)  # [B, R_total, C]
            k_sum = Phi_k.sum(dim=1)  # [B, R_total]
            denom = torch.bmm(Phi_q, k_sum.unsqueeze(-1)) + self.eps  # [B, N, 1]
            fsa = torch.bmm(Phi_q, kv) / denom  # [B, N, C]
        else:
            kv = torch.bmm(k_prime.transpose(1, 2), v)  # [B,2M,N] @ [B,N,C] -> [B,2M,C]
            k_sum = k_prime.sum(dim=1)  # [B,2M]
            denom = torch.bmm(q_prime, k_sum.unsqueeze(-1))  # [B,N,2M] @ [B,2M,1] -> [B,N,1]
            denom = denom + self.eps
            fsa = torch.bmm(q_prime, kv) / denom  # [B,N,2M] @ [B,2M,C] -> [B,N,C]

        fsa = fsa.transpose(1, 2).contiguous()  # [B,C,N]

        out = self.proj(x - fsa) + x
        self._sync_if_cuda(x)
        self.attn_time_s += time.perf_counter() - t0
        return out


class PCTClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        npoints: int = 1024,
        dropout: float = 0.5,
        performer: bool = False,
        performer_nb_features: int = 64,
        add_dist: bool = False,
    ):
        super().__init__()
        self.embed = NeighborEmbedding(npoints=npoints)
        if performer:
            self.att1 = PerformerOffsetAttention(256, nb_features=performer_nb_features, add_dist=add_dist)
            self.att2 = PerformerOffsetAttention(256, nb_features=performer_nb_features, add_dist=add_dist)
            self.att3 = PerformerOffsetAttention(256, nb_features=performer_nb_features, add_dist=add_dist)
            self.att4 = PerformerOffsetAttention(256, nb_features=performer_nb_features, add_dist=add_dist)
        else:
            # Baseline path is kept identical regardless of add_dist.
            self.att1 = OffsetAttention(256)
            self.att2 = OffsetAttention(256)
            self.att3 = OffsetAttention(256)
            self.att4 = OffsetAttention(256)
        self._performer_enabled = performer
        self._add_dist = add_dist
        self.linear_fuse = nn.Sequential(
            nn.Conv1d(256 * 4, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
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
        xyz2, feat = self.embed(x)   # xyz2: [B, N/4, 3], feat: [B, N/4, 256]
        f = feat.transpose(1, 2).contiguous()  # [B,256,N']

        if self._performer_enabled:
            if self._add_dist:
                f1 = self.att1(f, xyz2)
                f2 = self.att2(f1, xyz2)
                f3 = self.att3(f2, xyz2)
                f4 = self.att4(f3, xyz2)
            else:
                f1 = self.att1(f)
                f2 = self.att2(f1)
                f3 = self.att3(f2)
                f4 = self.att4(f3)
        else:
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

    def redraw_projection_matrices(self):
        for module in self.modules():
            if hasattr(module, "redraw_projection_matrix"):
                module.redraw_projection_matrix()

    def reset_attention_timing(self):
        for module in [self.att1, self.att2, self.att3, self.att4]:
            if hasattr(module, "reset_timing"):
                module.reset_timing()

    def get_attention_timing(self) -> float:
        total = 0.0
        for module in [self.att1, self.att2, self.att3, self.att4]:
            if hasattr(module, "get_timing"):
                total += module.get_timing()
        return total
