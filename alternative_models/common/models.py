"""Model architectures for the alternative-model experiments.

- BaselineMLP: exact replica of the current production model
  (src/models/classifier.py::TSLClassifier) — residual MLP, 126 -> 57.
- LandmarkTransformer: joints as tokens with learned joint/hand embeddings,
  Transformer encoder + CLS pooling.
- HandGCN: graph convolutional network on the anatomical hand graph
  (21 nodes, MediaPipe connectivity), shared weights across the two hands.
- HandGAT: graph attention network on the same topology.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .paths import REPO_ROOT
from .hand_graph import normalized_adjacency

sys.path.insert(0, str(REPO_ROOT))
from src.models.classifier import TSLClassifier  # noqa: E402


class BaselineMLP(TSLClassifier):
    """The current production model (used as baseline, retrained on the
    clean protocol)."""

    def __init__(self, num_classes: int = 57, input_size: int = 126,
                 dropout_rate: float = 0.2):
        super().__init__(num_classes=num_classes, input_size=input_size,
                         dropout_rate=dropout_rate)


class _TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block built from plain PyTorch ops
    (manual QKV attention + softmax). nn.TransformerEncoderLayer and
    nn.MultiheadAttention emit fused native ops (aten::_transformer_encoder_
    layer_fwd / aten::_native_multi_head_attention) that the ONNX exporter
    cannot convert; this block keeps the model export-friendly."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int,
                 dropout: float = 0.2):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(dim_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        q, k, v = self.qkv(n).chunk(3, dim=-1)
        B, T, _ = q.shape
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        h = self.attn_drop(attn) @ v
        h = h.transpose(1, 2).reshape(B, T, self.d_model)
        x = x + self.drop(self.out(h))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class LandmarkTransformer(nn.Module):
    """Transformer over hand-landmark tokens.

    Each of the 2x21 joints is one token with features [x, y, z]
    (optionally extended with per-joint bone vectors), plus learned
    joint-index and hand-index embeddings. A CLS token aggregates the
    sequence; the head maps it to the 57 classes.
    """

    def __init__(self, num_classes: int = 57, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dim_ff: int = 128,
                 dropout: float = 0.2, token_dim: int = 3):
        super().__init__()
        self.d_model = d_model
        self.n_joints = 21
        self.n_hands = 2
        self.token_dim = token_dim

        self.joint_emb = nn.Embedding(21, d_model)
        self.hand_emb = nn.Embedding(2, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.input_proj = nn.Linear(token_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 1 + 2 * 21, d_model) * 0.02)

        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, 21, token_dim) or (B, 42, token_dim)."""
        if x.dim() == 4:
            B = x.shape[0]
            x = x.reshape(B, self.n_hands * self.n_joints, self.token_dim)
        B = x.shape[0]

        hand_ids = torch.arange(2, device=x.device).repeat_interleave(21)
        joint_ids = torch.arange(21, device=x.device).repeat(2)
        tok = self.input_proj(x) + self.joint_emb(joint_ids) + self.hand_emb(hand_ids)

        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tok], dim=1) + self.pos_emb
        for block in self.blocks:
            seq = block(seq)
        pooled = seq[:, 0]
        return self.head(pooled)


class GCNBlock(nn.Module):
    """Graph convolution: X' = A_hat X W (+ residual + BN)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2,
                 residual: bool = True):
        super().__init__()
        self.gc = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual and in_dim == out_dim
        if self.residual:
            self.res_bn = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        """x: (B, N, F), a_hat: (N, N) -> (B, N, F')"""
        h = torch.matmul(a_hat, x)
        h = self.gc(h)
        h = self.bn(h.transpose(1, 2)).transpose(1, 2)
        if self.residual:
            h = self.res_bn(x.transpose(1, 2)).transpose(1, 2) + h
        return F.relu(self.dropout(h))


class HandGCN(nn.Module):
    """GCN on the anatomical hand graph. The two hands share weights;
    masked mean+max pooling per hand, then a shared MLP head."""

    def __init__(self, num_classes: int = 57, hidden: int = 64,
                 num_layers: int = 2, dropout: float = 0.2,
                 input_dim: int = 3, with_geo: bool = False):
        super().__init__()
        self.n_joints = 21
        self.input_dim = input_dim
        adj = normalized_adjacency(include_self=True)
        self.register_buffer("a_hat", torch.from_numpy(adj))
        self.num_layers = num_layers

        layers = [nn.Linear(input_dim, hidden, bias=False)]
        for _ in range(num_layers - 1):
            layers.append(GCNBlock(hidden, hidden, dropout=dropout))
        self.blocks = nn.ModuleList(layers)

        self.pool_dim = hidden * 2  # mean + max per hand
        self.head = nn.Sequential(
            nn.Linear(self.pool_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, 21, input_dim)"""
        B = x.shape[0]
        h = x
        for i, layer in enumerate(self.blocks):
            if i == 0:
                h = F.relu(layer(h.reshape(B * 2, self.n_joints, self.input_dim)))
            else:
                h = layer(h, self.a_hat)
        h = h.reshape(B, 2, self.n_joints, -1)  # (B, 2, 21, hidden)

        mask = (x.abs().sum(-1) > 1e-8).float().unsqueeze(-1)  # (B,2,21,1)
        count = mask.sum(dim=2).clamp(min=1)
        mean = (h * mask).sum(dim=2) / count
        maxp = (h * mask).max(dim=2).values  # 0 for absent hand
        pooled = torch.cat([mean, maxp], dim=-1)  # (B, 2, 2*hidden)
        pooled = pooled.reshape(B, -1)
        return self.head(pooled)


class GATBlock(nn.Module):
    """Graph attention layer (GATv1-style) over a fixed adjacency.

    e_ij = LeakyReLU(a_src . W h_i + a_dst . W h_j); alpha_ij = softmax_j(e_ij)
    over the (undirected) neighbors of node i; h'_i = sum_j alpha_ij W h_j.
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.att_dim = max(out_dim // n_heads, 8)
        self.W = nn.Linear(in_dim, n_heads * self.att_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(1, 1, self.n_heads, self.att_dim))
        self.a_dst = nn.Parameter(torch.zeros(1, 1, self.n_heads, self.att_dim))
        self.out = nn.Linear(n_heads * self.att_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: (B, N, F), adj: (N, N) -> (B, N, out_dim)"""
        B, N, _ = x.shape
        h = self.W(x).reshape(B, N, self.n_heads, self.att_dim)
        e = (h * self.a_src).sum(-1).unsqueeze(1) + \
            (h * self.a_dst).sum(-1).unsqueeze(2)          # (B, N, N, H)
        e = self.leaky(e)
        mask = (adj == 0).unsqueeze(0).unsqueeze(-1)       # (1, N, N, 1)
        e = e.masked_fill(mask, float("-inf"))
        att = torch.softmax(e, dim=2)
        att = self.dropout(att)
        att = att.permute(0, 3, 1, 2)          # (B, H, N, N)
        hv = h.permute(0, 2, 1, 3)             # (B, H, N, D)
        out = torch.matmul(att, hv)            # (B, H, N, D)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.n_heads * self.att_dim)
        return self.out(out)


class HandGAT(nn.Module):
    """GAT on the hand graph with the same pooling/head as HandGCN."""

    def __init__(self, num_classes: int = 57, hidden: int = 64,
                 num_layers: int = 2, n_heads: int = 4, dropout: float = 0.2,
                 input_dim: int = 3):
        super().__init__()
        self.n_joints = 21
        self.input_dim = input_dim
        adj = normalized_adjacency(include_self=True)
        self.register_buffer("a_hat", torch.from_numpy(adj))

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GATBlock(input_dim, hidden, n_heads, dropout))
            else:
                layers.append(GATBlock(hidden, hidden, n_heads, dropout))
        self.blocks = nn.ModuleList(layers)
        self.bn = nn.BatchNorm1d(hidden)

        self.head = nn.Sequential(
            nn.Linear(hidden * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = x.reshape(B * 2, self.n_joints, self.input_dim)
        for i, layer in enumerate(self.blocks):
            h = layer(h, self.a_hat)
        h = self.bn(h.transpose(1, 2)).transpose(1, 2)
        h = F.relu(h).reshape(B, 2, self.n_joints, -1)

        mask = (x.abs().sum(-1) > 1e-8).float().unsqueeze(-1)
        count = mask.sum(dim=2).clamp(min=1)
        mean = (h * mask).sum(dim=2) / count
        maxp = (h * mask).max(dim=2).values  # 0 for absent hand
        pooled = torch.cat([mean, maxp], dim=-1).reshape(B, -1)
        return self.head(pooled)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False