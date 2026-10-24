"""Compact FT-Transformer for tabular regression (CPU, CUDA or Apple MPS).

Each numeric feature is a token; missing values use a learned per-feature
missing token, so hidden driver history is represented as it is for a new
driver at deployment. Float32 only.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn


def pick_device(requested: str | None = None) -> torch.device:
    req = (requested or os.environ.get('EVTRUST_TORCH_DEVICE', 'auto')).lower()
    if req in ('cuda', 'auto') and torch.cuda.is_available():
        return torch.device('cuda')
    if req in ('mps', 'auto') and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class _FT(nn.Module):
    def __init__(self, n_num, n_cat, d=64, heads=4, layers=3, dropout=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_num, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_num, d))
        self.missing = nn.Parameter(torch.randn(n_num, d) * 0.1)
        self.cat = nn.Embedding(n_cat + 1, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, xn, xm, xc):
        tok = xn.unsqueeze(-1) * self.w + self.b
        tok = torch.where(xm.unsqueeze(-1), self.missing.expand_as(tok), tok)
        t = torch.cat([self.cls.expand(len(xn), -1, -1), tok, self.cat(xc).unsqueeze(1)], dim=1)
        return self.head(self.encoder(t)[:, 0]).squeeze(-1)


class _MLP(nn.Module):
    """Residual MLP baseline: value plus missing indicator per feature, station embedding."""

    def __init__(self, n_num, n_cat, width=256, depth=3, dropout=0.1):
        super().__init__()
        self.cat = nn.Embedding(n_cat + 1, 16)
        self.inp = nn.Linear(2 * n_num + 16, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.ReLU(),
                                                   nn.Dropout(dropout), nn.Linear(width, width)) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, xn, xm, xc):
        h = self.inp(torch.cat([xn, xm.float(), self.cat(xc)], dim=1))
        for b in self.blocks:
            h = h + b(h)
        return self.out(h).squeeze(-1)


class FTRegressor:
    def __init__(self, num_cols, cat_col, seed=0, epochs=40, batch=1024, lr=1e-3, patience=6, device=None,
                 verbose=False, arch='ft'):
        self.num_cols, self.cat_col, self.seed = list(num_cols), cat_col, seed
        self.epochs, self.batch, self.lr, self.patience = epochs, batch, lr, patience
        self.device, self.verbose, self.arch = pick_device(device), verbose, arch

    def _arrays(self, X):
        v = X[self.num_cols].to_numpy(np.float32)
        m = np.isnan(v)
        v = (np.nan_to_num(v) - self.mu) / self.sd
        v[m] = 0.0
        c = X[self.cat_col].to_numpy(np.int64) + 1
        c[(c < 0) | (c > self.n_cat)] = 0
        return torch.from_numpy(v.astype(np.float32)), torch.from_numpy(m), torch.from_numpy(c)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        v = X[self.num_cols].to_numpy(np.float32)
        self.mu = np.nan_to_num(np.nanmean(v, 0)); self.sd = np.nan_to_num(np.nanstd(v, 0), nan=1.0) + 1e-6
        self.n_cat = int(X[self.cat_col].max()) + 1
        y = np.asarray(y, np.float32); self.ym, self.ys = float(y.mean()), float(y.std() + 1e-6)
        rng = np.random.default_rng(self.seed); idx = rng.permutation(len(y))
        nv = max(1, int(0.1 * len(y))); va, tr = idx[:nv], idx[nv:]
        xn, xm, xc = self._arrays(X); yt = torch.from_numpy((y - self.ym) / self.ys)
        net = _FT if self.arch == 'ft' else _MLP
        self.model = net(len(self.num_cols), self.n_cat).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = nn.L1Loss(); best, state, wait = np.inf, None, 0
        for ep in range(self.epochs):
            self.model.train()
            perm = tr[rng.permutation(len(tr))]
            for i in range(0, len(perm), self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                loss_fn(self.model(xn[b].to(self.device), xm[b].to(self.device), xc[b].to(self.device)),
                        yt[b].to(self.device)).backward()
                opt.step()
            val = float(np.mean(np.abs(self._pred(xn[va], xm[va], xc[va]) - yt[va].numpy())))
            if self.verbose:
                print(f'      epoch {ep + 1:3d}  validation MAE (scaled) {val:.4f}', flush=True)
            if val < best - 1e-4:
                best, wait = val, 0
                state = {k: t.detach().cpu().clone() for k, t in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.model.load_state_dict(state)
        return self

    @torch.no_grad()
    def _pred(self, xn, xm, xc):
        self.model.eval(); out = []
        for i in range(0, len(xn), 8192):
            out.append(self.model(xn[i:i + 8192].to(self.device), xm[i:i + 8192].to(self.device),
                                  xc[i:i + 8192].to(self.device)).float().cpu().numpy())
        return np.concatenate(out)

    def predict(self, X):
        return self._pred(*self._arrays(X)) * self.ys + self.ym
