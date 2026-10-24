from __future__ import annotations
from dataclasses import dataclass
import copy
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _select_device() -> torch.device:
    """Choose CUDA, Apple Metal (MPS), or CPU, with an optional env override."""
    forced = os.environ.get('EVTRUST_TORCH_DEVICE', 'auto').lower()
    if forced != 'auto':
        if forced == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        if forced == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        if forced == 'cpu':
            return torch.device('cpu')
        raise RuntimeError(f"Requested torch device {forced!r} is unavailable")
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class TabularEncoder:
    def __init__(self, numeric: list[str], categorical: list[str]):
        self.numeric = numeric
        self.categorical = categorical
        self.medians = None
        self.means = None
        self.stds = None
        self.maps: dict[str, dict[str, int]] = {}

    def fit(self, df: pd.DataFrame):
        x = df[self.numeric].apply(pd.to_numeric, errors='coerce')
        # A column with no observed value in training (for example driver requests in a dataset
        # that has none) has an undefined median and mean; it is set to zero so it carries no signal.
        self.medians = x.median().fillna(0.0)
        x = x.fillna(self.medians)
        self.means = x.mean().fillna(0.0)
        self.stds = x.std().replace(0, 1).fillna(1)
        for c in self.categorical:
            vals = df[c].astype('string').fillna('__MISSING__').astype(str)
            uniq = sorted(vals.unique().tolist())
            self.maps[c] = {v: i + 1 for i, v in enumerate(uniq)}  # 0 = unseen
        return self

    def transform(self, df: pd.DataFrame):
        xn = df[self.numeric].apply(pd.to_numeric, errors='coerce').fillna(self.medians)
        xn = ((xn - self.means) / self.stds).to_numpy(np.float32)
        xc = []
        for c in self.categorical:
            vals = df[c].astype('string').fillna('__MISSING__').astype(str)
            m = self.maps[c]
            xc.append(vals.map(lambda v: m.get(v, 0)).to_numpy(np.int64))
        xc = np.stack(xc, axis=1) if xc else np.empty((len(df), 0), np.int64)
        return xn, xc

    @property
    def cardinalities(self):
        return [len(self.maps[c]) + 1 for c in self.categorical]


class FTRegressor(nn.Module):
    """Compact FT-Transformer-like regressor for mixed tabular data."""
    def __init__(self, n_num: int, cat_cardinalities: list[int], d_token=96, n_heads=8, n_layers=4, dropout=.1):
        super().__init__()
        self.num_weight = nn.Parameter(torch.randn(n_num, d_token) * 0.02)
        self.num_bias = nn.Parameter(torch.zeros(n_num, d_token))
        self.cat_embeds = nn.ModuleList([nn.Embedding(card, d_token) for card in cat_cardinalities])
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dim_feedforward=4*d_token,
                                           dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(nn.Linear(d_token, d_token), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_token, 1))

    def forward(self, x_num, x_cat):
        tokens = []
        if x_num.shape[1]:
            tokens.append(x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0))
        if x_cat.shape[1]:
            cat_tokens = [emb(x_cat[:, i]).unsqueeze(1) for i, emb in enumerate(self.cat_embeds)]
            tokens.append(torch.cat(cat_tokens, dim=1))
        x = torch.cat(tokens, dim=1)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        return self.head(self.norm(x[:, 0])).squeeze(-1)


@dataclass
class TorchTabularModel:
    encoder: TabularEncoder
    model: FTRegressor
    device: torch.device

    def predict(self, df: pd.DataFrame, batch_size: int = 4096) -> np.ndarray:
        xn, xc = self.encoder.transform(df)
        ds = TensorDataset(torch.from_numpy(xn), torch.from_numpy(xc))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=self.device.type == 'cuda')
        preds = []
        self.model.eval()
        with torch.inference_mode():
            for nb, cb in loader:
                nb = nb.to(self.device, non_blocking=True)
                cb = cb.to(self.device, non_blocking=True)
                preds.append(self.model(nb, cb).detach().cpu().numpy())
        return np.concatenate(preds)


def fit_torch_ft(train: pd.DataFrame, y: np.ndarray, validation: pd.DataFrame, y_val: np.ndarray,
                 numeric: list[str], categorical: list[str], params: dict, seed: int) -> TorchTabularModel:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _select_device()
    enc = TabularEncoder(numeric, categorical).fit(train)
    xtr_n, xtr_c = enc.transform(train)
    xva_n, xva_c = enc.transform(validation)
    ytr = np.asarray(y, np.float32)
    yva = np.asarray(y_val, np.float32)

    model = FTRegressor(len(numeric), enc.cardinalities, params['d_token'], params['n_heads'], params['n_layers'], params['dropout']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    train_ds = TensorDataset(torch.from_numpy(xtr_n), torch.from_numpy(xtr_c), torch.from_numpy(ytr))
    batch_size = int(os.environ.get('EVTRUST_BATCH_SIZE', params['batch_size']))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=device.type == 'cuda', num_workers=int(os.environ.get('EVTRUST_NUM_WORKERS', '0' if device.type in {'cpu','mps'} else '2')))
    amp_enabled = bool(params.get('amp', True) and device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler() if amp_enabled else None

    best_state = copy.deepcopy(model.state_dict())
    best_mae = float('inf')
    stale = 0
    for epoch in range(params['epochs']):
        model.train()
        for nb, cb, yb in train_loader:
            nb, cb, yb = nb.to(device), cb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            if amp_enabled:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred = model(nb, cb)
                    loss = loss_fn(pred, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(nb, cb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        tm = TorchTabularModel(enc, model, device)
        pred_val = tm.predict(validation)
        mae = float(np.mean(np.abs(pred_val - yva)))
        if mae < best_mae - 1e-5:
            best_mae = mae
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= params['patience']:
            break
    model.load_state_dict(best_state)
    return TorchTabularModel(enc, model, device)
