"""Regional LSTM rainfall-runoff model (Kratzert et al., HESS 2019).

One network for every catchment: daily forcing plus static catchment
attributes in, streamflow out, with no observed flow as input (simulation
mode, the same setting as GR4J and the conceptual models).

- Windows of two years: the first year only warms up the cell state, the
  loss is taken on the second. Inference runs the same way, one calendar
  year at a time after a year of warm-up, so training and prediction see
  identical context lengths.
- NSE* loss: squared error divided by (s_b + 0.1)^2, with s_b the catchment's
  flow standard deviation, so wet catchments with large flows don't dominate.
- Early stopping on the median validation NSE (1981-1990), never on test.
- Inputs are standardized with training-period statistics only.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import FORCING, PERIODS, STATIC, Camels
from .metrics import nse

log = logging.getLogger(__name__)

WARMUP = 365
HORIZON = 365


@dataclass
class Arrays:
    gauges: np.ndarray  # (G,)
    dates: pd.DatetimeIndex  # (T,)
    x: np.ndarray  # (G, T, F) standardized forcing
    s: np.ndarray  # (G, S) standardized attributes
    y: np.ndarray  # (G, T) flow in mm/day, NaN where unobserved
    y_mean: float
    y_std: float

    def year_start(self, year: int) -> int:
        return int(self.dates.searchsorted(pd.Timestamp(year=year, month=1, day=1)))


def static_features(static: pd.DataFrame) -> pd.DataFrame:
    s = static[STATIC].astype(float).copy()
    s["cwc_area"] = np.log10(s["cwc_area"])
    s["num_dams"] = np.log1p(s["num_dams"])
    return s


def build(c: Camels, train_gauges: np.ndarray | None = None) -> Arrays:
    """Dense (gauge, day) arrays, standardized with statistics of the training
    years of `train_gauges` (all gauges by default)."""
    d = c.daily.sort_values(["gauge", "date"])
    gauges = np.sort(d["gauge"].unique())
    dates = pd.DatetimeIndex(np.sort(d["date"].unique()))
    G, T = len(gauges), len(dates)
    assert len(d) == G * T, "every gauge must cover the same days"
    x = d[FORCING].to_numpy(np.float32).reshape(G, T, len(FORCING))
    y = d["q"].to_numpy(np.float32).reshape(G, T)
    fit = np.isin(gauges, gauges if train_gauges is None else train_gauges)
    lo, hi = (dates.searchsorted(pd.Timestamp(p)) for p in PERIODS["train"])
    hi += 1
    xm = x[fit, lo:hi].reshape(-1, x.shape[-1])
    x = (x - xm.mean(0)) / xm.std(0)
    st = static_features(c.static).reindex(gauges)
    sm, ss = st[fit].mean(), st[fit].std()
    s = ((st - sm) / ss).fillna(0.0).to_numpy(np.float32)
    yt = y[fit, lo:hi]
    return Arrays(gauges, dates, x.astype(np.float32), s, y,
                  float(np.nanmean(yt)), float(np.nanstd(yt)))  # fmt: skip


class Net(nn.Module):
    def __init__(self, n_dyn: int, n_stat: int, hidden: int = 256, dropout: float = 0.4):
        super().__init__()
        self.lstm = nn.LSTM(n_dyn + n_stat, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        with torch.no_grad():  # start by remembering (Gers et al. 2000)
            self.lstm.bias_hh_l0[hidden : 2 * hidden].fill_(3.0)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, s[:, None, :].expand(-1, x.shape[1], -1)], dim=-1)
        h, _ = self.lstm(z)
        return self.head(self.drop(h)).squeeze(-1)


def _windows(a: Arrays, gauge_idx: np.ndarray, min_obs: int = 30) -> np.ndarray:
    """(gauge, start) pairs whose second year lies in the training period and
    has at least `min_obs` observed days."""
    lo, hi = (a.dates.searchsorted(pd.Timestamp(p)) for p in PERIODS["train"])
    out = []
    obs = np.isfinite(a.y)
    for g in gauge_idx:
        c = np.r_[0, np.cumsum(obs[g])]
        starts = np.arange(max(lo - WARMUP, 0), hi + 1 - WARMUP - HORIZON + 1)
        n = c[starts + WARMUP + HORIZON] - c[starts + WARMUP]
        out.append(np.c_[np.full((n >= min_obs).sum(), g), starts[n >= min_obs]])
    return np.concatenate(out)


def predict(models: list[Net], a: Arrays, years, gauge_idx=None, device="cuda") -> np.ndarray:
    """Flow (mm/day) for whole calendar years, each after a year of warm-up;
    NaN elsewhere. The ensemble mean if several models are given."""
    gauge_idx = np.arange(len(a.gauges)) if gauge_idx is None else np.asarray(gauge_idx)
    out = np.full((len(gauge_idx), len(a.dates)), np.nan, np.float32)
    xs = torch.from_numpy(a.s[gauge_idx]).to(device)
    for m in models:
        m.eval()
    with torch.no_grad():
        for year in years:
            lo, mid, hi = a.year_start(year - 1), a.year_start(year), a.year_start(year + 1)
            xb = torch.from_numpy(a.x[gauge_idx, lo:hi]).to(device)
            p = torch.stack([m(xb, xs) for m in models]).mean(0)[:, mid - lo :]
            out[:, mid:hi] = (p * a.y_std + a.y_mean).clamp_min(0).cpu().numpy()
    return out


def median_nse(a: Arrays, pred: np.ndarray, gauge_idx, period: str) -> float:
    lo, hi = (a.dates.searchsorted(pd.Timestamp(p)) for p in PERIODS[period])
    scores = []
    for row, g in enumerate(gauge_idx):
        o = a.y[g, lo : hi + 1]
        if np.isfinite(o).sum() >= 365:
            scores.append(nse(o, pred[row, lo : hi + 1]))
    return float(np.nanmedian(scores))


def train(
    a: Arrays,
    gauge_idx: np.ndarray | None = None,
    seed: int = 0,
    epochs: int = 30,
    windows_per_epoch: int = 6000,
    batch: int = 128,
    hidden: int = 256,
    device: str = "cuda",
) -> tuple[Net, list[dict]]:
    """One ensemble member, early-stopped on the validation years of the
    same (training) catchments."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gauge_idx = np.arange(len(a.gauges)) if gauge_idx is None else np.asarray(gauge_idx)
    win = _windows(a, gauge_idx)
    x = torch.from_numpy(a.x).to(device)
    s = torch.from_numpy(a.s).to(device)
    y = torch.from_numpy((a.y - a.y_mean) / a.y_std).to(device)
    lo, hi = (a.dates.searchsorted(pd.Timestamp(p)) for p in PERIODS["train"])
    sd = np.nanstd(np.where(np.isfinite(a.y[:, lo : hi + 1]), a.y[:, lo : hi + 1], np.nan), axis=1)
    weight = torch.from_numpy(1.0 / (sd / a.y_std + 0.1) ** 2).float().to(device)
    in_train = torch.zeros(len(a.dates), dtype=torch.bool, device=device)
    in_train[lo : hi + 1] = True

    net = Net(a.x.shape[-1], a.s.shape[-1], hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[10, 25], gamma=0.5)
    span = torch.arange(WARMUP + HORIZON, device=device)
    valid_years = range(1981, 1991)
    best, best_score, history = None, -np.inf, []
    for epoch in range(epochs):
        net.train()
        pick = win[rng.choice(len(win), windows_per_epoch, replace=len(win) < windows_per_epoch)]
        losses = []
        for i in range(0, len(pick), batch):
            g = torch.from_numpy(pick[i : i + batch, 0]).to(device)
            t = torch.from_numpy(pick[i : i + batch, 1]).to(device)[:, None] + span
            pred = net(x[g[:, None], t], s[g])[:, WARMUP:]
            target = y[g[:, None], t[:, WARMUP:]]
            mask = torch.isfinite(target) & in_train[t[:, WARMUP:]]
            err = (pred - torch.nan_to_num(target)) ** 2 * weight[g][:, None]
            loss = (err * mask).sum() / mask.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        score = median_nse(a, predict([net], a, valid_years, gauge_idx, device), gauge_idx, "valid")
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "valid_nse": score})
        log.info(
            "seed %d epoch %d loss %.4f valid median NSE %.3f",
            seed,
            epoch + 1,
            np.mean(losses),
            score,
        )
        if score > best_score:
            best, best_score = copy.deepcopy(net.state_dict()), score
    net.load_state_dict(best)
    return net, history
