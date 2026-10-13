"""GR4J (Perrin, Michel & Andreassian, 2003): the standard conceptual
rainfall-runoff baseline, four parameters, daily step.

    x1  production store capacity (mm)
    x2  groundwater exchange coefficient (mm/day; < 0 loses water)
    x3  routing store capacity (mm)
    x4  unit hydrograph time base (days)

Rain net of potential evaporation fills a soil (production) store; what it
does not hold, plus percolation, is split 90/10 between a slow path (unit
hydrograph UH1, then a non-linear routing store) and a fast path (UH2), with
a groundwater exchange term on both. The step order follows airGR's
reference implementation. Compiled with numba: calibrating one catchment
takes thousands of 25-year simulations.

Calibration maximizes NSE over the training years with differential
evolution, after a one-year warm-up driven by forcing only.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.optimize import differential_evolution

# search space: log(x1), x2, log(x3), x4
BOUNDS = [(np.log(10.0), np.log(3000.0)), (-10.0, 5.0), (np.log(1.0), np.log(1000.0)), (0.5, 10.0)]


@njit(cache=True)
def _s_curve1(t, x4):
    if t <= 0:
        return 0.0
    if t < x4:
        return (t / x4) ** 2.5
    return 1.0


@njit(cache=True)
def _s_curve2(t, x4):
    if t <= 0:
        return 0.0
    if t <= x4:
        return 0.5 * (t / x4) ** 2.5
    if t < 2 * x4:
        return 1.0 - 0.5 * (2.0 - t / x4) ** 2.5
    return 1.0


@njit(cache=True)
def unit_hydrographs(x4):
    n1 = max(1, int(np.ceil(x4)))
    n2 = max(1, int(np.ceil(2 * x4)))
    uh1 = np.empty(n1)
    uh2 = np.empty(n2)
    for i in range(n1):
        uh1[i] = _s_curve1(i + 1.0, x4) - _s_curve1(float(i), x4)
    for i in range(n2):
        uh2[i] = _s_curve2(i + 1.0, x4) - _s_curve2(float(i), x4)
    return uh1, uh2


@njit(cache=True)
def simulate(p, e, x1, x2, x3, x4, s0=0.3, r0=0.5):
    """Daily flow (mm/day) and the stores, from rain p and potential evaporation e.

    Returns q, actual evaporation, final production store, final routing
    store, water still in the unit-hydrograph buffers, and total exchange.
    """
    n = len(p)
    q = np.empty(n)
    aet = np.empty(n)
    s = s0 * x1
    r = r0 * x3
    uh1, uh2 = unit_hydrographs(x4)
    b1 = np.zeros(len(uh1))
    b2 = np.zeros(len(uh2))
    exch_total = 0.0
    for t in range(n):
        pt, et = p[t], e[t]
        if pt >= et:
            pn, en = pt - et, 0.0
            w = np.tanh(min(pn / x1, 13.0))
            ps = x1 * (1.0 - (s / x1) ** 2) * w / (1.0 + s / x1 * w)
            es = 0.0
            aet[t] = et
        else:
            pn, en = 0.0, et - pt
            w = np.tanh(min(en / x1, 13.0))
            es = s * (2.0 - s / x1) * w / (1.0 + (1.0 - s / x1) * w)
            ps = 0.0
            aet[t] = pt + es
        s = s - es + ps
        perc = s * (1.0 - (1.0 + (4.0 / 9.0 * s / x1) ** 4) ** -0.25)
        s -= perc
        pr = perc + (pn - ps)
        for k in range(len(uh1) - 1):
            b1[k] = b1[k + 1] + uh1[k] * 0.9 * pr
        b1[len(uh1) - 1] = uh1[len(uh1) - 1] * 0.9 * pr
        for k in range(len(uh2) - 1):
            b2[k] = b2[k + 1] + uh2[k] * 0.1 * pr
        b2[len(uh2) - 1] = uh2[len(uh2) - 1] * 0.1 * pr
        q9, q1 = b1[0], b2[0]
        b1[0] = 0.0  # released this step (kept explicit for the mass balance)
        b2[0] = 0.0
        exch = x2 * (r / x3) ** 3.5
        r_new = r + q9 + exch
        exch_r = exch if r_new >= 0 else exch - r_new  # exchange actually applied
        r = max(0.0, r_new)
        qr = r * (1.0 - (1.0 + (r / x3) ** 4) ** -0.25)
        r -= qr
        qd_raw = q1 + exch
        qd = max(0.0, qd_raw)
        exch_d = exch if qd_raw >= 0 else -q1
        q[t] = qr + qd
        exch_total += exch_r + exch_d
    buffered = b1.sum() + b2.sum()
    return q, aet, s, r, buffered, exch_total


def to_params(z) -> tuple[float, float, float, float]:
    return float(np.exp(z[0])), float(z[1]), float(np.exp(z[2])), float(z[3])


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    m = np.isfinite(obs)
    o, s = obs[m], sim[m]
    return float(1.0 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2))


def calibrate(p, e, q_obs, fit_mask, seed: int = 0, maxiter: int = 60) -> tuple[tuple, float]:
    """Parameters maximizing NSE on `fit_mask` days (with observed flow)."""
    target = q_obs.copy()
    target[~fit_mask] = np.nan

    def loss(z):
        sim = simulate(p, e, *to_params(z))[0]
        v = nse(target, sim)
        return 1.0 - v if np.isfinite(v) else 1e6

    res = differential_evolution(loss, BOUNDS, seed=seed, maxiter=maxiter, popsize=12, tol=1e-6,
                                 polish=True, updating="deferred", workers=1)  # fmt: skip
    return to_params(res.x), 1.0 - float(res.fun)
