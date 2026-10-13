"""GR4J against its definition: unit hydrographs, water balance, recession,
and calibration recovering known parameters."""

import numpy as np
import pytest

from varuna.bench.gr4j import calibrate, simulate, unit_hydrographs


def forcing(n=3 * 365, seed=0):
    rng = np.random.default_rng(seed)
    wet = rng.random(n) < 0.35
    p = np.where(wet, rng.gamma(0.8, 12.0, n), 0.0)
    e = 3.0 + 2.0 * np.sin(np.arange(n) * 2 * np.pi / 365)
    return p, e


@pytest.mark.parametrize("x4", [0.7, 1.0, 2.3, 6.5])
def test_unit_hydrographs_sum_to_one(x4):
    uh1, uh2 = unit_hydrographs(x4)
    assert uh1.sum() == pytest.approx(1.0) and uh2.sum() == pytest.approx(1.0)
    assert (uh1 >= 0).all() and (uh2 >= 0).all()
    assert len(uh2) == int(np.ceil(2 * x4))


def test_water_is_conserved_without_groundwater_exchange():
    p, e = forcing()
    x1, x3 = 350.0, 90.0
    q, aet, s, r, buffered, exch = simulate(p, e, x1, 0.0, x3, 1.7)
    start = 0.3 * x1 + 0.5 * x3
    end = s + r + buffered
    assert exch == 0.0
    assert p.sum() == pytest.approx(aet.sum() + q.sum() + (end - start), rel=1e-9)


def test_exchange_is_accounted_for():
    p, e = forcing(seed=1)
    x1, x3 = 300.0, 60.0
    q, aet, s, r, buffered, exch = simulate(p, e, x1, -1.5, x3, 2.0)
    start = 0.3 * x1 + 0.5 * x3
    assert exch < 0  # the catchment loses water to the aquifer
    assert p.sum() + exch == pytest.approx(
        aet.sum() + q.sum() + (s + r + buffered - start), rel=1e-9
    )


def test_flow_recedes_without_rain():
    p = np.zeros(200)
    e = np.full(200, 2.0)
    q = simulate(p, e, 400.0, 0.0, 120.0, 2.0, 0.8, 0.9)[0]
    assert (np.diff(q[5:]) <= 1e-12).all() and q[-1] < q[5]


def test_calibration_recovers_a_planted_catchment():
    p, e = forcing(n=4 * 365, seed=3)
    true = (420.0, -0.8, 75.0, 1.9)
    q = simulate(p, e, *true)[0]
    fit = np.zeros(len(p), bool)
    fit[365:] = True  # one warm-up year, as in the experiments
    params, score = calibrate(p, e, q, fit, maxiter=40)
    assert score > 0.995
    assert params[0] == pytest.approx(true[0], rel=0.15)
    assert params[3] == pytest.approx(true[3], rel=0.15)
