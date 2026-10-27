"""Tests for binning, event merging, attribution statistics and compositing, on hand-built data."""
import numpy as np
import pandas as pd
import pytest

from highveld_co2.attribute import control_test, link_events, wilson_ci
from highveld_co2.composite import radial_profile
from highveld_co2.config import Params
from highveld_co2.events import bin_passes, group_events
from highveld_co2.sources import SourceIndex

P = Params()


def _index():
    src = pd.DataFrame({"name": ["Plant A"], "x_m": [500_000.0], "y_m": [7_000_000.0], "complex_id": [0]})
    return SourceIndex(src)


def _soundings(plume_bins=(), amp=2.0, seed=0, n_per_bin=20, n_bins=60):
    """One synthetic pass, 5 km bins, noise 0.7 ppm, optional +amp ppm in the given bins."""
    rng = np.random.default_rng(seed)
    s = np.repeat(np.arange(n_bins) * 5 + 2.5, n_per_bin)
    anom = rng.normal(0, 0.7, s.size)
    in_plume = np.isin(np.floor(s / 5).astype(int), list(plume_bins))
    anom = anom + np.where(in_plume, amp, 0.0)
    return pd.DataFrame({
        "pass_id": "p0", "s_km": s, "anomaly_ppm": anom, "sigma_ppm": 0.7, "xco2": 415 + anom,
        "lat": -26.0, "lon": 28.0, "x_m": 500_000.0 + s * 1000, "y_m": 7_000_000.0,
        "time": pd.Timestamp("2023-07-15 11:30", tz="UTC"),
    })


def test_a_plume_becomes_exactly_one_event():
    bins = bin_passes(_soundings(plume_bins=range(20, 24)), _index(), P)
    ev = group_events(bins, P)
    assert len(ev) == 1
    assert ev["n_bins"].iloc[0] == 4 and ev["length_km"].iloc[0] == 20
    assert 1.5 < ev["peak_anomaly_ppm"].iloc[0] < 2.6


def test_two_separated_plumes_make_two_events():
    bins = bin_passes(_soundings(plume_bins=[10, 11, 40, 41]), _index(), P)
    assert len(group_events(bins, P)) == 2


def test_pure_noise_gives_almost_no_events():
    flagged = sum(bin_passes(_soundings(seed=s), _index(), P)["flagged"].sum() for s in range(30))
    assert flagged <= 3        # 1800 bins at z>=3 plus >=1 ppm: essentially none


def test_bins_below_minimum_count_are_dropped():
    df = _soundings(n_per_bin=3)
    assert bin_passes(df, _index(), P).empty


def test_link_events_attributes_by_distance():
    ev = pd.DataFrame({"x_m": [500_000 + 10_000, 500_000 + 100_000], "y_m": [7_000_000.0] * 2})
    out = link_events(ev, _index(), P)
    assert out["attributed"].tolist() == [True, False]
    assert out["nearest_source"].iloc[0] == "Plant A"


def test_wilson_ci_known_values():
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson_ci(5, 10)
    assert (lo, hi) == (pytest.approx(0.2366, abs=1e-3), pytest.approx(0.7634, abs=1e-3))
    assert all(np.isnan(wilson_ci(0, 0)))


def _bins_for_control(rng, n_pass=40):
    rows = []
    for p in range(n_pass):
        for _ in range(10):      # near bins: 20 % flagged
            rows.append({"pass_id": f"p{p}", "dist_source_km": rng.uniform(0, 25), "flagged": rng.random() < 0.20})
        for _ in range(40):      # far bins: 2 % flagged
            rows.append({"pass_id": f"p{p}", "dist_source_km": rng.uniform(70, 150), "flagged": rng.random() < 0.02})
    return pd.DataFrame(rows)


def test_control_test_detects_enrichment_and_gives_a_bootstrap_interval():
    c = control_test(_bins_for_control(np.random.default_rng(0)), P, n_boot=500)
    assert c["near_rate"] > 5 * c["far_rate"]
    assert c["fisher_p"] < 1e-6
    lo, hi = c["rate_ratio_ci"]
    assert lo > 1 and lo < c["rate_ratio"] < hi


def test_control_test_is_calibrated_when_there_is_no_enrichment():
    """With the same flag rate everywhere, p < 0.01 should be rare (about 1 run in 100).

    A single seed would make this test fail by chance now and then, so run 20 and allow a few.
    (A separate 400-run check gave 0.7 % below 0.01 and 3.5 % below 0.05.)
    """
    small_p = 0
    for seed in range(20):
        rng = np.random.default_rng(1000 + seed)
        b = _bins_for_control(rng)
        b["flagged"] = rng.random(len(b)) < 0.05
        c = control_test(b, P, n_boot=20)
        assert 0.3 < c["rate_ratio"] < 3.5
        small_p += c["fisher_p"] < 0.01
    assert small_p <= 3


def test_radial_profile_recovers_a_decaying_signal():
    rng = np.random.default_rng(3)
    d = rng.uniform(0, 100, 6000)
    bins = pd.DataFrame({"pass_id": rng.integers(0, 60, d.size).astype(str), "dist_source_km": d,
                         "mean_anomaly_ppm": 1.0 * np.exp(-d / 20) + rng.normal(0, 0.2, d.size)})
    prof = radial_profile(bins, edges_km=(0, 10, 30, 60, 100), n_boot=300)
    m = prof["mean_anomaly_ppm"].values
    assert (np.diff(m) < 0).all()                      # monotonically decreasing
    assert prof["ci_low"].iloc[0] < m[0] < prof["ci_high"].iloc[0]
    assert abs(m[-1]) < 0.1
