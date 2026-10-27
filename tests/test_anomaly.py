"""Tests for pass segmentation, along-track coordinate and background estimation."""
import numpy as np
import pandas as pd

from highveld_co2.anomaly import MAD_TO_SIGMA, along_track_km, assign_passes, estimate_background


def _times(starts_s, n=50, step=0.3):
    t = np.concatenate([s + np.arange(n) * step for s in starts_s])
    return pd.DataFrame({"t_s": t, "time": pd.to_datetime(t, unit="s", utc=True)})


def test_assign_passes_splits_on_time_gap_and_numbers_within_day():
    base = pd.Timestamp("2023-07-15 10:00", tz="UTC").timestamp()
    df = assign_passes(_times([base, base + 6000, base + 12000]), gap_s=300)
    assert df["pass_id"].nunique() == 3
    assert sorted(df["pass_id"].unique()) == ["2023-07-15_0", "2023-07-15_1", "2023-07-15_2"]


def test_assign_passes_keeps_one_pass_together():
    base = pd.Timestamp("2023-07-15 10:00", tz="UTC").timestamp()
    assert assign_passes(_times([base], n=500), gap_s=300)["pass_id"].nunique() == 1


def test_along_track_recovers_length_and_orientation():
    # a straight 100 km track heading north-west, 8 footprints wide
    u = np.linspace(0, 100_000, 400)
    x = 500_000 - 0.14 * u + np.random.default_rng(0).normal(0, 3000, 400)
    y = 7_000_000 + 0.99 * u
    t = np.linspace(0, 15, 400)
    s = along_track_km(x, y, t)
    assert s.min() == 0
    assert 95 < s.max() < 105
    assert np.corrcoef(s, t)[0, 1] > 0.99          # oriented along increasing time
    s_rev = along_track_km(x, y, t[::-1])
    assert np.corrcoef(s_rev, t[::-1])[0, 1] > 0.99


def test_running_background_follows_a_gradient_that_the_median_misses():
    rng = np.random.default_rng(1)
    s = np.linspace(0, 400, 2000)
    truth = 410 + 0.01 * s                                    # +4 ppm over the pass
    xco2 = truth + rng.normal(0, 0.6, s.size)
    xco2 = xco2 + np.where((s > 180) & (s < 220), 3.0, 0.0)   # a plume in the middle
    far = ~((s > 150) & (s < 250))                            # plume zone excluded from the background
    bg_run, sig = estimate_background(s, xco2, far, "running", 100.0, 0.3)
    bg_med, _ = estimate_background(s, xco2, far, "median", 100.0, 0.3)
    assert np.sqrt(np.mean((bg_run - truth) ** 2)) < 0.3
    assert np.sqrt(np.mean((bg_med - truth) ** 2)) > 1.0      # the median cannot follow a gradient
    assert 0.5 < sig < 0.8                                    # close to the true noise of 0.6


def test_sigma_is_robust_to_outliers_and_floored():
    rng = np.random.default_rng(2)
    s = np.linspace(0, 300, 1500)
    x = 415 + rng.normal(0, 0.6, s.size)
    x[:30] += 15.0                                            # 2 % wild outliers
    far = np.ones(s.size, bool)
    _, sig = estimate_background(s, x, far, "median", 100.0, 0.3)
    assert 0.5 < sig < 0.75                                   # a plain std would be ~2.5
    _, floored = estimate_background(s, np.full(s.size, 415.0), far, "median", 100.0, 0.3)
    assert floored == 0.3
    assert MAD_TO_SIGMA == 1.4826
