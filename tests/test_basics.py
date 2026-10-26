"""Unit tests for geography, sources and the download helpers (no network, no data files needed)."""
from datetime import date

import numpy as np
import pytest

from highveld_co2.download import cycle_days, date_from_filename, parse_version, pick_latest_version
from highveld_co2.geo import haversine_km, to_lonlat, to_metric
from highveld_co2.sources import SourceIndex, group_complexes, load_sources


# ---------------- geo ----------------
def test_metric_roundtrip():
    lon, lat = np.array([27.5, 29.2, 30.9]), np.array([-27.5, -26.2, -24.6])
    x, y = to_metric(lon, lat)
    lon2, lat2 = to_lonlat(x, y)
    assert np.allclose(lon, lon2, atol=1e-8) and np.allclose(lat, lat2, atol=1e-8)


def test_metric_distance_agrees_with_haversine():
    # 1 degree of latitude and a 1.5 degree east-west step, both inside the study box
    for (lo1, la1, lo2, la2) in [(29.0, -27.0, 29.0, -26.0), (28.0, -26.0, 29.5, -26.0)]:
        x, y = to_metric([lo1, lo2], [la1, la2])
        metric = np.hypot(x[1] - x[0], y[1] - y[0]) / 1000
        assert metric == pytest.approx(haversine_km(lo1, la1, lo2, la2), rel=0.005)


# ---------------- sources ----------------
def test_load_sources_contains_known_plants():
    s = load_sources()
    assert {"Kendal power station", "Majuba power station"} <= set(s["name"])
    assert s["capacity_mw"].isna().sum() >= 1          # Secunda has no capacity figure
    assert s["source_id"].tolist() == list(range(len(s)))


def test_complexes_chain_neighbours_and_order_by_capacity():
    src, comp = group_complexes(load_sources(), eps_km=30)
    assert comp["complex_id"].tolist() == list(range(len(comp)))
    assert comp["total_capacity_mw"].is_monotonic_decreasing
    cid = dict(zip(src["name"], src["complex_id"]))
    assert cid["Matla power station"] == cid["Kriel power station"]      # ~4 km apart
    assert cid["Majuba power station"] != cid["Kendal power station"]    # >100 km apart


def test_source_index_distance_is_zero_at_a_source():
    src, _ = group_complexes(load_sources())
    idx = SourceIndex(src)
    d, i = idx.query(src["x_m"].values, src["y_m"].values)
    assert np.allclose(d, 0)
    assert (i == np.arange(len(src))).all()


# ---------------- download helpers ----------------
def _g(version):
    return {"umm": {"CollectionReference": {"Version": version}}}


def test_parse_version():
    assert parse_version("11.2r") == 11.2 and parse_version("10r") == 10.0 and parse_version("junk") == -1.0


def test_pick_latest_version_keeps_one_version():
    gr = [_g("10r"), _g("11.1r"), _g("11.2r"), _g("11.2r")]
    kept, used = pick_latest_version(gr)
    assert used == "11.2r" and len(kept) == 2
    kept, used = pick_latest_version(gr, wanted="10r")
    assert used == "10r" and len(kept) == 1
    assert pick_latest_version([]) == ([], None)


def test_cycle_days_repeat_every_16_days():
    days = cycle_days(date(2023, 6, 1), [2, 7], date(2023, 6, 1), date(2023, 7, 31))
    assert days == [date(2023, 6, 3), date(2023, 6, 8), date(2023, 6, 19), date(2023, 6, 24),
                    date(2023, 7, 5), date(2023, 7, 10), date(2023, 7, 21), date(2023, 7, 26)]


def test_date_from_filename():
    assert date_from_filename("oco2_LtCO2_230715_B11014Ar_230925185436s.nc4") == date(2023, 7, 15)
    assert date_from_filename("something_else.nc") is None
