"""Integration tests: synthetic files -> reader -> full pipeline (-> optional PostGIS)."""
import os

import numpy as np
import pytest
from netCDF4 import Dataset

from highveld_co2.pipeline import run
from highveld_co2.qc import apply_qc
from highveld_co2.read_oco2 import read_lite_file, read_many
from highveld_co2.synthetic import make_files


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic")
    make_files(d, start="2021-06-01", n_days=400, seed=3)
    return d


def test_reader_returns_box_soundings_and_flags_synthetic(synth_dir):
    df = read_many(sorted(synth_dir.glob("*.nc4")))
    assert df.attrs["synthetic"] is True
    assert df["lon"].between(27, 31).all() and df["lat"].between(-28, -24.5).all()
    assert {"xco2", "xco2_unc", "qf", "t_s", "time", "op_mode", "land_water"} <= set(df.columns)


def test_qc_removes_flagged_soundings_and_reports(synth_dir):
    df = read_many(sorted(synth_dir.glob("*.nc4")))
    clean, rep = apply_qc(df)
    assert (clean["qf"] == 0).all()
    assert rep["n_out"] == len(clean) < rep["n_in"]
    assert rep["removed_quality_flag"] > 0


def test_qc_skips_rules_whose_columns_are_missing(synth_dir):
    df = read_many(sorted(synth_dir.glob("*.nc4")))
    df["op_mode"] = np.nan
    df["land_water"] = np.nan
    _, rep = apply_qc(df)
    assert "skipped" in rep["operation_mode_rule"] and "skipped" in rep["land_water_rule"]


def test_missing_required_variable_gives_actionable_error(tmp_path):
    p = tmp_path / "bad.nc4"
    with Dataset(p, "w") as ds:
        ds.createDimension("n", 3)
        ds.createVariable("latitude", "f8", ("n",))[:] = [1, 2, 3]
    with pytest.raises(KeyError, match="inspect_file"):
        read_lite_file(p)


def test_refuses_to_mix_synthetic_and_real(synth_dir, tmp_path):
    src = sorted(synth_dir.glob("*.nc4"))[0]
    real_like = tmp_path / "oco2_LtCO2_230101_B11014Ar_real.nc4"
    with Dataset(src) as a, Dataset(real_like, "w") as b:       # copy without the synthetic attribute
        b.createDimension("sounding_id", a.dimensions["sounding_id"].size)
        for name, v in a.variables.items():
            nv = b.createVariable(name, v.dtype, v.dimensions)
            nv[:] = v[:]
        sg = b.createGroup("Sounding")
        for name, v in a.groups["Sounding"].variables.items():
            nv = sg.createVariable(name, v.dtype, v.dimensions)
            nv[:] = v[:]
    with pytest.raises(ValueError, match="mix"):
        read_many([src, real_like])


def test_pipeline_end_to_end(synth_dir, tmp_path):
    out = tmp_path / "out"
    r = run(synth_dir, out, make_figures=True, verbose=False)
    assert r["summary"]["synthetic_data"] is True
    assert r["summary"]["n_valid_overpasses"] > 30
    for f in ("summary.json", "events.csv", "bins.csv.gz", "complex_stats.csv", "fig_anomaly_map.png",
              "fig_radial_profile.png", "fig_example_pass.png", "fig_control_test.png", "map.html"):
        assert (out / f).exists(), f
    c = r["control"]
    assert c["near_rate"] > 3 * c["far_rate"]                  # the injected plumes are found near sources
    assert c["fisher_p"] < 1e-6
    # events must sit near sources MORE than chance: compare with how close the average bin is
    baseline = (r["bins"]["dist_source_km"] <= r["params"].attrib_km).mean()
    assert r["events"]["attributed"].mean() > 1.5 * baseline
    top = r["hotspots"].iloc[0]
    assert top["explained_by_source"]                          # the busiest hotspot is the coal cluster
    assert r["profile"]["mean_anomaly_ppm"].iloc[0] > r["profile"]["mean_anomaly_ppm"].iloc[-1]


def test_pipeline_is_deterministic(synth_dir, tmp_path):
    a = run(synth_dir, tmp_path / "a", make_figures=False, verbose=False)
    b = run(synth_dir, tmp_path / "b", make_figures=False, verbose=False)
    assert a["summary"]["n_events"] == b["summary"]["n_events"]
    assert a["control"]["near_flagged"] == b["control"]["near_flagged"]


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"),
                    reason="set TEST_DATABASE_URL to a THROWAWAY PostGIS database to run (tables are truncated)")
def test_postgis_roundtrip(synth_dir, tmp_path):
    from sqlalchemy import create_engine, text

    from highveld_co2.db import save_all
    r = run(synth_dir, tmp_path / "db", make_figures=False, verbose=False)
    eng = create_engine(os.environ["TEST_DATABASE_URL"])
    counts = save_all(r["sources"], r["complexes"], r["passes"], r["bins"], r["events"], r["cstats"], engine=eng)
    assert counts["events"] == len(r["events"]) and counts["bins"] == len(r["bins"])
    with eng.connect() as con:
        nan_rows = con.execute(text("SELECT COUNT(*) FROM complex_stats WHERE frequency = 'NaN'::float8")).scalar()
    assert nan_rows == 0                                       # NaN must be stored as NULL

    # the SQL control test must reproduce the Python one exactly (regression: it once used a stale 30 km)
    from highveld_co2.db import run_query
    sql = run_query("control_test", engine=eng).set_index("zone")
    c = r["control"]
    assert (int(sql.loc["near", "bins"]), int(sql.loc["near", "flagged"])) == (c["near_bins"], c["near_flagged"])
    assert (int(sql.loc["far", "bins"]), int(sql.loc["far", "flagged"])) == (c["far_bins"], c["far_flagged"])
