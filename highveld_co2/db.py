"""PostgreSQL/PostGIS persistence and example spatial queries.

    docker compose up -d            # starts PostGIS (see docker-compose.yml)
    python -m highveld_co2.pipeline --raw data/raw/oco2 --out results --db
"""
import os

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from .config import DEFAULT_PARAMS, ROOT

SCHEMA_SQL = (ROOT / "sql" / "schema.sql").read_text()
DEFAULT_URL = "postgresql+psycopg2://co2:co2@localhost:5432/co2"


def get_engine():
    load_dotenv(ROOT / ".env")
    return create_engine(os.getenv("DATABASE_URL", DEFAULT_URL))


def _null_safe(df: pd.DataFrame) -> pd.DataFrame:
    """NaN -> None for every non-geometry column.

    Without this, NaN floats are written as the value 'NaN' (not NULL), which silently breaks
    AVG(), COUNT(col) and any 'IS NULL' test. astype(object) also turns numpy ints/bools into
    plain Python values, which the database driver can adapt.
    """
    out = df.copy()
    for c in out.columns:
        if c == "geom":
            continue
        out[c] = out[c].astype(object).where(out[c].notna(), None)
    return out


def _points(df, lon="lon", lat="lat", cols=()):
    g = gpd.GeoDataFrame(df[list(cols)].copy(), geometry=gpd.points_from_xy(df[lon], df[lat]), crs=4326)
    return _null_safe(g.rename_geometry("geom"))


def save_all(sources, complexes, passes, bins, events, cstats, engine=None):
    """Create the schema if needed, replace all table contents, then load. Idempotent."""
    engine = engine or get_engine()
    with engine.begin() as con:
        con.execute(text(SCHEMA_SQL))
        con.execute(text("TRUNCATE complex_stats, events, bins, passes, sources, source_complexes CASCADE"))

    _points(complexes, "longitude", "latitude",
            ["complex_id", "name", "n_sources", "total_capacity_mw"]).to_postgis(
        "source_complexes", engine, if_exists="append", index=False)
    _points(sources, "longitude", "latitude",
            ["source_id", "name", "fuel", "capacity_mw", "complex_id", "geolocation_source"]).to_postgis(
        "sources", engine, if_exists="append", index=False)

    pcols = ["pass_id", "obs_date", "t_start", "n_good", "n_far", "extent_km", "background_ppm",
             "sigma_ppm", "unc_median_ppm", "is_valid"]
    _null_safe(passes[pcols]).to_sql("passes", engine, if_exists="append", index=False)

    _points(bins, cols=["pass_id", "bin_idx", "n", "mean_anomaly_ppm", "se_ppm", "z", "dist_source_km",
                        "flagged"]).to_postgis("bins", engine, if_exists="append", index=False, chunksize=5000)

    if len(events):
        ecols = ["event_id", "pass_id", "obs_date", "n_bins", "length_km", "peak_anomaly_ppm",
                 "mean_anomaly_ppm", "peak_z", "nearest_source", "dist_source_km", "attributed",
                 "complex_id", "hotspot_id"]
        _points(events, cols=ecols).to_postgis("events", engine, if_exists="append", index=False)

    scols = ["complex_id", "observed_passes", "event_passes", "frequency", "ci_low", "ci_high"]
    _null_safe(cstats[scols]).to_sql("complex_stats", engine, if_exists="append", index=False)
    return {t: int(pd.read_sql(text(f"SELECT COUNT(*) FROM {t}"), engine).iloc[0, 0])
            for t in ("source_complexes", "sources", "passes", "bins", "events", "complex_stats")}


EXAMPLE_QUERIES = {
    # detection frequency per complex, best-sampled first
    "complex_frequency": """
        SELECT c.name, s.observed_passes, s.event_passes,
               ROUND(s.frequency::numeric, 3) AS frequency,
               ROUND(s.ci_low::numeric, 3) AS ci_low, ROUND(s.ci_high::numeric, 3) AS ci_high
        FROM complex_stats s JOIN source_complexes c USING (complex_id)
        ORDER BY s.observed_passes DESC;""",
    # events within :km of a point (geography => true metres on the ellipsoid)
    "events_near_point": """
        SELECT event_id, obs_date, ROUND(peak_anomaly_ppm::numeric, 2) AS peak_ppm,
               ROUND((ST_Distance(geom::geography,
                      ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography) / 1000)::numeric, 1) AS km
        FROM events
        WHERE ST_DWithin(geom::geography, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography, :km * 1000)
        ORDER BY km;""",
    # events that no known source explains
    "unattributed_events": """
        SELECT event_id, obs_date, ROUND(peak_anomaly_ppm::numeric, 2) AS peak_ppm,
               nearest_source, ROUND(dist_source_km::numeric, 1) AS dist_km
        FROM events WHERE NOT attributed ORDER BY peak_anomaly_ppm DESC LIMIT 20;""",
    # seasonality of events
    "events_by_month": """
        SELECT date_trunc('month', obs_date)::date AS month, COUNT(*) AS events,
               ROUND(AVG(peak_anomaly_ppm)::numeric, 2) AS mean_peak_ppm
        FROM events GROUP BY 1 ORDER BY 1;""",
    # the control test in SQL: flag rate near vs far. :near and :far must equal Params.near_km / far_km
    # (defaults below), otherwise it will not reproduce the Python numbers.
    "control_test": """
        SELECT CASE WHEN dist_source_km <= :near THEN 'near' WHEN dist_source_km >= :far THEN 'far' END AS zone,
               COUNT(*) AS bins, SUM(flagged::int) AS flagged,
               ROUND(AVG(flagged::int)::numeric, 4) AS flag_rate
        FROM bins WHERE dist_source_km <= :near OR dist_source_km >= :far GROUP BY 1 ORDER BY 1;""",
}


# bind-parameter defaults, taken from the analysis defaults so SQL and Python agree out of the box
QUERY_DEFAULTS = {"control_test": {"near": DEFAULT_PARAMS.near_km, "far": DEFAULT_PARAMS.far_km}}


def run_query(name, engine=None, **params):
    """Run one of EXAMPLE_QUERIES. Extra keyword arguments are SQL bind parameters (:lon, :lat, :km ...)."""
    engine = engine or get_engine()
    with engine.connect() as con:
        return pd.read_sql(text(EXAMPLE_QUERIES[name]), con, params={**QUERY_DEFAULTS.get(name, {}), **params})
