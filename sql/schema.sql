-- PostGIS schema for the Highveld CO2 project.
-- Tables are listed parents-first so foreign keys resolve. All geometries are WGS84 points.
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS source_complexes (
    complex_id        INTEGER PRIMARY KEY,
    name              TEXT NOT NULL,
    n_sources         INTEGER NOT NULL,
    total_capacity_mw DOUBLE PRECISION,
    geom              GEOMETRY(Point, 4326) NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    source_id          INTEGER PRIMARY KEY,
    name               TEXT NOT NULL,
    fuel               TEXT,
    capacity_mw        DOUBLE PRECISION,
    complex_id         INTEGER NOT NULL REFERENCES source_complexes(complex_id),
    geolocation_source TEXT,
    geom               GEOMETRY(Point, 4326) NOT NULL
);

-- One row per OCO-2 overpass of the study box (valid or not)
CREATE TABLE IF NOT EXISTS passes (
    pass_id        TEXT PRIMARY KEY,
    obs_date       DATE NOT NULL,
    t_start        TIMESTAMPTZ,
    n_good         INTEGER,
    n_far          INTEGER,
    extent_km      DOUBLE PRECISION,
    background_ppm DOUBLE PRECISION,
    sigma_ppm      DOUBLE PRECISION,
    unc_median_ppm DOUBLE PRECISION,
    is_valid       BOOLEAN NOT NULL
);

-- One row per 5 km along-track bin of a valid pass
CREATE TABLE IF NOT EXISTS bins (
    pass_id           TEXT NOT NULL REFERENCES passes(pass_id),
    bin_idx           INTEGER NOT NULL,
    n                 INTEGER NOT NULL,
    mean_anomaly_ppm  DOUBLE PRECISION,
    se_ppm            DOUBLE PRECISION,
    z                 DOUBLE PRECISION,
    dist_source_km    DOUBLE PRECISION,
    flagged           BOOLEAN NOT NULL,
    geom              GEOMETRY(Point, 4326) NOT NULL,
    PRIMARY KEY (pass_id, bin_idx)
);

-- One row per enhancement event (merged run of flagged bins)
CREATE TABLE IF NOT EXISTS events (
    event_id          INTEGER PRIMARY KEY,
    pass_id           TEXT NOT NULL REFERENCES passes(pass_id),
    obs_date          DATE NOT NULL,
    n_bins            INTEGER,
    length_km         DOUBLE PRECISION,
    peak_anomaly_ppm  DOUBLE PRECISION,
    mean_anomaly_ppm  DOUBLE PRECISION,
    peak_z            DOUBLE PRECISION,
    nearest_source    TEXT,
    dist_source_km    DOUBLE PRECISION,
    attributed        BOOLEAN,
    complex_id        INTEGER REFERENCES source_complexes(complex_id),
    hotspot_id        INTEGER,
    geom              GEOMETRY(Point, 4326) NOT NULL
);

CREATE TABLE IF NOT EXISTS complex_stats (
    complex_id      INTEGER PRIMARY KEY REFERENCES source_complexes(complex_id),
    observed_passes INTEGER,
    event_passes    INTEGER,
    frequency       DOUBLE PRECISION,
    ci_low          DOUBLE PRECISION,
    ci_high         DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS bins_geom_gix    ON bins    USING GIST (geom);
CREATE INDEX IF NOT EXISTS events_geom_gix  ON events  USING GIST (geom);
CREATE INDEX IF NOT EXISTS sources_geom_gix ON sources USING GIST (geom);
CREATE INDEX IF NOT EXISTS events_date_ix   ON events (obs_date);
CREATE INDEX IF NOT EXISTS bins_flag_ix     ON bins (flagged) WHERE flagged;
