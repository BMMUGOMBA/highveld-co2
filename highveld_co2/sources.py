"""Known CO2 point sources: loading, grouping into complexes, fast nearest-source lookup."""
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from .config import DEFAULT_PARAMS, REF_DIR
from .geo import to_lonlat, to_metric


def load_sources(ref_dir=REF_DIR, min_capacity_mw: float = 0.0, include_extra: bool = True) -> gpd.GeoDataFrame:
    """Read the reference CSVs into one GeoDataFrame (EPSG:4326) with metric x/y columns.

    `extra_sources.csv` holds sources that are not power plants (e.g. Secunda). Their
    capacity is blank, so min_capacity_mw only filters rows that have a capacity.
    """
    frames = [pd.read_csv(ref_dir / "power_plants_highveld.csv")]
    extra = ref_dir / "extra_sources.csv"
    if include_extra and extra.exists():
        frames.append(pd.read_csv(extra))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["capacity_mw"].isna() | (df["capacity_mw"] >= min_capacity_mw)].reset_index(drop=True)
    df.insert(0, "source_id", np.arange(len(df)))
    x, y = to_metric(df["longitude"].values, df["latitude"].values)
    df["x_m"], df["y_m"] = x, y
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)


def group_complexes(sources: gpd.GeoDataFrame, eps_km: float = DEFAULT_PARAMS.complex_eps_km):
    """Chain sources closer than eps_km into complexes.

    The satellite cannot separate sources a few km apart (a 5 km bin and a ~10 km wide swath),
    so attribution is only meaningful at complex level. DBSCAN with min_samples=1 is
    single-linkage: A-B within eps and B-C within eps puts A, B, C in one complex.

    Returns (sources with complex_id, complexes table). Complex ids are ordered by total
    capacity, largest first, so id 0 is the biggest complex.
    """
    xy = sources[["x_m", "y_m"]].values
    raw = DBSCAN(eps=eps_km * 1000.0, min_samples=1).fit_predict(xy)
    src = sources.copy()
    src["_raw"] = raw

    rows = []
    for raw_id, g in src.groupby("_raw"):
        cap = g["capacity_mw"].fillna(0).sum()
        lead = g.sort_values("capacity_mw", ascending=False, na_position="last").iloc[0]
        cx, cy = g["x_m"].mean(), g["y_m"].mean()
        lon, lat = to_lonlat(cx, cy)
        rows.append({"_raw": raw_id, "n_sources": len(g), "total_capacity_mw": cap,
                     "name": (lead["name"] + " complex") if len(g) > 1 else lead["name"],
                     "longitude": float(lon), "latitude": float(lat), "x_m": cx, "y_m": cy})
    comp = pd.DataFrame(rows).sort_values("total_capacity_mw", ascending=False).reset_index(drop=True)
    comp.insert(0, "complex_id", np.arange(len(comp)))
    mapping = dict(zip(comp["_raw"], comp["complex_id"]))
    src["complex_id"] = src["_raw"].map(mapping).astype(int)
    src = src.drop(columns="_raw")
    comp = comp.drop(columns="_raw")
    comp = gpd.GeoDataFrame(comp, geometry=gpd.points_from_xy(comp.longitude, comp.latitude), crs=4326)
    return src, comp


class SourceIndex:
    """Fast 'how far is the nearest source?' lookups in metres, via a KD-tree.

    Building the tree once and querying it for every sounding is what keeps the pipeline
    fast: a naive loop over sources for each of ~100k soundings would be far slower.
    """

    def __init__(self, sources: gpd.GeoDataFrame):
        self.sources = sources.reset_index(drop=True)
        self.tree = cKDTree(self.sources[["x_m", "y_m"]].values)

    def query(self, x_m, y_m):
        """Return (distance_km, source_row_index) of the nearest source for each point."""
        d, i = self.tree.query(np.c_[np.asarray(x_m), np.asarray(y_m)])
        return d / 1000.0, i

    def names(self, idx):
        return self.sources["name"].values[np.asarray(idx)]

    def complex_ids(self, idx):
        return self.sources["complex_id"].values[np.asarray(idx)]
