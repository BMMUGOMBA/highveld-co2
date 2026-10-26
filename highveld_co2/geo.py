"""Small geography helpers: lon/lat <-> metres, great-circle distance."""
import numpy as np
from pyproj import Transformer

from .config import METRIC_CRS

_TO_M = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
_TO_LL = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)

EARTH_RADIUS_KM = 6371.0088


def to_metric(lon, lat):
    """lon/lat (degrees) -> x/y in metres in METRIC_CRS. Accepts scalars or arrays."""
    x, y = _TO_M.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.asarray(x), np.asarray(y)


def to_lonlat(x, y):
    """x/y metres in METRIC_CRS -> lon/lat degrees."""
    lon, lat = _TO_LL.transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return np.asarray(lon), np.asarray(lat)


def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km. Used to cross-check the metric-CRS distances."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
