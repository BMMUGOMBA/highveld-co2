"""Print the structure of a NetCDF file: groups, variables, shapes, units, fill values.

    python -m highveld_co2.inspect_file data/raw/oco2/oco2_LtCO2_230715_B11014Ar_*.nc4

Run this FIRST on a real file. If a variable the reader needs has a different name or
sits in a different group, this is how you find out.
"""
import sys

from netCDF4 import Dataset


def describe(path, max_vars_per_group=60):
    with Dataset(path) as ds:
        print(f"File: {path}")
        print("Global attributes:", {k: ds.getncattr(k) for k in list(ds.ncattrs())[:8]}, "...")

        def walk(node, prefix=""):
            vars_ = list(node.variables.items())
            print(f"\n[{prefix or '/'}]  {len(vars_)} variables")
            for name, v in vars_[:max_vars_per_group]:
                units = getattr(v, "units", "")
                print(f"  {name:38s} {str(v.dtype):9s} shape={v.shape}  {units}")
            if len(vars_) > max_vars_per_group:
                print(f"  ... {len(vars_) - max_vars_per_group} more")
            for gname, g in node.groups.items():
                walk(g, f"{prefix}/{gname}")

        walk(ds)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    describe(sys.argv[1])
