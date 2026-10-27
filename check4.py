from pathlib import Path
from highveld_co2.read_oco2 import read_many
from highveld_co2.qc import apply_qc
df = read_many(sorted(Path("data/raw/synthetic").glob("*.nc*")))
clean, rep = apply_qc(df)
print(len(df), "soundings read;", len(clean), "after QC")
print(rep)