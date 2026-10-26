from highveld_co2.sources import load_sources, group_complexes
s, c = group_complexes(load_sources())
print(len(s), "sources,", len(c), "complexes")
print(c[["complex_id", "name", "n_sources"]].to_string(index=False))