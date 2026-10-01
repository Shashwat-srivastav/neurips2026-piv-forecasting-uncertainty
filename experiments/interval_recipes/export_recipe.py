"""Export fitted tables (SELECT fit, proxy k_T) for the ridge-lookup and p1-lookup recipes."""
import json
from rlib import *
out = {"k_fit_select_Tmatched": KT["select"], "LOGF(log(x+1e-4))": LOGF, "LINF": LINF, "extra": ["t^2/400", "yy^2", "xx^2", "1"]}
for c in (0, 1):
    ds = dat[("select", c)]; a = ds["ae"] * KT["select"]
    e1, h1 = lookup(ds["p1"], a, 40); e2, h2 = lookup(ds["ridge"], a, 40)
    out[f"ch{c}"] = dict(ridge_w=W[c].tolist(), ridge_edges=e2.tolist(), ridge_h=h2.tolist(), p1_edges=e1.tolist(), p1_h=h1.tolist())
    print(c, "p1 h range", h1.min().round(4), h1.max().round(4), "ridge h range", h2.min().round(4), h2.max().round(4))
json.dump(out, open("recipe_tables.json", "w"), indent=0)
