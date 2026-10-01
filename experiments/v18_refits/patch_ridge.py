"""Patch a copy of the v17 submission.py with the v18 ridge-rank band width. usage: patch_ridge.py path/to/submission.py"""
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
assert "_ridge_width" not in s
s = s.replace("WIDTH_TAB as _WIDTH_TAB_L  # noqa: E402",
 "WIDTH_TAB as _WIDTH_TAB_L  # noqa: E402\nfrom _blend_tables import RIDGE_W as _RIDGE_W_L, RWIDTH_TAB as _RWIDTH_TAB_L  # noqa: E402", 1)
ridge = open(__file__.replace("patch_ridge.py", "ridge_torch.txt"), encoding="utf-8").read()
s = s.replace("def _apply_tab(", ridge + "def _apply_tab(", 1)
old = "            yb[..., :2], centre = _blend(yb[..., :2], xc, sl)\n\n            if _WIDTH_TAB_L is None:"
assert old in s
s = s.replace(old, "            p_raw = yb[..., :2].clone() if _RIDGE_W_L is not None else None\n            yb[..., :2], centre = _blend(yb[..., :2], xc, sl)\n\n            if _RIDGE_W_L is not None:\n                # v18: ridge-score within-call rank -> (lead group, channel, bin) half-width\n                hb = torch.zeros_like(yb)\n                hb[..., :2] = _ridge_width(xc, p_raw)\n            elif _WIDTH_TAB_L is None:", 1)
open(p, "w", encoding="utf-8").write(s)
print("patched", p)
