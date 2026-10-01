"""Fit v17 recipe tables on SELECT (per-window bins). -> tabs_select_pw.pkl"""
from lib2 import *; import pickle, time
t0 = time.time(); X, Y, P, _ = load("select")
S = prep(X, Y, P, KA["select"], KT["select"]); tabs = fit_recipe(S)
pickle.dump(tabs, open("tabs_select_pw.pkl", "wb")); print("fit select", round(time.time() - t0), "s")
