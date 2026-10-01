"""A2 sensitivity: random split, but SELECT's k values (k_A 1.534, k_T 1.394)."""
from lib2 import *; import pickle
d = np.load("cache_rand_s20.npz"); S = prep(d["X"], d["Y"], d["P"], KA["select"], KT["select"])
pickle.dump(fit_recipe(S), open("tabs_randk_pw.pkl", "wb")); print("saved")
