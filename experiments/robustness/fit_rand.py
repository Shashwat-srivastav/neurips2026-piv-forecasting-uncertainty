"""A2: fit v17 recipe tables on the random AoA-stratified train-pool split (k fitted the same way as for SELECT)."""
from lib2 import *; import pickle, time
d = np.load("cache_rand_s20.npz"); X, Y, P = d["X"], d["Y"], d["P"]
S = prep(X, Y, P); print("rand n", len(X), "k_A (rel-matched) %.3f k_T (v8 T=0.4914) %.3f" % S["k"], flush=True)
print("rand raw rel %.4f" % float(sc.rel_l2_per_sample(P, Y, 2).mean()))
pickle.dump(fit_recipe(S), open("tabs_rand_pw.pkl", "wb")); print("saved")
