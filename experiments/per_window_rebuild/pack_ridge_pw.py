"""Pack a (2, 6*16, F, 20) ridge W into the build's ridge_w.npz layout (16 blocks, F, 2*6*20)."""
import sys, numpy as np
def pack(W): return W.reshape(2, 6, 16, W.shape[2], 20).transpose(2, 3, 0, 1, 4).reshape(16, W.shape[2], 240).astype(np.float32)
ref = np.load("../build_v19/ridge_w.npz"); old = np.load("../v19/W_sxb3.npz")
print("layout check vs shipped ridge_w.npz: max diff", np.abs(pack(old["W"]) - ref["W"]).max())
new = np.load("W_pw.npz")
np.savez(sys.argv[1], W=pack(new["W"]), m=np.array(10), k=np.array(5))
print("wrote", sys.argv[1])
