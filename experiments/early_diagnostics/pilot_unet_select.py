import json, os, sys, time
import h5py, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'src', d) for d in ('training', 'evaluation', 'splits')]
KIT = os.path.join(REPO, "realpde_t1_starting_kit_v9", "realpde_t1_starting_kit_v9")
ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
sys.path.insert(0, KIT); sys.path.insert(0, HERE)
import importlib.util as _iu
spec = _iu.spec_from_file_location("sc9", os.path.join(KIT, "scoring.py"))
sc = _iu.module_from_spec(spec); spec.loader.exec_module(sc)
from rpdebench_unet import Unet3d

torch.set_num_threads(8)
split = json.load(open("data/splits/holdout_v12_3.json"))
names = split["select"]  # 4 trajectories, ~168 windows -- fast pilot

def load_windows(names, stride=20):
    X, Y = [], []
    for nm in names:
        with h5py.File(os.path.join(ROOT, nm + ".h5")) as f:
            u = np.asarray(f["u"][:, ::2, ::2], np.float32)
            v = np.asarray(f["v"][:, ::2, ::2], np.float32)
        fr = np.stack([u, v, np.zeros_like(u)], -1)
        for i in range(0, len(fr) - 40 + 1, stride):
            X.append(fr[i:i + 20]); Y.append(fr[i + 20:i + 40])
    return np.stack(X), np.stack(Y)

X, Y = load_windows(names)
print(f"windows: {len(X)}", flush=True)

model = Unet3d(dim=64, out_channels=3, dim_mults=(1, 2, 4), channels=3, in_time=20, out_time=20)
model.load_checkpoint("e:/NeurIPS/rpdebench_ckpts/foil/unet/finetune.pth", "cpu")
model.eval()

CONFIGS = {
    "kit-normalizer": (
        np.array([0.154960856, -0.000513992854, 0.0], np.float32),
        np.array([0.0968056545, 0.015960684, 1.0], np.float32),
        np.array([0.154962569, -0.000517793698, 0.0], np.float32),
        np.array([0.0968104079, 0.0159636438, 1.0], np.float32),
    ),
    "own-real-normalizer": (
        np.load("data/derived/own_real_normalizer_mean.npy").astype(np.float32),
        np.array([0.10605387, 0.0135536, 1.0], np.float32),
        np.load("data/derived/own_real_normalizer_mean.npy").astype(np.float32),
        np.array([0.10605387, 0.0135536, 1.0], np.float32),
    ),
}

BATCH = 5
for cfg_name, (MEAN_IN, STD_IN, MEAN_TGT, STD_TGT) in CONFIGS.items():
    mi, si, mt, st = (torch.from_numpy(z) for z in (MEAN_IN, STD_IN, MEAN_TGT, STD_TGT))
    t0 = time.perf_counter()
    P = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            chunk = X[i:i + BATCH]
            got = chunk.shape[0]
            if got < BATCH:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], BATCH - got, axis=0)], axis=0)
            xb = torch.from_numpy(np.ascontiguousarray(chunk))
            yb = model((xb - mi) / si) * st + mt
            yb[..., 2] = 0.0
            P.append(yb.float().numpy()[:got])
            print(f"  [{cfg_name}] {i+got}/{len(X)}  elapsed {time.perf_counter()-t0:.0f}s", flush=True)
    P = np.concatenate(P)
    rel = sc.score_error(float(sc.rel_l2_per_sample(P, Y, 2).mean()))
    tke = sc.score_error(float(sc.tke_rel_l2_per_sample(P, Y, 2).mean()))
    mvpe = sc.score_error(float(sc.mvpe_rel_l2_per_sample(P, Y).mean()))
    print(f"=== {cfg_name}: rel {rel:.3f}  tke {tke:.3f}  mvpe {mvpe:.3f}  "
          f"({time.perf_counter()-t0:.0f}s total, {(time.perf_counter()-t0)/len(X)*1000:.0f} ms/sample) ===\n",
          flush=True)
