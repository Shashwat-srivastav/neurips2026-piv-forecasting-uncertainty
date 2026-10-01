#!/usr/bin/env python3
"""SWA (stochastic weight averaging) over the per-epoch snapshots produced by
finetune_fno_kaggle.py's --snapshot-last-k. Genuinely new to this campaign
(6 Sep) -- the campaign record has zero prior mentions of
SWA/EMA anywhere. NOT the same thing as the already-closed WiSE-FT test
(wiseft_test.py, 22 Aug): that interpolated between two DISTANT checkpoints
(sim_fno <-> sim_real_fno, ||delta||/||theta|| = 1.094, no shared basin ->
monotonically catastrophic). This averages consecutive epochs of the SAME
fine-tune run, which by construction sit close together in weight space --
the textbook SWA setting, not the one already shown to fail.

Only meaningful for a run that stays in one basin throughout -- i.e. a
--freeze-bn-fc run, where every snapshot has BIT-IDENTICAL spectral weights
(frozen) and only fc0/fc1/fc2/bns.* actually differ epoch to epoch. Averaging
snapshots from an UNFROZEN run (still moving through weight space, possibly
crossing loss barriers) is not standard SWA and is not what this was built
for -- don't feed it those.

USAGE:
    python swa_average.py out/snap_ep010.pth out/snap_ep011.pth ... \
        --out out/fno_swa.pth

CAVEAT (read before trusting the result): BatchNorm layers carry two kinds of
state -- trainable affine params (bns.*.weight/bias, meaningfully different
per epoch, safe to average) and non-trainable running_mean/running_var
BUFFERS. Those buffers keep updating during forward passes in .train() mode
EVEN under --freeze-bn-fc (freezing only stops gradients, not the running
stat update), so they differ per snapshot too. Naively averaging running
stats across epochs is a known SWA rough edge -- the textbook fix is one
extra forward pass over the training data in .train() mode, AFTER loading
the averaged weights, to recompute clean running statistics, before ever
touching SELECT/FINAL. This script averages everything (parameters AND
buffers) by default because that is still a reasonable starting point and
costs nothing extra to compute, but prints a ready-to-paste Kaggle snippet
to do the proper BN recalibration pass afterward -- run that before scoring
if you want the textbook-correct version, and compare both.
"""
import argparse
import sys

import torch


def load_sd(path):
    ck = torch.load(path, map_location="cpu")
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    meta = {k: ck[k] for k in ("epoch", "select_sum") if k in ck}
    return sd, meta


def average_state_dicts(paths):
    sds, metas = [], []
    for p in paths:
        sd, meta = load_sd(p)
        sds.append(sd)
        metas.append(meta)
        tag = f"epoch {meta.get('epoch', '?')}  SELECT sum {meta.get('select_sum', float('nan')):.3f}" \
            if meta else "(no epoch/select_sum metadata saved)"
        print(f"  {p}  {tag}")

    keys = sds[0].keys()
    for i, sd in enumerate(sds[1:], 1):
        if sd.keys() != keys:
            raise ValueError(f"{paths[i]} has different keys than {paths[0]} -- "
                              "not snapshots from the same run/architecture.")

    avg = {}
    n_float, n_other = 0, 0
    for k in keys:
        tensors = [sd[k] for sd in sds]
        if tensors[0].is_floating_point():
            avg[k] = torch.stack(tensors, dim=0).mean(dim=0)
            n_float += 1
        else:
            # int buffers (e.g. num_batches_tracked): averaging is meaningless,
            # take the last snapshot's value instead.
            avg[k] = tensors[-1].clone()
            n_other += 1
    print(f"averaged {n_float} floating tensors, carried {n_other} "
          "integer/non-float buffers from the last snapshot unchanged")
    return avg


def print_bn_recalibration_snippet(out_path):
    print(f"""
--- optional but recommended: recompute BN running stats before scoring ---
paste in the SAME Kaggle cell/session that has finetune_fno_kaggle's build()
and Windows/DataLoader helpers already defined (import it or paste it first):

    import torch, finetune_fno_kaggle as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = F.build(kit="<same --kit path used for training>", ckpt="{out_path}")
    mi, si = (torch.from_numpy(z).to(dev) for z in (F.MEAN_IN, F.STD_IN))
    mt, st = (torch.from_numpy(z).to(dev) for z in (F.MEAN_TGT, F.STD_TGT))
    tr = F.Windows("<same --data path>", <train_names used for the run>, stride=2)
    dl = torch.utils.data.DataLoader(tr, batch_size=32, shuffle=True)
    model.train()
    with torch.no_grad():          # updates running_mean/var, no gradients needed
        for i, (xb, yb) in enumerate(dl):
            xb = xb.to(dev)
            F.forward_physical(model, xb, mi, si, mt, st)
            if i >= 200:            # a few hundred batches is enough to settle
                break               # running_mean/var; no need for a full epoch
    torch.save({{"model_state_dict": model.state_dict()}}, "{out_path}.bn_recalibrated.pth")

Score BOTH {out_path} and the _bn_recalibrated version against the fair
frozen-sim_fno holdout baseline (see score_finetuned_ckpt.py) before deciding
which one, if either, is worth a submission slot.
----------------------------------------------------------------------------""")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshots", nargs="+", help="snap_epNNN.pth files to average")
    ap.add_argument("--out", required=True, help="output path for the averaged checkpoint")
    a = ap.parse_args()

    if len(a.snapshots) < 2:
        sys.exit("need at least 2 snapshots to average -- that's not SWA, "
                  "it's just a copy.")

    print(f"averaging {len(a.snapshots)} snapshots:")
    avg = average_state_dicts(a.snapshots)
    torch.save({"model_state_dict": avg}, a.out)
    print(f"\nsaved averaged checkpoint -> {a.out}")
    print_bn_recalibration_snippet(a.out)
    print("\nNEXT STEP: score this against the FAIR frozen-sim_fno holdout "
          "baseline (not sim_real_fno's contaminated one) before spending any "
          "submission slot on it.")


if __name__ == "__main__":
    main()
