"""Cheap Jev gates for the research loop (stdlib only).

Each gate is one TypeSafe call (~300-600 tokens) that decides whether an
expensive downstream analysis is worth running at all.

  python tools/jev_gate.py closed  "<proposal text>"   -> is it a closed direction?
  python tools/jev_gate.py route   "<question text>"   -> which expert pack
  python tools/jev_gate.py triage  <log_file>          -> what a pasted Kaggle log says
"""
import json, os, sys, urllib.request
from pathlib import Path

LAB = Path(__file__).resolve().parent
URL = "https://api.typesafe.ai/v1/systemone"


def ask(state, questions):
    req = urllib.request.Request(
        URL, method="POST",
        data=json.dumps({"state": state, "model": "jev-latest",
                         "questions": questions}).encode(),
        headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["answers"]


def closed(proposal):
    """One noul per closed direction, all in one parallel request."""
    dirs = json.loads((LAB / "closed_directions.json").read_text())
    qs = {d["id"]: {"type": "noul", "instructions":
          f"Is the `proposal` essentially the same experiment as this already-"
          f"closed direction, with no genuinely new mechanism? Closed: "
          f"{d['what']} (closed because: {d['why']})"} for d in dirs}
    ans = ask({"proposal": proposal}, qs)
    hits = sorted(((k, v["noul"]) for k, v in ans.items()), key=lambda x: -x[1])
    return {"max": hits[0], "hits": [h for h in hits if h[1] >= 0.5]}


PACKS = {
    "sim2real_gap": "Why real PIV differs from CFD; noise, resolution, measurement, augmentation",
    "architecture": "Operator architecture choice, freezing, capacity, resolution handling",
    "physics_loss": "Adding physics/spectral/TKE/divergence terms to the training loss",
    "uncertainty": "SPS bands, intervals, coverage, calibration",
    "generalization": "Unseen Reynolds number / angle of attack, conditioning",
    "efficiency": "Runtime, Time Score, model size, packaging speed",
    "submission": "Building, preflighting, or debugging a submission zip",
    "leaderboard": "Interpreting a surprising live or local score, metric interaction",
}


def route(question):
    a = ask({"question": question}, {"pack": {
        "type": "choice", "instructions":
        "Which specialist group should handle `question` in a RealPDE sim-to-real "
        "neural-operator competition lab?", "criteria": PACKS}})["pack"]
    return {"pack": a["choice"], "confidence": a["confidence"]}


def triage(log_text):
    tail = log_text[-6000:]
    a = ask({"kaggle_log_tail": tail}, {
        "status": {"type": "choice", "instructions":
                   "What state did this training run end in?", "criteria": {
                       "finished": "Training completed and final/FINAL scores are printed",
                       "running": "Log stops mid-training with no error",
                       "crashed": "A Python traceback or fatal error ended the run",
                       "aborted": "Run stopped itself early (abort-if-flat, time budget)"}},
        "beats_stock": {"type": "noul", "instructions":
                        "Does the log report the fine-tuned model scoring HIGHER than "
                        "the stock/frozen baseline on the summed FINAL (or SELECT, if "
                        "no FINAL) subscores?"}})
    return {"status": a["status"]["choice"], "status_conf": a["status"]["confidence"],
            "p_beats_stock": a["beats_stock"]["noul"]}


def decide(spec):
    """Generic decision: spec = {"question": str, "options": {name: description},
    "evidence": str|dict, optional "checks": {id: yes/no question}}. The choice is
    made on the evidence; checks are independent nouls asked in the same call."""
    qs = {"decision": {"type": "choice", "instructions": spec["question"],
                       "criteria": spec["options"]}}
    for k, q in spec.get("checks", {}).items():
        qs[k] = {"type": "noul", "instructions": q}
    ans = ask({"evidence": spec["evidence"]}, qs)
    d = ans.pop("decision")
    out = {"choice": d["choice"], "confidence": d["confidence"],
           "probabilities": d["probabilities"]}
    out.update({k: v["noul"] for k, v in ans.items()})
    with open(LAB / "decisions.jsonl", "a") as f:
        f.write(json.dumps({"question": spec["question"], **out}) + "\n")
    return out


if __name__ == "__main__":
    cmd, arg = sys.argv[1], sys.argv[2]
    if cmd == "triage":
        arg = Path(arg).read_text(encoding="utf-8", errors="replace")
    if cmd == "decide":
        arg = json.loads(Path(arg).read_text(encoding="utf-8"))
    print(json.dumps({"closed": closed, "route": route, "triage": triage,
                      "decide": decide}[cmd](arg)))
