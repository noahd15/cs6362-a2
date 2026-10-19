#!/usr/bin/env python
"""Plot Figure 1 (a, b, c) from saved run JSONs only. No training.

    python plot_fig1.py --results-dir ./results_fig1 --threshold 0.99
    python plot_fig1.py --threshold 0.95 --test-error-at final

--threshold picks which recorded train-accuracy crossing (0.95 / 0.98 / 0.99) defines
"converged" for panel (b) time-to-overfit and, by default, the panel (c) test error.
"""
import argparse, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results-dir", default="./results_fig1")
ap.add_argument("--threshold", type=float, default=0.99, choices=[0.95, 0.98, 0.99])
ap.add_argument("--test-error-at", default="crossing", choices=["crossing", "final"])
ap.add_argument("--time", default="steps", choices=["steps", "seconds"])
ap.add_argument("--batch-size", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
a = ap.parse_args()

runs = []
for f in glob.glob(os.path.join(a.results_dir, "*.json")):
    with open(f) as fh:
        r = json.load(fh)
    if r.get("status") == "done":
        runs.append(r)
print(f"{len(runs)} finished runs in {a.results_dir}")


def find(model, mode, p, **extra):
    """The unique finished run matching these settings; refuses to guess between duplicates."""
    want = dict(model=model, mode=mode, p=round(p, 6), batch_size=a.batch_size, seed=a.seed, **extra)
    hits = [r for r in runs if all(r["config"].get(k) == v for k, v in want.items())]
    if len(hits) > 1:
        sys.exit(f"ambiguous: {len(hits)} runs match {want}; pass a cleaner --results-dir")
    return hits[0] if hits else None


key = f"{a.threshold:g}"
def crossing_time(r):
    c = r["crossings"].get(key)
    if c is None:
        return None
    return c["step"] if a.time == "steps" else c["train_seconds"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# (a) learning curves, Inception
A = [("true labels", "labels", 0.0, "b", "s"), ("random labels", "labels", 1.0, "r", "o"),
     ("shuffled pixels", "shuffled_pixels", 0.0, "g", "*"), ("random pixels", "random_pixels", 0.0, "m", None),
     ("gaussian", "gaussian", 0.0, "k", "D")]
for label, mode, p, c, mk in A:
    cands = [r for r in runs if r["config"]["model"] == "inception" and r["config"]["mode"] == mode
             and r["config"]["p"] == p and r["config"]["batch_size"] == a.batch_size and r["config"]["seed"] == a.seed]
    if not cands:
        print(f"(a) missing: {label}"); continue
    r = max(cands, key=lambda r: r["final"]["step"])          # prefer the longest curve (e.g. min_steps run)
    s = np.array(r["step_log"]["step"]) / 1000
    axes[0].plot(s, r["step_log"]["loss"], color=c, marker=mk, markevery=max(1, len(s) // 25), markersize=4,
                 lw=1.5, label=label, markerfacecolor="none" if mk in ("o", "s") else c)
axes[0].set(xlabel="thousand steps", ylabel="average_loss", xlim=(0, 25), ylim=(0, 2.5))
axes[0].legend(fontsize=8)

# (b) relative time to overfit, (c) test error vs label corruption
S = {"inception": ("Inception", "k", "s"), "alexnet": ("AlexNet", "#22cc22", "o"), "mlp_1x512": ("MLP 1x512", "b", "*")}
P = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
for model, (name, c, mk) in S.items():
    base = find(model, "labels", 0.0, min_steps=0)
    t0 = crossing_time(base) if base else None
    bx, by, cx, cy = [], [], [], []
    for p in P:
        r = find(model, "labels", p, min_steps=0)
        if r is None:
            print(f"{name} p={p}: no run"); continue
        if r["hit_epoch_cap"]:
            print(f"{name} p={p}: FLAG hit epoch cap without converging at its stop threshold")
        t = crossing_time(r)
        if t is None or t0 is None:
            print(f"{name} p={p}: never reached train acc {a.threshold}; omitted from (b)" +
                  (" and (c)" if a.test_error_at == "crossing" else ""))
        else:
            bx.append(p); by.append(t / t0)
        if a.test_error_at == "final":
            cx.append(p); cy.append(r["final"]["test_error"])
        elif t is not None:
            cx.append(p); cy.append(r["crossings"][key]["test_error"])
    kw = dict(color=c, marker=mk, label=name, lw=2, markerfacecolor="none" if mk == "o" else c)
    if bx: axes[1].plot(bx, by, **kw)
    if cx: axes[2].plot(cx, cy, **kw)
axes[1].set(xlabel="label corruption", ylabel="time to overfit", xlim=(0, 1)); axes[1].legend(loc="upper left", fontsize=8)
axes[2].axhline(0.9, color="r", ls="--", lw=1.5)
axes[2].set(xlabel="label corruption", ylabel="test_error", xlim=(0, 1), ylim=(0.1, 1.0)); axes[2].legend(loc="lower right", fontsize=8)
for ax, t in zip(axes, ["(a) learning curves", "(b) convergence slowdown", "(c) generalization error growth"]):
    ax.set_title(t, y=-0.28); ax.grid(alpha=0.3)
plt.tight_layout()
out = a.out or os.path.join(a.results_dir, f"figure1_thr{key}_{a.test_error_at}")
plt.savefig(out + ".pdf", bbox_inches="tight"); plt.savefig(out + ".png", dpi=150, bbox_inches="tight")
print("saved", out + ".pdf/.png")

# Batch-size runs (not in Fig. 1; reported for sensitivity)
for bs in (64, 128, 256):
    hits = [r for r in runs if r["config"]["model"] == "inception" and r["config"]["mode"] == "labels"
            and r["config"]["p"] == 1.0 and r["config"]["batch_size"] == bs and r["config"]["seed"] == a.seed]
    for r in hits:
        cr = r["crossings"].get(key)
        print(f"inception random labels bs={bs}: crossing {key} at step {cr and cr['step']} "
              f"(epoch {cr and cr['epoch']}), cap_hit={r['hit_epoch_cap']}")
