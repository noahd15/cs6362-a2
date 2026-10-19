#!/usr/bin/env python
"""Train SB/LB minimisers and evaluate the linear interpolation path for Figure 3
(F1, C1, C2) of Keskar et al., "On Large-Batch Training for Deep Learning:
Generalization Gap and Sharp Minima", ICLR 2017.

Written from the paper alone. Every choice the paper does not specify is marked
ASSUMPTION. Outputs, all under --out-dir:

  checkpoints/{net}_trial{t}.pt          init + SB/LB minimisers + per-epoch histories
  checkpoints/{net}_trial{t}_{sb,lb}_progress.pt   mid-run state (deleted when the run finishes)
  {net}_results.json                     everything the notebook needs to plot
  logs/{net}.log                         full log (appended across restarts)

Re-running the same command resumes: finished trials are loaded, an interrupted run
continues from its last progress checkpoint, and a finished path is skipped.
"""
import argparse, json, logging, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

DATASET_OF = {"F1": "MNIST", "C1": "CIFAR-10", "C2": "CIFAR-10"}   # Table 1, p.4
log = logging.getLogger("keskar")


# ----------------------------------------------------------------------------- config
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", choices=list(DATASET_OF))
    p.add_argument("--prepare-data", action="store_true", help="only download datasets, then exit")
    p.add_argument("--stage", choices=["all", "train", "path"], default="all",
                   help="train: minimisers only; path: interpolation only (needs trained trials)")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out-dir", default="./results")
    # Sec. 2.2 (p.4): SB = 256, LB = 10% of the training set, ADAM for both.
    p.add_argument("--sb-batch", type=int, default=256)
    p.add_argument("--lb-fraction", type=float, default=0.10)
    # ASSUMPTION: ADAM hyperparameters unstated; Kingma & Ba defaults.
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999])
    p.add_argument("--adam-eps", type=float, default=1e-8)
    # Sec. 2.2 (p.4): train "until the loss function ceased to improve". ASSUMPTION: stop
    # when full-training-set loss fails to improve by a relative --min-rel-improvement for
    # --patience epochs; --max-epochs is a safety cap.
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-rel-improvement", type=float, default=1e-3)
    # Sec. 2.2 (p.4): 5 trials; Fig. 3 plots one random (SB, LB) pair (Sec. 2.2.1).
    p.add_argument("--n-trials", type=int, default=1, help="5 = full paper setting")
    p.add_argument("--seed", type=int, default=0)
    # ASSUMPTION: SB and LB of a trial share their initial weights.
    p.add_argument("--independent-init", action="store_true",
                   help="give SB and LB different inits and pick their trials independently")
    # Sec. 2.2.1 / Fig. 3 caption: alpha in [-1, 2]. ASSUMPTION: 25 points.
    p.add_argument("--alpha-min", type=float, default=-1.0)
    p.add_argument("--alpha-max", type=float, default=2.0)
    p.add_argument("--n-alpha", type=int, default=25)
    # ASSUMPTION: BN statistics at interpolated weights (paper silent).
    p.add_argument("--bn-mode", choices=["recalibrate", "interpolate"], default="recalibrate")
    p.add_argument("--bn-recal-samples", type=int, default=0, help="0 = whole training set")
    p.add_argument("--bn-recal-batch", type=int, default=1000)
    p.add_argument("--eval-batch", type=int, default=2000)
    p.add_argument("--micro-batch", type=int, default=0,
                   help="0 = whole mini-batch per forward pass (faithful); >0 accumulates gradients, "
                        "but BN then normalises over micro-batches")
    p.add_argument("--amp", action="store_true", help="mixed precision (fp32 by default)")
    p.add_argument("--save-every", type=int, default=5, help="epochs between progress checkpoints")
    p.add_argument("--force-path", action="store_true", help="recompute the path even if results exist")
    return p.parse_args(argv)

TRAIN_KEYS = ["sb_batch", "lb_fraction", "lr", "betas", "adam_eps", "max_epochs", "patience",
              "min_rel_improvement", "independent_init", "micro_batch", "amp"]
PATH_KEYS = ["alpha_min", "alpha_max", "n_alpha", "bn_mode", "bn_recal_samples", "bn_recal_batch"]

def sub(args, keys):
    return {k: getattr(args, k) for k in keys}


# ----------------------------------------------------------------------------- data
# Appendix A (p.12): data sets used "without any pre-processing". ASSUMPTION: pixels
# scaled to [0, 1], no normalisation, no augmentation. Whole sets live on the GPU.
def load_dataset(name, root):
    if name == "MNIST":
        tr = torchvision.datasets.MNIST(root, train=True, download=True)
        te = torchvision.datasets.MNIST(root, train=False, download=True)
        prep = lambda d: d.data.float().div(255.0).unsqueeze(1)
        return (prep(tr), tr.targets.long()), (prep(te), te.targets.long())
    if name == "CIFAR-10":
        tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
        te = torchvision.datasets.CIFAR10(root, train=False, download=True)
        prep = lambda d: torch.from_numpy(d.data).permute(0, 3, 1, 2).float().div(255.0)
        return (prep(tr), torch.tensor(tr.targets).long()), (prep(te), torch.tensor(te.targets).long())
    raise ValueError(name)

def get_data(dataset, root, device):
    (xtr, ytr), (xte, yte) = load_dataset(dataset, root)
    return {"train": (xtr.to(device), ytr.to(device)), "test": (xte.to(device), yte.to(device))}


# ----------------------------------------------------------------------------- networks (Appendix B)
def make_F1():
    # B.1: 784 input, 5 batch-normalised layers of 512 ReLU units, 10-way softmax.
    layers, d = [nn.Flatten()], 784
    for _ in range(5):
        layers += [nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU()]
        d = 512
    layers.append(nn.Linear(512, 10))          # softmax is inside cross_entropy
    return nn.Sequential(*layers)

def _dense_head(flat, widths, n_out):
    # B.3/B.4: BN on all layers, ReLU, dropout with 0.5 retention on the dense layers.
    # ASSUMPTION: Linear -> BN -> ReLU -> Dropout; no BN on the output layer.
    layers, d = [nn.Flatten()], flat
    for w in widths:
        layers += [nn.Linear(d, w), nn.BatchNorm1d(w), nn.ReLU(), nn.Dropout(0.5)]
        d = w
    layers.append(nn.Linear(d, n_out))
    return layers

def _flat_dim(features):
    with torch.no_grad():
        return features.eval()(torch.zeros(1, 3, 32, 32)).numel()

def make_C1(n_out=10):
    # B.3: "2 sets of [64,5,5,2]-MaxPool(3)", dense (384, 192), output.
    # ASSUMPTION: padding 2 on the stride-2 convs; MaxPool(3) = kernel 3, stride 2, padding 1.
    feats, c_in = [], 3
    for _ in range(2):
        feats += [nn.Conv2d(c_in, 64, 5, stride=1, padding=2), nn.BatchNorm2d(64), nn.ReLU(),
                  nn.MaxPool2d(3, stride=2, padding=1)]
        c_in = 64
    feats = nn.Sequential(*feats)
    return nn.Sequential(*feats, *_dense_head(_flat_dim(feats), (384, 192), n_out))

def make_C2(n_out=10):
    # B.4 ("The C3 network uses" is read as a typo for C2): 2x[64,3,3,1], 2x[128,3,3,1],
    # 3x[256,3,3,1], 3x[512,3,3,1], 3x[512,3,3,1], MaxPool(2) after each stack, 512 dense, 10 out.
    # ASSUMPTION: padding 1 on the 3x3 convs.
    feats, c_in = [], 3
    for n, c in [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]:
        for _ in range(n):
            feats += [nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU()]
            c_in = c
        feats.append(nn.MaxPool2d(2))
    feats = nn.Sequential(*feats)
    return nn.Sequential(*feats, *_dense_head(_flat_dim(feats), (512,), n_out))

BUILDERS = {"F1": make_F1, "C1": make_C1, "C2": make_C2}

def build_model(name, seed):
    # Sec. 2.2: "uniformly distributed random" starts. ASSUMPTION: PyTorch default init
    # (Kaiming-uniform weights, uniform biases).
    torch.manual_seed(seed)
    return BUILDERS[name]()


# ----------------------------------------------------------------------------- helpers
def atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def atomic_write_json(obj, path):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)

def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

@torch.no_grad()
def evaluate(model, X, Y, args):
    """Mean cross-entropy and accuracy (%) over a whole set, in eval mode."""
    model.eval()
    loss_sum, correct = 0.0, 0
    for xb, yb in zip(X.split(args.eval_batch), Y.split(args.eval_batch)):
        with torch.autocast(X.device.type, enabled=args.amp):
            logits = model(xb)
        logits = logits.float()
        loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
    return loss_sum / len(Y), 100.0 * correct / len(Y)


# ----------------------------------------------------------------------------- training
def train_run(name, batch_size, init_state, seed, data, args, device, progress_path, tag):
    model = build_model(name, 0).to(device)
    model.load_state_dict(init_state)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=tuple(args.betas), eps=args.adam_eps)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)
    Xtr, Ytr = data["train"]; Xte, Yte = data["test"]
    N = len(Ytr)
    steps = N // batch_size          # Sec. 1.1: fixed batch size -> remainder dropped each epoch
    micro = args.micro_batch or batch_size
    gen = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)          # dropout RNG
    history, best, bad, start, stopped = [], math.inf, 0, 1, False

    if progress_path.exists():
        p = torch.load(progress_path, map_location="cpu", weights_only=False)
        if p["train_cfg"] != sub(args, TRAIN_KEYS) or p["seed"] != seed:
            raise RuntimeError(f"{progress_path} was written with a different config; "
                               "re-run with the original flags or move it away to restart this run.")
        model.load_state_dict(p["model"]); opt.load_state_dict(p["opt"]); scaler.load_state_dict(p["scaler"])
        gen.set_state(p["gen"]); torch.set_rng_state(p["cpu_rng"])
        if device.type == "cuda" and p["cuda_rng"] is not None:
            torch.cuda.set_rng_state(p["cuda_rng"])
        history, best, bad, start = p["history"], p["best"], p["bad"], p["epoch"] + 1
        stopped = p["stopped"]
        log.info(f"[{tag}] resuming after epoch {p['epoch']}")

    def save_progress(epoch):
        atomic_save({"model": cpu_state(model), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                     "gen": gen.get_state(), "cpu_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                     "history": history, "best": best, "bad": bad, "epoch": epoch,
                     "stopped": stopped,
                     "train_cfg": sub(args, TRAIN_KEYS), "seed": seed}, progress_path)

    for epoch in range(start, args.max_epochs + 1):
        if stopped:
            break
        t0 = time.time()
        model.train()
        perm = torch.randperm(N, device=device, generator=gen)
        for s in range(steps):
            idx = perm[s * batch_size:(s + 1) * batch_size]
            opt.zero_grad(set_to_none=True)
            for chunk in idx.split(micro):
                with torch.autocast(device.type, enabled=args.amp):
                    loss = F.cross_entropy(model(Xtr[chunk]), Ytr[chunk]) * (len(chunk) / len(idx))
                scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_s = time.time() - t0
        tr_loss, tr_acc = evaluate(model, Xtr, Ytr, args)
        te_loss, te_acc = evaluate(model, Xte, Yte, args)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "test_loss": te_loss, "test_acc": te_acc, "epoch_seconds": train_s})
        log.info(f"[{tag}] ep {epoch:3d}  train {tr_loss:.4f} / {tr_acc:6.2f}%   "
                 f"test {te_loss:.4f} / {te_acc:6.2f}%   {train_s:.1f}s")
        if not math.isfinite(tr_loss):
            log.warning(f"[{tag}] non-finite training loss; stopping"); stopped = True
        elif tr_loss < best * (1.0 - args.min_rel_improvement):
            best, bad = tr_loss, 0
        else:
            bad += 1
            stopped = bad >= args.patience
        if stopped or epoch % args.save_every == 0:
            save_progress(epoch)
        if stopped:
            break
    h = history[-1]
    reason = "plateau" if bad >= args.patience else ("non-finite" if not math.isfinite(h["train_loss"]) else "max_epochs")
    log.info(f"[{tag}] finished at epoch {h['epoch']} ({reason}): "
             f"train {h['train_acc']:.2f}%  test {h['test_acc']:.2f}%")
    # The minimiser is the final iterate, not the best-test-accuracy epoch.
    return cpu_state(model), history, reason


def get_trial(name, t, data, args, device, ckdir):
    seed = args.seed + 1000 * t
    ck = ckdir / f"{name}_trial{t}.pt"
    if ck.exists():
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        if blob["train_cfg"] != sub(args, TRAIN_KEYS) or blob["seed"] != seed:
            raise RuntimeError(f"{ck} was trained with a different config; move it away to retrain.")
        log.info(f"{name} trial {t}: loaded {ck.name}")
        return blob
    N = len(data["train"][1])
    lb_batch = int(round(args.lb_fraction * N))
    init_sb = cpu_state(build_model(name, seed))
    init_lb = cpu_state(build_model(name, seed + 1)) if args.independent_init else init_sb
    runs = {}
    for kind, bs, init, rseed in [("sb", args.sb_batch, init_sb, seed + 2), ("lb", lb_batch, init_lb, seed + 3)]:
        prog = ckdir / f"{name}_trial{t}_{kind}_progress.pt"
        runs[kind] = train_run(name, bs, init, rseed, data, args, device, prog, f"{name} t{t} {kind.upper()} {bs}")
    blob = {"seed": seed, "train_cfg": sub(args, TRAIN_KEYS), "sb_batch": args.sb_batch, "lb_batch": lb_batch,
            "init_sb": init_sb, "init_lb": init_lb,
            "sb_state": runs["sb"][0], "sb_history": runs["sb"][1], "sb_stop": runs["sb"][2],
            "lb_state": runs["lb"][0], "lb_history": runs["lb"][1], "lb_stop": runs["lb"][2]}
    atomic_save(blob, ck)
    for kind in ("sb", "lb"):
        (ckdir / f"{name}_trial{t}_{kind}_progress.pt").unlink(missing_ok=True)
    log.info(f"{name} trial {t}: saved {ck.name}")
    return blob


# ----------------------------------------------------------------------------- interpolation path
def lerp_state(sd_s, sd_l, a):
    """(1 - a) x_s + a x_l for every floating-point tensor (parameters and BN buffers)."""
    return {k: ((1.0 - a) * v + a * sd_l[k]) if v.is_floating_point() else v.clone() for k, v in sd_s.items()}

@torch.no_grad()
def recalibrate_bn(model, X, args):
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    moms = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats(); m.momentum = None      # cumulative average
    model.eval()                                        # dropout off
    for m in bns:
        m.train()                                       # BN collects statistics
    for xb in X.split(args.bn_recal_batch):
        with torch.autocast(X.device.type, enabled=args.amp):
            model(xb)
    for m, mo in zip(bns, moms):
        m.momentum = mo
    model.eval()

def evaluate_path(name, sd_s, sd_l, data, args, device):
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.n_alpha)
    Xtr, Ytr = data["train"]; Xte, Yte = data["test"]
    Xrc = Xtr
    if args.bn_recal_samples:
        g = torch.Generator(device=device).manual_seed(args.seed)
        Xrc = Xtr[torch.randperm(len(Ytr), device=device, generator=g)[:args.bn_recal_samples]]
    model = build_model(name, 0).to(device)
    res = {k: [] for k in ["alpha", "train_loss", "train_acc", "test_loss", "test_acc", "n_running_var_clamped"]}
    for a in alphas:
        sd = lerp_state(sd_s, sd_l, float(a))
        n_clamped = 0
        if args.bn_mode == "interpolate":
            for k in sd:
                if k.endswith("running_var"):
                    n_clamped += int((sd[k] < 0).sum())
                    sd[k] = sd[k].clamp_min(0.0)
        model.load_state_dict(sd)
        if args.bn_mode == "recalibrate":
            recalibrate_bn(model, Xrc, args)
        trl, tra = evaluate(model, Xtr, Ytr, args)
        tel, tea = evaluate(model, Xte, Yte, args)
        for k, v in zip(res, [float(a), trl, tra, tel, tea, n_clamped]):
            res[k].append(v)
        log.info(f"{name} alpha {a:+.3f}  train {trl:10.4f} / {tra:6.2f}%   test {tel:10.4f} / {tea:6.2f}%")
    return res


def param_dist(sd_a, sd_b, names):
    return math.sqrt(sum(float((sd_a[k] - sd_b[k]).double().pow(2).sum()) for k in names))

def summary(hist, stop):
    h = hist[-1]
    return {"epochs": h["epoch"], "stop_reason": stop, "final_train_loss": h["train_loss"],
            "final_train_acc": h["train_acc"], "final_test_loss": h["test_loss"],
            "final_test_acc": h["test_acc"], "history": hist}


# ----------------------------------------------------------------------------- main
def setup_logging(out, name):
    (out / "logs").mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO); log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(out / "logs" / f"{name}.log")):
        h.setFormatter(fmt); log.addHandler(h)

def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out_dir)
    if args.prepare_data:
        for ds in sorted(set(DATASET_OF.values())):
            load_dataset(ds, args.data_root)
        print("datasets ready in", args.data_root)
        return
    if not args.network:
        sys.exit("--network is required (or use --prepare-data)")
    name = args.network
    setup_logging(out, name)
    ckdir = out / "checkpoints"; ckdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    log.info(f"{name} on {DATASET_OF[name]} | device {device} | torch {torch.__version__} | "
             f"args {json.dumps(vars(args))}")

    out_json = out / f"{name}_results.json"
    if args.stage != "train" and out_json.exists() and not args.force_path:
        prev = json.loads(out_json.read_text())
        if prev["train_cfg"] == sub(args, TRAIN_KEYS) and prev["path_cfg"] == sub(args, PATH_KEYS) \
                and prev["n_trials"] == args.n_trials and prev["seed"] == args.seed:
            log.info(f"{out_json} already up to date; nothing to do"); return

    data = get_data(DATASET_OF[name], args.data_root, device)
    if args.stage == "path":
        missing = [t for t in range(args.n_trials) if not (ckdir / f"{name}_trial{t}.pt").exists()]
        if missing:
            sys.exit(f"--stage path needs trained trials; missing {missing}")
    t_start = time.time()
    trials = [get_trial(name, t, data, args, device, ckdir) for t in range(args.n_trials)]
    if args.stage == "train":
        log.info(f"training done in {(time.time() - t_start) / 60:.1f} min"); return

    pnames = [n for n, _ in build_model(name, 0).named_parameters()]
    trial_info = []
    for t, b in enumerate(trials):
        info = {"trial": t, "seed": b["seed"], "sb": summary(b["sb_history"], b["sb_stop"]),
                "lb": summary(b["lb_history"], b["lb_stop"])}
        if not args.independent_init:   # Sec. 3 (p.9): ratio was 3-10 in the paper
            info["dist_ratio_sb_over_lb"] = param_dist(b["sb_state"], b["init_sb"], pnames) / \
                                            param_dist(b["lb_state"], b["init_lb"], pnames)
        trial_info.append(info)

    # Sec. 2.2.1 (p.5): a randomly chosen (SB, LB) pair from the trials.
    rng = np.random.default_rng(args.seed)
    ts = int(rng.integers(args.n_trials))
    tl = int(rng.integers(args.n_trials)) if args.independent_init else ts
    log.info(f"{name}: interpolating SB minimiser of trial {ts} and LB minimiser of trial {tl}")
    t0 = time.time()
    path = evaluate_path(name, trials[ts]["sb_state"], trials[tl]["lb_state"], data, args, device)
    result = {"network": name, "dataset": DATASET_OF[name], "seed": args.seed, "n_trials": args.n_trials,
              "train_cfg": sub(args, TRAIN_KEYS), "path_cfg": sub(args, PATH_KEYS),
              "sb_batch": trials[0]["sb_batch"], "lb_batch": trials[0]["lb_batch"],
              "pair": {"sb_trial": ts, "lb_trial": tl}, "trials": trial_info, "path": path,
              "path_seconds": time.time() - t0, "device": str(device),
              "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
              "torch": torch.__version__, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    atomic_write_json(result, out_json)
    log.info(f"wrote {out_json}")


if __name__ == "__main__":
    main()
