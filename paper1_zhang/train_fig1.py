#!/usr/bin/env python
"""One training run for the Figure 1 reproduction of Zhang et al. (ICLR 2017),
"Understanding deep learning requires rethinking generalization".

Written from the paper only. Each run writes one JSON file whose name embeds a hash of every
setting that affects the run; a finished file with an identical config is reused, anything else
is retrained. The JSON is rewritten atomically after every epoch.

Example:
    python train_fig1.py --model inception --mode labels --p 1.0 --device cuda
"""
import argparse, hashlib, json, math, os, platform, sys, time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

CODE_VERSION = "fig1-v2"            # bump when training semantics change; it is part of the cache key
CROSSING_THRESHOLDS = (0.95, 0.98, 0.99)
DEFAULT_LR = {"inception": 0.1, "alexnet": 0.01, "mlp_1x512": 0.01}   # App. A
MODES = ("labels", "shuffled_pixels", "random_pixels", "gaussian")


# ----------------------------------------------------------------------------- arguments / config
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(DEFAULT_LR))
    ap.add_argument("--mode", default="labels", choices=MODES)
    ap.add_argument("--p", type=float, default=0.0, help="label corruption probability (mode=labels)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None, help="initial LR; default per model from App. A")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--lr-decay", type=float, default=0.95, help="multiplicative LR decay, once per epoch")
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--stop-threshold", type=float, default=0.99,
                    help="convergence: full-train-set accuracy at an epoch end reaches this")
    ap.add_argument("--extra-epochs", type=int, default=2, help="epochs trained after convergence")
    ap.add_argument("--min-steps", type=int, default=0, help="never stop before this many steps")
    ap.add_argument("--log-every", type=int, default=50, help="steps per logged loss/acc window")
    ap.add_argument("--corrupt-test-labels", action="store_true")
    ap.add_argument("--subset", type=int, default=0, help="use only the first N training images (0 = all)")
    ap.add_argument("--no-tf32", action="store_true", help="disable TF32 matmul/conv on Ampere GPUs")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--results-dir", default="./results_fig1")
    ap.add_argument("--force", action="store_true", help="retrain even if a matching finished run exists")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.mode != "labels" and a.p != 0.0:
        ap.error("--p only applies to --mode labels")
    if not 0.0 <= a.p <= 1.0:
        ap.error("--p must be in [0, 1]")
    return a


def run_config(a):
    """Every setting that can change what a run computes or records. This dict defines the cache key."""
    return dict(
        code_version=CODE_VERSION, model=a.model, mode=a.mode, p=round(a.p, 6), seed=a.seed,
        batch_size=a.batch_size, lr=a.lr if a.lr is not None else DEFAULT_LR[a.model],
        momentum=a.momentum, lr_schedule="staircase_per_epoch", lr_decay=a.lr_decay,
        max_epochs=a.max_epochs, stop_threshold=a.stop_threshold, extra_epochs=a.extra_epochs,
        min_steps=a.min_steps, log_every=a.log_every,
        crossing_thresholds=sorted(set(CROSSING_THRESHOLDS) | {a.stop_threshold}),
        corrupt_test_labels=a.corrupt_test_labels, subset=a.subset,
        tf32=(not a.no_tf32) and a.device == "cuda", device=a.device,
        loss="cross_entropy", init="pytorch_default", preprocessing="div255_centercrop28_perimagewhiten",
    )


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def result_path(results_dir, cfg):
    name = (f"{cfg['model']}__{cfg['mode']}__p{cfg['p']:.2f}__bs{cfg['batch_size']}"
            f"__s{cfg['seed']}__{config_hash(cfg)}.json")
    return os.path.join(results_dir, name)


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def get_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        sys.exit("error: --device cuda requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        sys.exit("error: --device mps requested but MPS is not available")
    return torch.device(name)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# ----------------------------------------------------------------------------- data
def load_base(data_root, subset=0):
    """CIFAR-10 scaled to [0,1] and center-cropped 32 -> 28 (App. A). NCHW float32 on CPU."""
    def prep(ds):
        x = torch.from_numpy(ds.data).float().div_(255.0).permute(0, 3, 1, 2)
        return x[:, :, 2:30, 2:30].contiguous(), torch.tensor(ds.targets, dtype=torch.long)
    xtr, ytr = prep(torchvision.datasets.CIFAR10(data_root, train=True, download=True))
    xte, yte = prep(torchvision.datasets.CIFAR10(data_root, train=False, download=True))
    if subset:
        xtr, ytr = xtr[:subset], ytr[:subset]
    return (xtr, ytr), (xte, yte)


def per_image_whiten(x):
    """TF per_image_whitening: (x - mean) / max(std, 1/sqrt(#elements)), per image."""
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True, unbiased=False)
    return (x - mean) / torch.clamp(std, min=1.0 / math.sqrt(x[0].numel()))


def corrupt_labels(y, p, gen, n_classes=10):
    """Sec. 2.1: with prob. p, independently, replace a label with a uniform random class. Drawn once."""
    y = y.clone()
    mask = torch.rand(len(y), generator=gen) < p
    y[mask] = torch.randint(0, n_classes, (int(mask.sum()),), generator=gen)
    return y


def permute_pixels_shared(x, perm):
    N, C, H, W = x.shape
    return x.reshape(N, C, H * W)[:, :, perm].reshape(N, C, H, W)


def permute_pixels_independent(x, gen):
    N, C, H, W = x.shape
    perms = torch.argsort(torch.rand(N, H * W, generator=gen), dim=1)
    return torch.gather(x.reshape(N, C, H * W), 2, perms.unsqueeze(1).expand(N, C, H * W)).reshape(N, C, H, W)


def gaussian_like(n, mean, std, gen):
    return torch.randn(n, 3, 28, 28, generator=gen) * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)


def build_data(cfg, base, device):
    """Applies the randomization, whitens, and moves everything to `device` once."""
    (xtr, ytr), (xte, yte) = base
    mode_id = MODES.index(cfg["mode"])
    gen = torch.Generator().manual_seed(10_000 * cfg["seed"] + 1000 * mode_id + int(round(100 * cfg["p"])))
    if cfg["mode"] == "labels":
        ytr = corrupt_labels(ytr, cfg["p"], gen)
        if cfg["corrupt_test_labels"]:
            yte = corrupt_labels(yte, cfg["p"], gen)
    elif cfg["mode"] == "shuffled_pixels":
        perm = torch.randperm(28 * 28, generator=gen)
        xtr, xte = permute_pixels_shared(xtr, perm), permute_pixels_shared(xte, perm)
    elif cfg["mode"] == "random_pixels":
        xtr, xte = permute_pixels_independent(xtr, gen), permute_pixels_independent(xte, gen)
    elif cfg["mode"] == "gaussian":
        mean, std = xtr.mean(dim=(0, 2, 3)), xtr.std(dim=(0, 2, 3))
        xtr, xte = gaussian_like(len(xtr), mean, std, gen), gaussian_like(len(xte), mean, std, gen)
    xtr, xte = per_image_whiten(xtr), per_image_whiten(xte)
    return (xtr.to(device), ytr.to(device)), (xte.to(device), yte.to(device))


# ----------------------------------------------------------------------------- models
class BatchNormShift(nn.Module):
    """BatchNorm with a learned shift and no learned scale (reproduces Table 1's parameter count)."""
    def __init__(self, c):
        super().__init__()
        self.bn = nn.BatchNorm2d(c, affine=False)
        self.beta = nn.Parameter(torch.zeros(c))
    def forward(self, x):
        return self.bn(x) + self.beta.view(1, -1, 1, 1)


class ConvModule(nn.Module):
    """Fig. 3: Convolution -> Batch Norm -> ReLU."""
    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, padding=k // 2, bias=False)
        self.bn = BatchNormShift(cout)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class InceptionModule(nn.Module):
    def __init__(self, cin, ch1, ch3):
        super().__init__()
        self.b1, self.b3, self.out_channels = ConvModule(cin, ch1, 1, 1), ConvModule(cin, ch3, 3, 1), ch1 + ch3
    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x)], dim=1)


class DownsampleModule(nn.Module):
    def __init__(self, cin, ch3):
        super().__init__()
        self.conv, self.pool = ConvModule(cin, ch3, 3, 2), nn.MaxPool2d(3, stride=2, padding=1)
        self.out_channels = cin + ch3
    def forward(self, x):
        return torch.cat([self.conv(x), self.pool(x)], dim=1)


class SmallInception(nn.Module):
    """Fig. 3: 28x28x3 input, 28 -> 14 -> 7, global 7x7 mean pool, 10-way FC. 1,649,402 params."""
    def __init__(self, n_classes=10):
        super().__init__()
        layers, c = [ConvModule(3, 96, 3, 1)], 96
        for item in [("I", 32, 32), ("I", 32, 48), ("D", 80), ("I", 112, 48), ("I", 96, 64), ("I", 80, 80),
                     ("I", 48, 96), ("D", 96), ("I", 176, 160), ("I", 176, 160)]:
            m = InceptionModule(c, item[1], item[2]) if item[0] == "I" else DownsampleModule(c, item[1])
            layers.append(m)
            c = m.out_channels
        self.features, self.fc = nn.Sequential(*layers), nn.Linear(c, n_classes)
    def forward(self, x):
        return self.fc(self.features(x).mean(dim=(2, 3)))


class LRN(nn.Module):
    """Cross-channel LRN in torch.nn.LocalResponseNorm's convention, AlexNet constants."""
    def __init__(self, size=5, alpha=5e-4, beta=0.75, k=2.0):
        super().__init__()
        self.size, self.alpha, self.beta, self.k = size, alpha, beta, k
    def forward(self, x):
        N, C, H, W = x.shape
        mean = F.avg_pool2d(x.pow(2).reshape(N, 1, C, H * W), (self.size, 1), stride=1,
                            padding=(self.size // 2, 0), count_include_pad=True)
        return x / (self.k + self.alpha * mean.reshape(N, C, H, W)).pow(self.beta)


class SmallAlexNet(nn.Module):
    """App. A: 2 x (conv5x5 -> maxpool3x3 -> LRN), FC 384, FC 192, linear 10. 1,387,786 params."""
    def __init__(self, n_classes=10):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 5, padding=2), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(3, stride=2, padding=1), LRN())
        self.features = nn.Sequential(block(3, 64), block(64, 64))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 384), nn.ReLU(inplace=True),
                                        nn.Linear(384, 192), nn.ReLU(inplace=True), nn.Linear(192, n_classes))
    def forward(self, x):
        return self.classifier(self.features(x))


class MLP(nn.Module):
    """App. A: MLP 1x512 = one hidden layer of 512 ReLU units. 1,209,866 params."""
    def __init__(self, hidden=(512,), in_dim=3 * 28 * 28, n_classes=10):
        super().__init__()
        layers, d = [nn.Flatten()], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            d = h
        self.net = nn.Sequential(*layers, nn.Linear(d, n_classes))
    def forward(self, x):
        return self.net(x)


MODELS = {"inception": SmallInception, "alexnet": SmallAlexNet, "mlp_1x512": MLP}
TABLE1_PARAMS = {"inception": 1_649_402, "alexnet": 1_387_786, "mlp_1x512": 1_209_866}


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------- training
@torch.inference_mode()
def evaluate(model, x, y, bs=2000):
    was_training = model.training
    model.eval()
    correct = torch.zeros((), device=x.device)
    loss = torch.zeros((), device=x.device)
    for i in range(0, len(x), bs):
        out = model(x[i:i + bs])
        loss += F.cross_entropy(out, y[i:i + bs], reduction="sum")
        correct += (out.argmax(1) == y[i:i + bs]).sum()
    model.train(was_training)
    return correct.item() / len(x), loss.item() / len(x)


def train(cfg, path, device, base, verbose=True):
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = cfg["tf32"]
        torch.backends.cudnn.allow_tf32 = cfg["tf32"]

    (xtr, ytr), (xte, yte) = build_data(cfg, base, device)       # all data lives on `device`
    model = MODELS[cfg["model"]]().to(device)
    assert n_params(model) == TABLE1_PARAMS[cfg["model"]], "architecture does not match Table 1"
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])
    n, bs = len(xtr), cfg["batch_size"]
    steps_per_epoch = math.ceil(n / bs)

    rec = dict(status="running", config=cfg, config_hash=config_hash(cfg), n_params=n_params(model),
               n_train=n, n_test=len(xte), steps_per_epoch=steps_per_epoch,
               env=dict(torch=torch.__version__, platform=platform.platform(),
                        gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None),
               step_log=dict(step=[], loss=[], acc=[]),   # windowed online training loss/acc (panel a)
               epochs=[],                                 # end-of-epoch full train / test evaluation
               crossings={f"{t:g}": None for t in cfg["crossing_thresholds"]},
               converged=None, hit_epoch_cap=None, final=None)

    win_loss = torch.zeros((), device=device)
    win_correct = torch.zeros((), device=device)
    ep_loss = torch.zeros((), device=device)
    ep_correct = torch.zeros((), device=device)
    win_n, step, stop_step = 0, 0, None
    t0, train_seconds = time.time(), 0.0
    model.train()
    for epoch in range(1, cfg["max_epochs"] + 1):
        lr = cfg["lr"] * cfg["lr_decay"] ** (epoch - 1)
        for g in opt.param_groups:
            g["lr"] = lr
        ep_loss.zero_(); ep_correct.zero_(); ep_n = 0
        t_ep = time.time()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = xtr[idx], ytr[idx]
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            with torch.no_grad():
                bl, bc = loss.detach() * len(yb), (out.detach().argmax(1) == yb).sum()
                win_loss += bl; win_correct += bc; ep_loss += bl; ep_correct += bc
            win_n += len(yb); ep_n += len(yb)
            if step % cfg["log_every"] == 0:              # host sync only at log points
                rec["step_log"]["step"].append(step)
                rec["step_log"]["loss"].append(win_loss.item() / win_n)
                rec["step_log"]["acc"].append(win_correct.item() / win_n)
                win_loss.zero_(); win_correct.zero_(); win_n = 0
            if stop_step is not None and step >= stop_step:
                break
        sync(device)
        train_seconds += time.time() - t_ep

        train_acc, train_loss = evaluate(model, xtr, ytr)   # on the labels the model is trained on
        test_acc, test_loss = evaluate(model, xte, yte)
        e = dict(epoch=epoch, step=step, lr=lr, online_train_loss=ep_loss.item() / ep_n,
                 online_train_acc=ep_correct.item() / ep_n, train_acc=train_acc, train_loss=train_loss,
                 test_acc=test_acc, test_loss=test_loss, test_error=1 - test_acc,
                 train_seconds=train_seconds, wall_seconds=time.time() - t0)
        rec["epochs"].append(e)
        for t in cfg["crossing_thresholds"]:
            if rec["crossings"][f"{t:g}"] is None and train_acc >= t:
                rec["crossings"][f"{t:g}"] = dict(step=step, epoch=epoch, train_acc=train_acc,
                                                  test_error=1 - test_acc, train_seconds=train_seconds)
        if rec["converged"] is None and train_acc >= cfg["stop_threshold"]:
            rec["converged"] = dict(step=step, epoch=epoch, train_acc=train_acc, test_error=1 - test_acc,
                                    train_seconds=train_seconds)
            stop_step = max(step + cfg["extra_epochs"] * steps_per_epoch, cfg["min_steps"])
        write_json_atomic(path, rec)                       # progress is visible and crash-safe
        if verbose:
            print(f"ep {epoch:3d} step {step:6d} lr {lr:.4g} | online loss {e['online_train_loss']:.4f} "
                  f"| train acc {train_acc:.4f} | test err {1 - test_acc:.4f} | {train_seconds:.0f}s", flush=True)
        if stop_step is not None and step >= stop_step:
            break

    rec["hit_epoch_cap"] = rec["converged"] is None
    last = rec["epochs"][-1]
    rec["final"] = dict(step=step, epoch=last["epoch"], train_acc=last["train_acc"],
                        test_error=last["test_error"], train_seconds=train_seconds)
    rec["status"] = "done"
    write_json_atomic(path, rec)
    if rec["hit_epoch_cap"]:
        print(f"WARNING: hit the {cfg['max_epochs']}-epoch cap without reaching train acc "
              f"{cfg['stop_threshold']} (final train acc {last['train_acc']:.4f})", flush=True)
    return rec


def main(argv=None):
    a = parse_args(argv)
    cfg = run_config(a)
    os.makedirs(a.results_dir, exist_ok=True)
    path = result_path(a.results_dir, cfg)
    if os.path.exists(path) and not a.force:
        with open(path) as f:
            old = json.load(f)
        if old.get("status") == "done" and old.get("config") == cfg:
            print(f"cached: {path}")
            return 0
        print(f"incomplete or mismatched result at {path}; retraining")
    device = get_device(a.device)
    print(f"run {os.path.basename(path)} on {device}", flush=True)
    rec = train(cfg, path, device, load_base(a.data_root, a.subset), verbose=not a.quiet)
    conv = rec["converged"]
    print(f"done: {path}\n  converged: {conv}\n  final: {rec['final']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
