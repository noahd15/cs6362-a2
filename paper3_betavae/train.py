#!/usr/bin/env python
"""beta-VAE on 2D shapes (Higgins et al., ICLR 2017), written from the paper only.

One invocation = one run (one beta, one seed). Everything goes to <runs>/<beta..._seed...>/:
    config.json      arguments of the run
    log.jsonl        step, loss, reconstruction, total and per-latent KL, throughput
    status.json      running | done | diverged
    ckpt_last.pt     periodic checkpoint (model, optimiser, RNG and sampler state) for resuming
    final.pt         final model weights
    analysis.npz     posterior means of all 737,280 images + learnt variance + KL per latent
                     (computed once after training; the notebook plots from this)

Modes:
    python train.py --prepare-data                 download + verify dSprites, write uncompressed cache
    python train.py --benchmark                    time training steps, project wall time per budget
    python train.py --beta 4 --seed 0              train (resumes automatically), then analyse
    python train.py --beta 4 --seed 0 --analyze-only   recompute analysis.npz from final.pt
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DSPRITES_URL = ('https://raw.githubusercontent.com/deepmind/dsprites-dataset/master/'
                'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
NPZ_NAME = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
BUDGETS = {'full': 500_000, 'reduced': 150_000, 'smoke': 300}   # iterations per run (assumption)
FACTORS = ['shape', 'scale', 'rotation', 'pos_x', 'pos_y']


def run_name(beta, seed):
    return f"beta{float(beta):g}_seed{int(seed)}"


def log(msg):
    print(time.strftime('[%Y-%m-%d %H:%M:%S] ') + msg, flush=True)


# ----------------------------------------------------------------------------- data
def prepare_data(data_dir):
    """Download dSprites, verify what each label column means from the images themselves,
    and write imgs_u8.npy (N, 4096) for fast memory-mapped loading plus meta.npz."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    imgs_path, meta_path = data_dir / 'imgs_u8.npy', data_dir / 'meta.npz'
    if imgs_path.exists() and meta_path.exists():
        return
    npz_path = data_dir / NPZ_NAME
    if not npz_path.exists():
        log(f'downloading {DSPRITES_URL}')
        urllib.request.urlretrieve(DSPRITES_URL, npz_path)
        if npz_path.stat().st_size < 1_000_000:
            npz_path.unlink()
            sys.exit(f'download failed; place {NPZ_NAME} from the dsprites-dataset repo in {data_dir}')

    # 'metadata' inside the npz is pickled and is never read.
    with np.load(npz_path, allow_pickle=False) as d:
        imgs = d['imgs']
        classes = d['latents_classes'].astype(np.int64)
        values = d['latents_values'].astype(np.float64)
    N = imgs.shape[0]
    card = [len(np.unique(classes[:, j])) for j in range(classes.shape[1])]
    log(f'images {imgs.shape} {imgs.dtype}, pixel values {np.unique(imgs[:1000])}, cardinalities {card}')
    assert N == 737280 and sorted(card) == [1, 3, 6, 32, 32, 40]

    col = {'shape': card.index(3), 'scale': card.index(6), 'rotation': card.index(40)}
    pos_cols = [j for j, c in enumerate(card) if c == 32]
    # Horizontal vs vertical position: correlate each 32-valued column with the centre of mass.
    sub = np.random.default_rng(0).choice(N, 20000, replace=False)
    im = imgs[sub].astype(np.float64)
    mass = im.sum((1, 2))
    com_row = (im.sum(2) * np.arange(64)).sum(1) / mass
    com_col = (im.sum(1) * np.arange(64)).sum(1) / mass
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    c = {j: (corr(classes[sub, j], com_col), corr(classes[sub, j], com_row)) for j in pos_cols}
    col['pos_x'] = max(pos_cols, key=lambda j: abs(c[j][0]))
    col['pos_y'] = [j for j in pos_cols if j != col['pos_x']][0]
    pos_y_down = c[col['pos_y']][1] > 0
    log(f'position check (corr with column COM, row COM): {c}; columns {col}; pos_y increases downward: {pos_y_down}')

    dims = [card[col[k]] for k in FACTORS]
    G = -np.ones(dims, dtype=np.int64)
    G[tuple(classes[:, col[k]] for k in FACTORS)] = np.arange(N)
    assert (G >= 0).all()

    # Shapes by rotational symmetry: square matches itself after a quarter turn, oval after a half turn.
    rot_vals = np.array([values[G[0, 0, r, 0, 0], col['rotation']] for r in range(dims[2])])
    r_q, r_h = int(np.argmin(np.abs(rot_vals - np.pi / 2))), int(np.argmin(np.abs(rot_vals - np.pi)))
    def iou(a, b):
        a, b = a.astype(bool), b.astype(bool)
        return (a & b).sum() / (a | b).sum()
    sym = {s: (iou(imgs[G[s, 5, 0, 16, 16]], imgs[G[s, 5, r_q, 16, 16]]),
               iou(imgs[G[s, 5, 0, 16, 16]], imgs[G[s, 5, r_h, 16, 16]])) for s in range(3)}
    square = max(sym, key=lambda s: sym[s][0])
    rest = [s for s in sym if s != square]
    oval = max(rest, key=lambda s: sym[s][1])
    heart = [s for s in rest if s != oval][0]
    names = np.array([''] * 3, dtype='<U6')
    names[square], names[oval], names[heart] = 'square', 'oval', 'heart'
    log(f'IoU after quarter/half turn: { {names[s]: tuple(round(float(v), 3) for v in sym[s]) for s in sym} }; '
        f'shape classes {list(names)}')

    np.savez(meta_path, G=G, shape_names=names, pos_y_down=np.array(pos_y_down), rot_values=rot_vals)
    tmp = data_dir / 'imgs_u8.npy.tmp.npy'
    np.save(tmp, imgs.reshape(N, -1))
    os.replace(tmp, imgs_path)
    log(f'wrote {imgs_path} and {meta_path}')


def load_images(data_dir, device):
    """All images as a uint8 tensor (N, 4096) on `device`, copied in chunks to bound host RAM."""
    prepare_data(data_dir)
    mm = np.load(Path(data_dir) / 'imgs_u8.npy', mmap_mode='r')
    X = torch.empty(mm.shape, dtype=torch.uint8, device=device)
    for i in range(0, mm.shape[0], 65536):
        X[i:i + 65536] = torch.from_numpy(np.array(mm[i:i + 65536]))
    return X


# ----------------------------------------------------------------------------- model
class BetaVAE(nn.Module):
    """Table 1, '2D shapes (VAE)': encoder FC 1200-1200 ReLU, 10 latents,
    decoder FC 1200-1200-1200-4096 Tanh with Bernoulli output."""

    def __init__(self, x_dim=4096, z_dim=10, enc_hidden=(1200, 1200), dec_hidden=(1200, 1200, 1200)):
        super().__init__()
        layers, d = [], x_dim
        for h in enc_hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.enc = nn.Sequential(*layers)
        self.enc_head = nn.Linear(d, 2 * z_dim)          # -> (mu, log sigma^2)   [assumption]
        layers, d = [], z_dim
        for h in dec_hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers += [nn.Linear(d, x_dim)]                  # Bernoulli logits
        self.dec = nn.Sequential(*layers)
        self.z_dim = z_dim

    def encode(self, x):
        mu, logvar = self.enc_head(self.enc(x)).chunk(2, dim=1)
        return mu, logvar

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar


def model_from_config(cfg):
    return BetaVAE(cfg['x_dim'], cfg['z_dim'], cfg['enc_hidden'], cfg['dec_hidden'])


def beta_vae_loss(x, logits, mu, logvar, beta):
    """Negative of Eq. 4: per-image summed Bernoulli NLL (Eq. 7) + beta * KL, averaged over the batch."""
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction='none').sum(1)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    return (recon + beta * kl.sum(1)).mean(), recon.mean(), kl.mean(0)


# ----------------------------------------------------------------------------- training
def atomic_save(obj, path):
    tmp = str(path) + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(cfg, X, device):
    rdir = Path(cfg['run_dir']) / run_name(cfg['beta'], cfg['seed'])
    rdir.mkdir(parents=True, exist_ok=True)
    status_path, ckpt_path = rdir / 'status.json', rdir / 'ckpt_last.pt'
    cfg_path = rdir / 'config.json'
    if cfg_path.exists():   # never silently continue a run under a different setting
        old = json.loads(cfg_path.read_text())
        keys = ['beta', 'seed', 'batch_size', 'lr', 'z_dim', 'enc_hidden', 'dec_hidden', 'allow_tf32']
        diff = {k: (old.get(k), cfg[k]) for k in keys if old.get(k) != cfg[k]}
        if diff:
            sys.exit(f'{rdir}: config differs from the existing run {diff}; use a different --run-dir')
    if status_path.exists():
        st = json.loads(status_path.read_text())
        if st['status'] == 'diverged':
            log(f'{rdir.name}: diverged at step {st["step"]}, left as is')
            return 'diverged'
        if st['status'] == 'done' and st['step'] >= cfg['n_iters']:
            log(f'{rdir.name}: already done at step {st["step"]}, nothing to train')
            return 'done'
        if st['status'] == 'done':   # a larger budget extends a finished run from its last checkpoint
            log(f'{rdir.name}: extending from step {st["step"]} to {cfg["n_iters"]}')
    cfg_path.write_text(json.dumps(cfg, indent=2))

    N = X.shape[0]
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    model = model_from_config(cfg).to(device)
    opt = torch.optim.Adagrad(model.parameters(), lr=cfg['lr'])     # PyTorch defaults otherwise
    gen = torch.Generator(device=device)
    gen.manual_seed(10_000 + cfg['seed'])
    step = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        gen.set_state(ck['gen_state'].cpu())
        torch.set_rng_state(ck['torch_rng'])
        if device.type == 'cuda' and ck['cuda_rng'] is not None:
            torch.cuda.set_rng_state(ck['cuda_rng'])
        step = ck['step']
        log_path = rdir / 'log.jsonl'
        if log_path.exists():   # drop records written after the checkpoint so steps are not duplicated
            keep = [l for l in log_path.read_text().splitlines() if l.strip() and json.loads(l)['step'] <= step]
            log_path.write_text(''.join(l + '\n' for l in keep))
        log(f'{rdir.name}: resuming from step {step}')

    def checkpoint():
        atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step,
                         gen_state=gen.get_state(), torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state() if device.type == 'cuda' else None),
                    ckpt_path)
        status_path.write_text(json.dumps(dict(status='running', step=step)))

    status_path.write_text(json.dumps(dict(status='running', step=step)))
    log_f = open(rdir / 'log.jsonl', 'a')
    M = cfg['z_dim']
    acc = torch.zeros(3 + M, device=device)    # loss, recon, kl total, kl per latent
    n_acc, t0 = 0, time.time()
    model.train()
    while step < cfg['n_iters']:
        idx = torch.randint(N, (cfg['batch_size'],), device=device, generator=gen)   # with replacement
        x = X[idx].float()
        logits, mu, logvar = model(x)
        loss, recon, kl = beta_vae_loss(x, logits, mu, logvar, cfg['beta'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        with torch.no_grad():
            acc += torch.cat([torch.stack([loss, recon, kl.sum()]), kl])
        n_acc += 1

        if step % cfg['log_every'] == 0 or step == cfg['n_iters']:
            a = (acc / n_acc).tolist()
            dt = time.time() - t0
            log_f.write(json.dumps(dict(step=step, loss=a[0], recon=a[1], kl=a[2], kl_per_latent=a[3:],
                                        it_per_s=n_acc / dt, time=time.time())) + '\n')
            log_f.flush()
            if not all(math.isfinite(v) for v in a):
                status_path.write_text(json.dumps(dict(status='diverged', step=step)))
                log(f'{rdir.name}: DIVERGED at step {step}; left as is (no checkpoint written after this)')
                log_f.close()
                return 'diverged'
            if step % (cfg['log_every'] * cfg['print_every_logs']) == 0 or step == cfg['n_iters']:
                log(f"{rdir.name} step {step:>7d}/{cfg['n_iters']}  loss {a[0]:8.2f}  recon {a[1]:8.2f}  "
                    f"KL {a[2]:6.2f}  {n_acc / dt:6.0f} it/s")
            acc.zero_()
            n_acc, t0 = 0, time.time()
        if step % cfg['ckpt_every'] == 0:
            checkpoint()

    checkpoint()
    atomic_save(dict(model=model.state_dict(), step=step, config=cfg), rdir / 'final.pt')
    status_path.write_text(json.dumps(dict(status='done', step=step)))
    log_f.close()
    log(f'{rdir.name}: done at step {step}')
    return 'done'


# ----------------------------------------------------------------------------- analysis
@torch.no_grad()
def analyze(cfg, X, device, bs=16384):
    """Encode every image with the final model; store posterior means (float32), the learnt
    variance (posterior sigma^2 averaged over the data) and the data-averaged KL per latent."""
    rdir = Path(cfg['run_dir']) / run_name(cfg['beta'], cfg['seed'])
    ck = torch.load(rdir / 'final.pt', map_location=device)
    model = model_from_config(ck['config']).to(device).eval()
    model.load_state_dict(ck['model'])
    N, M = X.shape[0], ck['config']['z_dim']
    mu_all = np.empty((N, M), dtype=np.float32)
    var_sum, kl_sum = torch.zeros(M, dtype=torch.float64), torch.zeros(M, dtype=torch.float64)
    for i in range(0, N, bs):
        mu, lv = model.encode(X[i:i + bs].float())
        var = lv.exp()
        mu_all[i:i + bs] = mu.cpu().numpy()
        var_sum += var.double().sum(0).cpu()
        kl_sum += (0.5 * (mu.pow(2) + var - 1 - lv)).double().sum(0).cpu()
    out = rdir / 'analysis.npz'
    tmp = rdir / 'analysis.tmp.npz'
    np.savez(tmp, mu=mu_all, learnt_var=(var_sum / N).numpy(), kl=(kl_sum / N).numpy(),
             step=np.array(ck['step']))
    os.replace(tmp, out)
    log(f'{rdir.name}: wrote {out} (checkpoint step {ck["step"]})')


# ----------------------------------------------------------------------------- benchmark
def benchmark(cfg, X, device, n=500, warmup=50):
    model = model_from_config(cfg).to(device)
    opt = torch.optim.Adagrad(model.parameters(), lr=cfg['lr'])
    for i in range(n + warmup):
        if i == warmup:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
        x = X[torch.randint(X.shape[0], (cfg['batch_size'],), device=device)].float()
        logits, mu, lv = model(x)
        loss, _, _ = beta_vae_loss(x, logits, mu, lv, 4.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    ips = n / (time.time() - t0)
    log(f'{ips:.0f} it/s at batch {cfg["batch_size"]} (tf32={cfg["allow_tf32"]})')
    for name, it in BUDGETS.items():
        log(f'  {name:8s} {it:>7d} it/run -> {it / ips / 60:6.1f} min/run, {6 * it / ips / 3600:5.2f} h for 6 runs')


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--beta', type=float, default=4.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--budget', choices=list(BUDGETS), default='full')
    p.add_argument('--n-iters', type=int, default=None, help='overrides --budget')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--z-dim', type=int, default=10)
    p.add_argument('--log-every', type=int, default=1000)
    p.add_argument('--print-every-logs', type=int, default=10, help='print to stdout every k log records')
    p.add_argument('--ckpt-every', type=int, default=25_000)
    p.add_argument('--tf32', action='store_true', help='allow TF32 matmuls (faster on Ampere)')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--run-dir', default='runs')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--prepare-data', action='store_true')
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--analyze-only', action='store_true')
    args = p.parse_args()

    if args.prepare_data:
        prepare_data(args.data_dir)
        return
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    cfg = dict(beta=args.beta, seed=args.seed, budget=args.budget,
               n_iters=args.n_iters or BUDGETS[args.budget], batch_size=args.batch_size, lr=args.lr,
               x_dim=4096, z_dim=args.z_dim, enc_hidden=[1200, 1200], dec_hidden=[1200, 1200, 1200],
               log_every=args.log_every, print_every_logs=args.print_every_logs, ckpt_every=args.ckpt_every,
               allow_tf32=args.tf32, data_dir=str(args.data_dir), run_dir=str(args.run_dir),
               device=str(device), torch_version=str(torch.__version__))
    if device.type == 'cuda':
        cfg['gpu'] = torch.cuda.get_device_name(device)

    log(f'loading images to {device}')
    X = load_images(args.data_dir, device)
    if args.benchmark:
        benchmark(cfg, X, device)
        return
    rdir = Path(args.run_dir) / run_name(args.beta, args.seed)
    if not args.analyze_only:
        status = train(cfg, X, device)
        if status != 'done':
            return
    if not (rdir / 'final.pt').exists():
        sys.exit(f'{rdir}/final.pt missing: the run has not finished')
    out = rdir / 'analysis.npz'
    if args.analyze_only or not out.exists() or \
            int(np.load(out)['step']) != int(torch.load(rdir / 'final.pt', map_location='cpu')['step']):
        analyze(cfg, X, device)


if __name__ == '__main__':
    main()
