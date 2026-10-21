# CS 6362 Assignment 2: Paper Reproductions

Reproductions of one figure from each of three ICLR 2017 papers, implemented in PyTorch with the help of an LLM (Claude Opus 5.5, via the claude.ai chat interface), as required by the assignment.

| Folder | Paper | Target figure |
|---|---|---|
| `paper1_zhang/` | Zhang et al., *Understanding deep learning requires rethinking generalization* | Figure 1 (a)–(c) |
| `paper2_keskar/` | Keskar et al., *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima* | Figure 3 (F1, C1, C2) |
| `paper3_betavae/` | Higgins et al., *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework* | Figure 7 (A, B) |

Each folder contains a training script that writes logs and results to disk, a driver script that runs the full sweep, and a notebook that regenerates the figure **from the saved results only**, without retraining. Each folder's `notes.md` is the running log of failures, fixes and findings that the reports draw on.

## Environment

Experiments ran on a single NVIDIA RTX 3090 (24 GB) with PyTorch 2.14.1 (CUDA 13.0) and torchvision 0.29.1.

```bash
pip install torch torchvision numpy matplotlib nbconvert ipykernel
```

CIFAR-10 and MNIST are downloaded by torchvision on first use. dSprites is downloaded from the official DeepMind repository (`deepmind/dsprites-dataset`) by `paper3_betavae/train.py --prepare-data`.

## Reproducing each figure

### Paper 1: Zhang et al., Figure 1

```bash
cd paper1_zhang
bash run_sweep.sh                       # 23 runs, ~80 min on a 3090; results in results_fig1/
jupyter nbconvert --to notebook --execute figure1.ipynb
```

`plot_fig1.py --threshold {0.95,0.98,0.99}` regenerates the figure at each convergence threshold. The main figure uses 0.99. The Phase 3 reruns used for panel (a) and the AlexNet diagnosis are:

```bash
python train_fig1.py --model inception --mode labels --p 0.0 --min-steps 25000
python train_fig1.py --model inception --mode labels --p 1.0 --min-steps 25000
python train_fig1.py --model alexnet   --mode labels --p 1.0 --max-epochs 200
```

### Paper 2: Keskar et al., Figure 3

```bash
cd paper2_keskar
bash run.sh                                                             # F1, C1, C2; results in results/
python train.py --network C1 --patience 50 --out-dir results_patience50   # Phase 3 C1 run used in the figure
jupyter nbconvert --to notebook --execute plot_results.ipynb
```

The figure uses `results/` for F1 and C2 and `results_patience50/` for C1. Copy `results_patience50/C1_results.json` to `results/C1_results.json` before running the notebook.

**Phase 4:** `train_phase4.py` differs from `train.py` in one line only: the C1 convolution stride is 1, as in the authors' released code, instead of the stride 2 that the paper's Appendix B.3 states.

```bash
python train_phase4.py --network C1 --patience 50 --out-dir results_phase4
```

### Paper 3: Higgins et al., Figure 7

```bash
cd paper3_betavae
bash run.sh                             # beta in {1,4} x seeds {0,1,2}, 500k iterations each; ~2 h on a 3090
jupyter nbconvert --to notebook --execute figure7.ipynb
```

The notebook draws one figure per seed (`figures/figure7AB_seed{0,1,2}.png`) and prints per-latent statistics for every run. Model checkpoints are not committed because of their size; rerun `run.sh` to recreate them.

## Phase 3 commits

The code as it stood at the end of Phase 3, before any existing implementation was consulted, is tagged:

| Tag | Commit |
|---|---|
| `p1-phase3` | `b65ab89` |
| `p2-phase3` | `b65ab89` |
| `p3-phase3` | `bf2e7ba` |

Only Keskar et al. released code ([keskarnitish/large-batch-training](https://github.com/keskarnitish/large-batch-training)). The single change made after consulting it is `paper2_keskar/train_phase4.py`, described above and in the report.
