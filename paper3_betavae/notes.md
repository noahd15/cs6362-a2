## Phase 2 — review of LLM output
- Architecture and optimizer match Table 1 "2D shapes (VAE)" exactly:
  Adagrad lr 1e-2; input 4096; encoder FC 1200-1200 ReLU; 10 latents;
  decoder FC 1200-1200-1200-4096 Tanh; Bernoulli output.
- Objective: summed Bernoulli NLL over pixels + beta * KL summed over
  latents (Eq. 4). Fig. 7: panel A beta=4, panel B beta=1.
- Unstated in paper, LLM assumed: batch size 64; 500k iterations per run;
  encoder head outputs (mu, log sigma^2); PyTorch default init; sampling
  with replacement.
- Seeds: beta in {1,4} x seeds {0,1,2} (run.sh).
- Data: official DeepMind dSprites, 737,280 binary images. Code checks
  which label column is which factor from the images (position via
  center-of-mass correlation; shape via rotational-symmetry IoU).
- Analysis caches posterior means for all images plus learnt variance per
  latent (the Fig. 7 ordering criterion, per its caption).
- TODO: check notebook for latent ordering and traversal (+-3 sd from a
  seed image's posterior mean, other 9 latents fixed).
- Notebook reviewed: columns ordered by ascending learnt variance (Fig. 7
  last row); traversal sets one latent to 5 values in [+3,-3] (prior mean
  0, prior sd 1) with the other 9 fixed at the posterior mean of one
  dataset image; rows 1-3 and shape colours match the caption; A = beta 4,
  B = beta 1; one figure per seed, no seed selected.
- Notebook assumptions: shows only latents with KL > 0.05 plus one
  Z_unused; factor labels from an eta^2 > 0.5 rule (paper labels by eye).
