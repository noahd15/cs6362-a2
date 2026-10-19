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
