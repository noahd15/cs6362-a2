## Phase 2 — review of LLM output
- Core setup matches the paper: SB 256, LB 10% of train set, ADAM, alpha in
  [-1,2], loss+acc on train+test, alpha=0 SB / alpha=1 LB.
- Unstated in paper, LLM assumed (all flagged in code): ADAM defaults
  (lr 1e-3); stopping rule = train loss fails to improve 0.1% for 15
  epochs (paper: "until the loss function ceased to improve"); 25 alpha
  points; shared init for SB/LB; BN stats recomputed at each alpha on full
  train set (interpolated running stats kept as ablation); PyTorch default
  init; pixels scaled to [0,1], no augmentation.
- Appendix B.4 says "C3" where C2 is meant; LLM read it as a typo.
- C1 notation [64,5,5,2]: App. B.3 defines n×[a,b,c,d] with d = stride,
  so stride 2 is correct. MaxPool(3) stride/padding unstated; LLM chose
  stride 2, pad 1 (assumption).
- C2 forward+backward at batch 5000, fp32, peaks at 13.2 GB on the 3090.
  Full large batch fits; no micro-batching or AMP needed, so BN
  normalizes over the true 5000-sample batch as in the paper.
- LLM verified only on CPU with random stand-in tensors, never real data.

## Phase 3 — results
- F1 (RTX 3090, ~2.5 min): SB stopped at epoch 57 (plateau), train 99.87%
  test 98.65%. LB ran to 300-epoch cap (loss still improving >0.1%),
  train 100.00% test 98.62%. Both fit the training set (paper: 99.66/
  99.92). Generalization gap 0.03 pts vs paper's 0.22 pts; F1 has the
  smallest gap of the six networks in the paper as well.
- C1, default stopping (patience 15, 0.1% rel. improvement): SB stopped
  epoch 119, train 98.37% test 68.79%; LB stopped epoch 111, train 96.61%
  test 65.15%. Paper Table 2: SB 99.89/80.04, LB 99.66/77.26. Gap
  direction reproduced (SB-LB test = 3.6 pts vs paper 2.8) but neither run
  fit the training set, and LB fit it worse, which confounds Fig. 3.
  Diagnosis: plateau rule too aggressive vs paper's "until the loss
  function ceased to improve". Fix (Phase 3): rerun with --patience 50
  into results_patience50/, keeping the original run as evidence.
- Remaining test-accuracy shortfall (~11 pts) may be architecture: the
  MaxPool(3) stride/padding is unstated. Check in Phase 4 against the
  authors' released code (github.com/keskarnitish/large-batch-training).
- C1 rerun, --patience 50 (results_patience50/): LB training log ends at
  98.42% train, but that eval uses training-time BN running stats, which
  are noisy with only 10 LB steps/epoch. With BN recalibrated on the full
  train set (path eval): alpha=0 (SB) train 99.80% test 70.61%; alpha=1
  (LB) train 99.86% test 67.22%. Both now fit the training set; SB-LB test
  gap 3.4 pts (paper 2.8).
- C1 path: SB side flat (train loss 0.006 at alpha=0 -> 0.36 at alpha=-1);
  LB side sharp (0.006 at alpha=1 -> 6.07 at alpha=1.125, train acc 39%).
  Loss barrier between them (max ~2.0 near alpha=0.875): separate basins.
  Qualitatively matches Fig. 3. Test acc ~10 pts below paper for both;
  suspect unstated MaxPool(3) stride/padding -> check in Phase 4.
- C2 SB under default patience 15: plateau at epoch 115, train 99.83%
  test 86.74% (paper 99.99/89.24). Fits the training set; kept.
- Stopping-rule decision: default patience 15; raised to 50 only for C1,
  where the default left both runs short of fitting the training set.
  C2 LB judged by alpha=1 train acc after BN recalibration; rerun with
  patience 50 only if it falls short.
- C1 rerun, --patience 50 (results_patience50/): LB training log ends at
  98.42% train, but that eval uses training-time BN running stats, which
  are noisy with only 10 LB steps/epoch. With BN recalibrated on the full
  train set (path eval): alpha=0 (SB) train 99.80% test 70.61%; alpha=1
  (LB) train 99.86% test 67.22%. Both now fit the training set; SB-LB test
  gap 3.4 pts (paper 2.8).
- C1 path: SB side flat (train loss 0.006 at alpha=0 -> 0.36 at alpha=-1);
  LB side sharp (0.006 at alpha=1 -> 6.07 at alpha=1.125, train acc 39%).
  Loss barrier between them (max ~2.0 near alpha=0.875): separate basins.
  Qualitatively matches Fig. 3. Test acc ~10 pts below paper for both;
  suspect unstated MaxPool(3) stride/padding -> check in Phase 4.
- C2 SB under default patience 15: plateau at epoch 115, train 99.83%
  test 86.74% (paper 99.99/89.24). Fits the training set; kept.
- Stopping-rule decision: default patience 15; raised to 50 only for C1,
  where the default left both runs short of fitting the training set.
  C2 LB judged by alpha=1 train acc after BN recalibration; rerun with
  patience 50 only if it falls short.
- C2 LB (default patience 15): plateau at epoch 73; training-log eval
  96.66% / 77.86% (training-time BN stats). Path eval with BN recalibrated:
  alpha=0 (SB) train 99.91% test 87.09%; alpha=1 (LB) train 99.64% test
  81.24% (paper SB 99.99/89.24, LB 99.99/87.26). Above the 99.5% bar, kept.
  SB-LB test gap 5.9 pts vs paper 2.0; LB stopped earlier and fits
  slightly less, which may inflate the gap.
- Final Fig. 3 sources: F1 and C2 from results/, C1 from
  results_patience50/.
- Fig. 3 reproduced (plot_results.ipynb from saved JSONs). Qualitative
  match on all three panels: F1 flat through alpha=1 then sharp rise after
  LB; C1 barrier just before alpha=1 then spike; C2 dip near alpha~0.75 as
  in the paper.
- Distance from init, ||x_s-x0|| / ||x_l-x0||: F1 9.32, C1 4.13, C2 11.87
  (paper Sec. 3: 3-10). Independent confirmation that SB travels much
  farther from the shared init than LB.
- Fig. 3 reproduced (plot_results.ipynb from saved JSONs). Qualitative
  match on all three panels: F1 flat through alpha=1 then sharp rise after
  LB; C1 barrier just before alpha=1 then spike; C2 dip near alpha~0.75 as
  in the paper.
- Distance from init, ||x_s-x0|| / ||x_l-x0||: F1 9.32, C1 4.13, C2 11.87
  (paper Sec. 3: 3-10). Independent confirmation that SB travels much
  farther from the shared init than LB.
