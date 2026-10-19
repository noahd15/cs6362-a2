## Phase 3 — sweep results (Oct 18, 82 min, 23/23 runs, RTX 3090)
- Panel (a): inception true-label and p=1 runs stopped early (6,256 and
  19,159 steps) because --min-steps 25000 only went to the input-
  randomization runs. Reran both with --min-steps 25000.
- AlexNet p=1.0 hit the 100-epoch cap at train acc 0.922; never crossed
  0.95. Paper Table 1 reports 99.82%. Hypothesis: lr decay 0.95/epoch
  from 0.01 leaves lr ~6e-5 by epoch 100, so the run stalls. Test:
  rerun with --max-epochs 200; if accuracy barely moves, it's the schedule.
- Three runs (gaussian; p=1 bs64; p=1 bs256) share final test error
  0.9012. Checked per-epoch curves: all three differ throughout and
  fluctuate around 0.90 (chance) by about ±0.005. Coincidence, not cache
  reuse; confirms the near-chance models end at chance as expected.
- RESULT, AlexNet p=1.0, 200 epochs: train acc 0.9639 (vs 0.9217 at 100
  epochs; paper Table 1: 99.82%). Crossed 0.95 but not 0.98/0.99. Growth
  decelerates because the schedule 0.01*0.95^e has only ~0.6% of its total
  lr mass after epoch 100 and ~0.004% after 200, so the run is effectively
  frozen. Conclusion: under the paper's stated App. A schedule AlexNet
  cannot reach Table 1's random-label accuracy; the authors' actual
  training must have differed (schedule, batch size or init). Documented
  mismatch, not a bug. For panel (b), AlexNet p=1 converges only under the
  0.95 threshold; report threshold sensitivity.
- Fig. 1 generated with plot_fig1.py at thresholds 0.95/0.98/0.99
  (test error at crossing). Main figure: 0.99. AlexNet p=1.0 never reached
  0.99 -> omitted from (b)/(c) and flagged; it does count at 0.95. The
  100-epoch AlexNet p=1 run moved to results_fig1_extra/.
- Batch-size sensitivity, Inception random labels, steps to 0.99:
  bs64 50,830 (ep 65); bs128 18,377 (ep 47); bs256 8,232 (ep 42). Step
  axis of Fig. 1a/1b depends strongly on the unstated batch size.
- Run-to-run: original p=1 bs128 run and its min-steps rerun (same seed)
  cross 0.95 at 17,986 vs 17,595 steps (~2%), GPU nondeterminism.
