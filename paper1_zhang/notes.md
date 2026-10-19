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
