# Claim graph

Edges are "depends on", not proof. Paper nodes are priors. They do not overwrite a local fact.

```
F1 encoder loads
  └─ F2 angle probe (degrees, this split)
  │    └─ K4 do not finetune for θ
  │         └─ Maes 2026 Table 1 block-angle r = 0.902
  │              same direction, different metric, not a replacement
  └─ F4 official CEM 45/50
       ├─ F5 ×5 scale 0/50 → K1
       │    └─ C1 the two rates are different planners
       ├─ F9 eval_state is position < 20 and angle < 20°
       ├─ F10 five misses all fail the position gate at step 50
       ├─ F11 35/45 successes finish inside the first 25 steps
       ├─ F13 action mean and std
       ├─ F14 one-step latent error, n = 1
       ├─ F16 yaml constants match the script
       └─ C3 Caselli 2026 flat LeWM 94.0% at offset 25
            different aggregate; F4 stays the local number
F3 constant Δz ratio 0.991 → K2
  └─ Myers 2023 GRIF aligns language with an image change
       and trains a policy; K2 is only the constant-vector case
F6 probe-cost angle reach 0.10 / 0.00
  ├─ C2 the ×5 table's 0.15 is a different planner
  └─ F12 those two hits translate 115 px and 130 px → K6
F7 oracle in-place reach 0.00
  ├─ K3 do not blame a text encoder
  ├─ F17 the rendered goal really is the commanded turn
  └─ C4 Caselli 2026 oracle 73.3% uses dataset-future latents
       a different goal family from F7
F8 Xeon cannot open Lance
  └─ F18 Lance, HDF5, CLIP weights, and velocity columns are absent
F15 skill MLP was scored under the ×5 planner only
F19 history length 1, predictor window 3, as the script comment defines it
```

## Cycle 001 edges

```
B1 offset-50 fixed planner
  depends on F4, F11, F16, C3
  falsifier: ≥ 14/20 at offset 50
  parked until a 4090 and a mounted Lance table exist

B3 +30° rendered chain
  depends on F7, F13, F16, C4
  falsifier: ≤ 1/20 finish in place
  parked behind the CPU script

B4 imagined angle of the oracle plan
  depends on F7, F2, F14
  falsifier: median imagined error < 20° with executed error > 60°
  parked behind the CPU script

B5 xy ridge, B6 rotation matrix, B7 state-dependent Δz, B10 pose catalog
  depend on F2, F3, F18, and for B7 also F15 and K2
  next action: scripts/latent_maps.py on the Xeon
  falsifiers: xy MAE ≥ 40 px; R ratio ≥ 0.90; command-map ratio ≥ 0.90;
              median neighbor > 30 px or > 20°

B9 live reset goal under F13
  depends on F4, F5, F9, F13
  falsifier: ≤ 4/20
  F5 is not this result
  parked behind the CPU script

B11 velocity probe
  depends on F1, F2, F18
  falsifier: pose-matched R² > 0.50
  needs new encodes; not the next action

B2 expert-chunk CEM for in-place 180
  cut. Depends on F7 and Caselli 2026 §3.3. Needs the Lance action column.

B8 five-step open-loop curve
  cut. Depends on F14 and Maes 2026 §3.2. Needs Lance windows.
```

Open after the CPU script, still not nodes of measurement:

- whether offset 50 hurts the fixed local planner
- whether a same-xy `+30°` chain is followable
- whether the oracle CEM ever imagines the rendered `+180°` goal
- whether `info["goal"]` is inside the set F4 already reaches
