# Open questions

Ranked after cycle 010. No Lance download, no HDF5 download, no CEM, and no finetune on the Xeon. The Xeon falsifiers through B11 are closed. The next bets need a 24 GB GPU. Starting one is an escalate.

Papers may not replace the pinned local numbers in `BASELINES.md`.

## 1. B1 is the only parked GPU bet

Offset 50 with the fixed official planner falls to at most 8/20. Falsifier: at least 14/20. It needs the Lance table on the GPU machine. That table was not downloaded. B3, B4, and B9 have ledger rows 012, 013, and 011.

## 2. Closed on the Xeon

Episode-level xy MAE and the 15 px gate held on seeds 0 and 1 (ledger rows 004 and 005). The +90 matrix, the state-dependent command map, and the cross-episode pose catalog were killed (rows 006, 007, 008). One frame does not linearly reveal commanded `v_x` on two blocks (rows 009 and 010): R² -0.0030707419982412 and -0.012655494701210301.

## 2. Episode-level xy MAE is supported

Ledger row 004. Seed 0: x=11.710440094140232, y=9.989942039258745. Seed 1: x=10.213864689000905, y=10.989067528789096. Do not add a model class.

## 3. Frame-level xy ridge, seeds 0 and 1, is supported

Seed 0, ledger row 002: test MAE x=12.70093186924053 px, y=9.677809205954556 px. Seed 1, ledger row 003: test MAE x=11.374770603169216 px, y=11.400641675149357 px. Both beat their train-mean constant predictors. Do not add a model class to this result.

## 3. B6, B7, B10, and B11 are closed

B6 killed: +90 test ratio 1.3427727374663092. B7 killed: state-map test ratio 1.1463429404945915. B10 killed: median angle 100.51038138097391°. B11 supported on two blocks. Do not refit them to change the result.

## Cut

B2 (expert-chunk CEM for in-place 180°) and B8 (five-step open-loop curve) stay cut. Both need the Lance table. Source: `research_os/cycles/001/04_protocols.md` parked cuts.
