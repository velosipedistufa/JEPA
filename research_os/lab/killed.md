# Killed mechanisms

Do not reopen these unless a protocol changes the mechanism, not the wording.

| id | mechanism | why |
|---|---|---|
| K1 | Guessed global action scale (×5 or similar) | F5. Official z-score is the only action hook that reached 90%. |
| K2 | Constant Δz lookup per command string | F3. Residual ratio 0.991. |
| K3 | Treating in-place 180° failure as a missing text encoder | F7. A true rendered goal image also scored 0. |
| K4 | Finetuning LeWM because 180° is missing from z | F2. Linear probe already reads 180°. |
| K5 | Text-to-image goal frames | Not measured as a success; rejected because the env frame is the goal, and a generator will not match Push-T calibration. |
| K6 | Reading the official probe-cost reach 0.10 as an in-place quarter-turn | F12. The two angle hits in `reports/gpu_4090/logs/probe_cost_official.json` have xy error 114.95 px (episode 52) and 130.42 px (episode 74). The oracle gate, angle `< 20°` and xy `< 15` px, holds for 0/20. |

Cycle 001 killed no proposed bet. B2 and B8 were cut as second projects. They are not rows here.
