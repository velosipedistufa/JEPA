# From words to contact

`JEPA_robot_demo.mp4`: 1:00, 1920×1080, 24 fps, H.264/AAC. Original synthesized ambient soundtrack; the film is readable without audio. All experimental numbers refer to this repository's saved runs unless explicitly marked as fresh captures.

The robot footage is a MuJoCo rendering of the **actual block and pusher trajectories** from fresh frozen-LeWM/CEM Push-T runs on the RTX 5090. A UR5e follows those pusher positions through inverse kinematics. This is a visualization of embodiment, not an evaluation of a UR5e pushing policy, arm collision avoidance, joint dynamics, or sim-to-real transfer. The block motion comes from the original Push-T physics, not MuJoCo contact. Frame interpolation and playback speed are for presentation; these are not real-time planner timings. The green task pose and amber next-demo pose are rendered from the commands and captured subgoals.

## Demo overview

- 00:00 — A target request and successful actual rollout.
- 00:06 — Why contact planning is needed.
- 00:09 — Final-image-only planning misses a distant goal at budget 300.
- 00:19 — Reset to exactly the same start, goal, checkpoint and budget.
- 00:22 — Breadcrumb-guided success in 73 executed steps.
- 00:34 — Saved paired offset-100 results: 10/50 flat versus 29/50 crumbs.
- 00:40 — A clockwise turn with unwanted translation.
- 00:49 — Fresh-run failure metrics and the saved strict-turn result.
- 00:54 — Contribution and the remaining real-arm work.

Selected trajectories are shown in full at accelerated playback speed.

## Evidence and limits

`rollouts.json` retains every state, executed action, subgoal selection, and result from six new visualization captures. The main paired example is held-out demo episode 5121, start 37, offset 100. Both variants use the same initial state, goal, seed 42, frozen model, original CEM, and budget 300. Flat misses; breadcrumbs succeeds in 73 steps. Additional captures from episode 2041/start 28 are retained with the trajectory data. The target task uses reset seed 0. The turn uses reset seed 1 and misses: +90.79 degrees with 42.53 px drift. These fresh captures are illustrative examples selected using existing saved results; they do not replace or expand the frozen scored evaluations.

`parser_for_demo.json` records the frozen Qwen parser's exact outputs for both displayed commands. `ik_checks.json` reports tool tracking errors; all selected trajectories used on screen track the pusher to within 0.1 mm at recorded states. `far3_flat` and `far3_crumbs` are unused captures and are not shown in the video.

Saved aggregate sources:

- `research/artifacts/5090/dataset100_flat_b300_n50.json`
- `research/artifacts/5090/dataset100_crumbs_b300_n50.json`
- `research/artifacts/5090/words_crumbs_b300_n64.json`
- `HANDOFF_5090.md`

## Re-render without rerunning the planner

Use Python 3.11, Ubuntu fonts, FFmpeg, and an OpenGL/EGL-capable system. From this folder:

```bash
python3 -m venv /tmp/jepa-video-render
/tmp/jepa-video-render/bin/pip install mujoco==3.17.0 numpy==2.4.6 Pillow==13.0.0
python3 demo_source/fetch_assets.py
python3 demo_source/make_scene.py
MUJOCO_GL=egl /tmp/jepa-video-render/bin/python demo_source/render_short.py
```

The recorded inverse kinematics are in `robot_q.npz`; `robot.py` can regenerate them. To make new GPU captures, run `JEPA_ROOT=/path/to/JEPA /path/to/JEPA/.venv/bin/python demo_source/capture_rollouts.py` with the original checkpoint, dataset, and experiment environment available. This overwrites local visualization capture files only, never the frozen research artifacts.

## Attribution

Robot model: Universal Robots UR5e, ROS Industrial Consortium, distributed through Google DeepMind's [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/cda5236f98383f6fbaa4820c6b18dba892cd087f/universal_robots_ur5e). License retained in `demo_source/LICENSE`; exact mesh hashes and revision are in `demo_source/asset_manifest.json`. Renderer: MuJoCo. Original Push-T control: this repository's frozen LeWM and CEM. Music and video composition: original procedural assets for this demonstration.
