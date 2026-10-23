<p align="center">
  <img src="https://raw.githubusercontent.com/velosipedistufa/JEPA/master/media/JEPA_robot_demo_poster.jpg" alt="UR5e arm following a frozen LeWM Push-T trajectory, with the next demonstration subgoal and a live overhead task view" width="1000" />
</p>

<h1 align="center">Language over Demo Breadcrumbs</h1>

<p align="center"><strong>Words name the goal. Demonstrations supply the steps.</strong></p>

<p align="center">
  <a href="https://raw.githubusercontent.com/velosipedistufa/JEPA/master/media/JEPA_robot_demo.mp4"><strong>Watch the 60-second demo ↗</strong></a>
  &nbsp;·&nbsp;
  <a href="presentation/JEPA_defence_clean.pdf">Slides · PDF</a>
  &nbsp;·&nbsp;
  <a href="presentation/JEPA_defence_clean.pptx">PowerPoint</a>
  &nbsp;·&nbsp;
  <a href="media/README.md">Demo evidence</a>
</p>

<p align="center">Alexander Bashirov · Didar Seyidov · Faith Ogunwoye<br />Skoltech</p>

<p align="center"><em>Teaser: actual frozen-LeWM Push-T motion visualized with a UR5e arm.</em></p>

A frozen world model can plan useful short pushes. We give it nearby goals from expert demonstrations so it can tackle longer tasks, and let language choose the goal region.

## What worked—and what remains

| Saved experiment | Result |
|---|---|
| Demo goal 100 steps ahead, matched budget | Flat 10/50 → breadcrumbs 29/50 |
| Held-out sentence parsing | 101/120 correct (84.2%) |
| Language tasks, same 111 eligible cases | Gold command 57/111; parsed command 51/111 |
| Turn templates | 8/20 with <40 px drift; 0/20 with strict <15 px drift |
| Learned text-to-subgoal head | 18/111; shuffled text 10/111 |

The encoder and world model stay frozen. The demo graph uses simulator state for retrieval. Real-robot transfer remains future work: the video uses recorded Push-T physics and a UR5e kinematic visualization, rather than an evaluated robot-arm controller.

## Course archive

- [Experiment handoff and frozen claims](HANDOFF_5090.md)
- [Saved evaluation artifacts](research/artifacts/5090/)
- [Planner and language scripts](scripts/)
- [Reports](reports/)
- [Original LeWM source](le-wm/)
- [Setup notes](SETUP.md)
- [Presentation source and provenance](presentation/source/)

`media/` retains the new visualization captures, renderer source, robot-model attribution, and reproduction instructions. The final course snapshot also preserves the local run logs and trained subgoal-head artifacts. Downloadable upstream datasets, foundation-model caches and virtual environments are not bundled.
