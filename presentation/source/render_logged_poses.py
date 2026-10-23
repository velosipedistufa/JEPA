"""Reconstruct saved Push-T poses for slides; no model or planner is run."""
import os,json
os.environ['SDL_VIDEODRIVER']='dummy';os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
from pathlib import Path
import numpy as np
from PIL import Image
import stable_worldmodel
import gymnasium as gym
out=Path(__file__).resolve().parent/'assets'
d=json.loads((out.parents[2]/'JEPA/research/artifacts/5090/words_crumbs_b300_n64.json').read_text())
r=next(x for x in d['episodes'] if x['seed']==1 and x['command']=='turn_cw');st=r['start_state']
env=gym.make('swm/PushT-v1',render_mode='rgb_array')
for name,b,show_agent in [('pusht_logged_start.png',st[2:5],True),('turn_seed1_start.png',st[2:5],False),('turn_seed1_end.png',r['end_block'],False)]:
 env.reset(seed=1);u=env.unwrapped;u.render_size=512
 if not show_agent:u.goal_pose=np.array([st[2],st[3],st[4]+np.pi/2])
 u.agent.position=tuple(st[:2]) if show_agent else (-100,-100);u.agent.velocity=(0,0)
 u.block.angle=b[2];u.block.position=tuple(b[:2]);u.block.velocity=(0,0);u.block.angular_velocity=0
 for body in (u.agent,u.block):
  u.space.reindex_shapes_for_body(body)
  for sh in body.shapes:sh.cache_bb()
 assert np.allclose(u._get_obs()[2:5],b)
 a=np.asarray(env.render());a=a.transpose(1,2,0) if a.shape[0] in(3,4) else a
 Image.fromarray(a).save(out/name)
env.close()
