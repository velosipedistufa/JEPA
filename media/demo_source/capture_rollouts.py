"""Fresh visualization captures; does not modify frozen evaluations or their artifacts."""
import sys,json,os
from pathlib import Path
import numpy as np
ROOT=Path(os.environ.get('JEPA_ROOT',str(Path(__file__).resolve().parents[2]))); OUT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
import eval_official_cem as official
import demo_breadcrumbs as d
import word_breadcrumbs as w
import torch
from PIL import Image
np.random.seed(42);torch.manual_seed(42)
ds=official.open_dataset(); mean,std,_=official.action_stats(ds)
states=np.asarray(ds.get_col_data('state'),dtype=np.float64)
lengths,offsets=np.asarray(ds.lengths),np.asarray(ds.offsets)
graph=d.DemoGraph(states,offsets,lengths,d.split_episodes(len(lengths))[0])
model=official.load_model(); transform=official.make_transform()
env,ce=official.make_env(),official.make_env()
all_runs={}; original_step=env.step; original_walk=graph.walk
track=None

def step(a):
 result=original_step(a)
 if track is not None:
  track['actions'].append(np.asarray(a).tolist());track['states'].append(np.asarray(env.unwrapped._get_obs()).tolist())
 return result

def walk(node,plan,max_steps):
 result=original_walk(node,plan,max_steps)
 if track is not None and max_steps==25:
  track['targets'].append({'at_step':len(track['actions']),'state':graph.state[result[0]].tolist(),'final':bool(result[1])})
 return result
env.step=step;graph.walk=walk

def begin(name,start,goal):
 global track
 track={'states':[start.tolist()],'actions':[],'targets':[],'goal':goal.tolist(),'note':'Fresh 5090 capture for explanatory video, separate from frozen scored artifacts.'}
 all_runs[name]=track

def finish(name,result):
 track['result']=result
 Image.fromarray(official.hwc(np.asarray(env.render()))).save(OUT/(name+'_end.png'))
 (OUT/'rollouts.json').write_text(json.dumps(all_runs,indent=2)+'\n')
 print(name,json.dumps(result),flush=True)

try:
 baseline=json.loads((ROOT/'research/artifacts/5090/dataset100_crumbs_b300_n50.json').read_text())['episodes']
 for case in [3,11]:
  r=baseline[case];ep,t=r['episode'],r['start']; start=states[int(offsets[ep])+t];goal=states[int(offsets[ep])+t+100]
  chunk=ds.load_chunk(np.array([ep]),np.array([t+100]),np.array([t+101]))[0]
  frame=official.as_numpy(chunk['pixels'][-1])
  for mode in ['flat','crumbs']:
   torch.manual_seed(42);env.reset(seed=0);env.unwrapped._set_state(start)
   name=f'far{case}_{mode}';begin(name,start,goal)
   result=d.run_episode(env,ce,model,transform,mean,std,goal,frame,300,graph if mode=='crumbs' else None)
   track['episode']=ep;track['demo_start']=t;finish(name,result)
 for seed,cmd in [(0,'target'),(1,'turn_cw')]:
  torch.manual_seed(42);env.reset(seed=seed);start=np.asarray(env.unwrapped._get_obs()).copy()
  nodes=np.nonzero(w.in_region(cmd,start,graph.state))[0];plan=w.plan_to_nodes(graph,nodes)
  goal_node,_=w.goal_image_node(graph,plan,nodes,start)
  goal=graph.state[goal_node]
  name=cmd;begin(name,start,goal);track['command']=cmd
  result=w.run_episode(env,ce,model,transform,mean,std,cmd,start,graph,plan,goal_node,300,'crumbs')
  finish(name,result)
finally:env.close();ce.close()
