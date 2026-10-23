import os
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import json,numpy as np,mujoco
from PIL import Image
P=Path(__file__).resolve().parent;OUT=P.parent
SCALE=.0009

def xy(p):return np.array([.4+(p[0]-256)*SCALE,-.08-(p[1]-256)*SCALE])

class Robot:
 def __init__(self,w=1920,h=1080):
  self.m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));self.d=mujoco.MjData(self.m)
  self.tip=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_SITE,'push_tip')
  self.renderer=mujoco.Renderer(self.m,height=h,width=w)
  self.cam=mujoco.MjvCamera();self.cam.lookat[:]=[.25,.0,1.02];self.cam.distance=1.9;self.cam.azimuth=120;self.cam.elevation=-27
  self.opt=mujoco.MjvOption();self.opt.geomgroup[3]=0;self.opt.sitegroup[:]=0
  self.jpos=np.zeros((3,self.m.nv));self.jrot=np.zeros_like(self.jpos)
  self.d.qpos[:]=[.7,-1.3,1.8,-2.0,-1.57,0]
 def ik(self,p,iterations=100):
  target=np.r_[xy(p),.77];R=np.diag([1.,-1.,-1.])
  for _ in range(iterations):
   mujoco.mj_forward(self.m,self.d)
   dr=target-self.d.site_xpos[self.tip];cur=self.d.site_xmat[self.tip].reshape(3,3)
   er=.5*sum(np.cross(cur[:,i],R[:,i]) for i in range(3))
   if np.linalg.norm(dr)<.00005 and np.linalg.norm(er)<.001:break
   mujoco.mj_jacSite(self.m,self.d,self.jpos,self.jrot,self.tip)
   J=np.vstack([self.jpos,self.jrot*.25]);e=np.r_[dr,er*.25]
   dq=J.T@np.linalg.solve(J@J.T+np.eye(6)*.0001,e)
   self.d.qpos[:]+=dq*min(1.,.15/max(1e-9,float(np.max(np.abs(dq)))))
  return self.d.qpos.copy(),float(np.linalg.norm(dr))
 def pose(self,name,p,z):
  body=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,name);i=self.m.body_mocapid[body]
  self.d.mocap_pos[i]=[*xy(p[:2]),z];self.d.mocap_quat[i]=[np.cos(p[2]/2),0,0,-np.sin(p[2]/2)]
 def frame(self,state,goal,q,crumb=None,az=120,elev=-27,dist=1.9,look=None):
  self.d.qpos[:]=q;self.pose('block',state[2:5],.772);self.pose('goal',goal[2:5],.753)
  if crumb is None:self.pose('crumb',[10000,10000,0],.756)
  else:self.pose('crumb',crumb[2:5],.756)
  mujoco.mj_forward(self.m,self.d)
  self.cam.azimuth=az;self.cam.elevation=elev;self.cam.distance=dist
  self.cam.lookat[:]=look or [.25,0,1.02]
  self.renderer.update_scene(self.d,camera=self.cam,scene_option=self.opt)
  return Image.fromarray(self.renderer.render())

if __name__=='__main__':
 r=Robot();runs=json.loads((OUT/'rollouts.json').read_text());qs={};errs={}
 for name,run in runs.items():
  r.d.qpos[:]=[.7,-1.3,1.8,-2,-1.57,0]
  q=[];err=[]
  for s in run['states']:
   a,e=r.ik(s[:2],250);q.append(a);err.append(e)
  qs[name]=np.asarray(q);errs[name]={'max_tip_error_m':max(err),'mean_tip_error_m':float(np.mean(err))}
  print(name,errs[name],flush=True)
  r.frame(run['states'][0],run['goal'],q[0]).save(OUT/(name+'_robot.png'))
 np.savez_compressed(OUT/'robot_q.npz',**qs)
 (OUT/'ik_checks.json').write_text(json.dumps(errs,indent=2)+'\n')
 r.renderer.close()
