"""Render actual LeWM Push-T captures using a UR5e kinematic embodiment.
No invented block trajectories. Captions distinguish recorded evaluation evidence.
"""
from robot import *
import math,sys,subprocess,wave
from PIL import ImageDraw,ImageFont,ImageFilter
W,H,FPS,DUR=1920,1080,24,116
RUNS=json.loads((OUT/'rollouts.json').read_text());QS=np.load(OUT/'robot_q.npz')
FONT='/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf';BOLD='/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf';MONO='/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf'
fonts={}
def f(s,b=False,mono=False):
 k=(s,b,mono)
 if k not in fonts:fonts[k]=ImageFont.truetype(MONO if mono else BOLD if b else FONT,s)
 return fonts[k]
INK='#f0f4f7';MUTE='#a6b5c4';TEAL='#64e2b4';GOLD='#ffc169';RED='#ff8d85';BLUE='#70cdeb'

def clamp(x):return max(0,min(1,x))
def ease(x):x=clamp(x);return x*x*(3-2*x)
def text(d,xy,s,size=28,color=INK,b=False,mono=False):d.text(xy,s.replace('→','>') if mono else s,font=f(size,b,mono),fill=color)
def panel(d,box):d.rounded_rectangle(box,radius=18,fill=(12,22,34,235),outline=(74,97,119,160),width=1)
# Every playback segment is a time-resampled recorded rollout (including angle wrap).
SCENES=[
(0,8,'target','LANGUAGE → ACTION','A goal is easy to say.', '“Fit the T into the green outline.”',0,1),
(8,14,'far11_flat','THE TASK','Contact changes everything.','Push from the wrong side and the T slides, spins—or stays put.',0,0),
(14,29,'far11_flat','01 / THE LIMIT','One distant goal. A short-sighted planner.','The frozen world model predicts nearby outcomes; this goal is 100 demo steps away.',0,1),
(29,34,'far11_crumbs','RESET / SAME START','Keep the planner. Change the waypoints.','Same checkpoint. Same start and goal. Same 300-step budget.',0,0),
(34,49,'far11_crumbs','02 / THE INTERVENTION','Give it a reachable next step.','Amber = a real expert-demo pose. Chase it, observe again, choose the next.',0,1),
(49,58,'far3_crumbs','ANOTHER START','The route can switch demonstrations.','The graph joins demos at nearby states; each hop stays close enough to plan.',0,1),
(58,66,'far11_crumbs','THE MEASURED GAIN','Does it help beyond these examples?','Saved offset-100 evaluation: 10/50 → 29/50 successes, at the same budget.',1,1),
(66,81,'target','03 / WORDS CHOOSE THE GOAL','Now ask for the task in words.','A local parser names the goal region. The graph supplies nearby demo frames.',0,1),
(81,98,'turn_cw','04 / WHERE IT BREAKS','“Turn it clockwise. Stay in place.”','Watch the center: the T rotates, but it also drifts away from where it started.',0,1),
(98,106,'turn_cw','FAILURE / NOT JUST AN ANGLE','The turn is there. The precision is not.','This fresh run: about 91° of rotation, but 42.5 px of drift. The command fails.',1,1),
(106,116,'target','WHAT THIS PROJECT CONTRIBUTES','One reachable step at a time.','Language names the goal. Demonstrations supply the steps. LeWM plans the pushes.',0,1),
]

def state_at(name,u):
 run=RUNS[name];ss=np.array(run['states']);j=clamp(u)*(len(ss)-1);a=int(j);b=min(a+1,len(ss)-1);v=j-a
 state=ss[a]*(1-v)+ss[b]*v
 da=(ss[b,4]-ss[a,4]+np.pi)%(2*np.pi)-np.pi;state[4]=ss[a,4]+da*v
 q=QS[name][a]*(1-v)+QS[name][b]*v
 target=None;hop=0
 for k,t in enumerate(run['targets']):
  if t['at_step']<=j:target=t['state'];hop=k+1
 return state,q,target,j,hop

def tee(d,p,box,color,outline=None,width=2):
 x,y,theta=p;bx,by,size=box;scale=size/512
 pts=[(-60,0),(60,0),(60,30),(15,30),(15,120),(-15,120),(-15,30),(-60,30)]
 c,s=math.cos(theta),math.sin(theta)
 points=[(bx+(x+c*a-s*b)*scale,by+(y+s*a+c*b)*scale) for a,b in pts]
 d.polygon(points,fill=color)
 if outline:d.line(points+[points[0]],fill=outline,width=width,joint='curve')

def topdown(d,name,state,target,j,x=1498,y=350,size=340,turn=False):
 run=RUNS[name];panel(d,(x-20,y-64,x+size+20,y+size+104))
 text(d,(x,y-45),'LIVE TASK VIEW',22,BLUE,mono=True)
 d.rectangle((x,y,x+size,y+size),fill='#dce4e9')
 for z in range(0,513,64):
  q=z*size/512;d.line((x+q,y,x+q,y+size),fill='#ccd7de');d.line((x,y+q,x+size,y+q),fill='#ccd7de')
 goal=run['goal'][2:5]
 if name=='target':goal=[256,256,np.pi/4]
 if turn:goal=list(run['states'][0][2:5]);goal[2]+=np.pi/2
 tee(d,goal,(x,y,size),'#82cdb0','#278e6e')
 if target is not None:tee(d,target[2:5],(x,y,size),'#e6c791','#ba852f')
 pts=[(x+s[2]*size/512,y+s[3]*size/512) for s in run['states'][:int(j)+1]]
 if len(pts)>1:d.line(pts,fill='#e28071' if turn else '#238b93',width=3)
 if turn:
  sx,sy=run['states'][0][2:4];cx=x+sx*size/512;cy=y+sy*size/512;rad=15*size/512
  d.ellipse((cx-rad,cy-rad,cx+rad,cy+rad),outline='#de6b5e',width=2)
 tee(d,state[2:5],(x,y,size),'#60768a','#243b50')
 px=x+state[0]*size/512;py=y+state[1]*size/512;rad=10*size/512
 d.ellipse((px-rad,py-rad,px+rad,py+rad),fill='#51bed6',outline='#246e88',width=2)
 text(d,(x,y+size+17),'GREEN  goal    BLUE  pusher',19,MUTE,mono=True)
 text(d,(x,y+size+44),'AMBER  next demo pose',19,GOLD,mono=True)

def frame(r,t):
 scene=next(s for s in SCENES if s[0]<=t<s[1]);a,b,name,kicker,title,caption,u0,u1=scene
 qtime=clamp((t-a)/(b-a-1.2));u=u0+(u1-u0)*qtime
 state,q,target,j,hop=state_at(name,u)
 goal=RUNS[name]['goal'].copy()
 if name=='target':goal[2:5]=[256,256,np.pi/4]
 if name=='turn_cw':goal[2:5]=list(RUNS[name]['states'][0][2:4])+[RUNS[name]['states'][0][4]+np.pi/2]
 az=122+2*math.sin(t*.09);dist=1.75;look=[.28,-.01,1.03]
 close=max([ease((t-start)/1.5)*ease((end-t)/1.5) if start<t<end else 0 for start,end in [(18,27),(37,48),(69,80),(84,97)]])
 dist-=.55*close;look=[.28+.08*close,-.01-.04*close,1.03-.14*close];az+=10*close
 im=r.frame(state,goal,q,target if 'crumbs' in name or name=='target' else None,az=az,elev=-27,dist=dist,look=look).convert('RGBA')
 ov=Image.new('RGBA',(W,H));d=ImageDraw.Draw(ov)
 # Quiet gradients leave the robot and the contact in view.
 for yy in range(240):d.line((0,yy,W,yy),fill=(7,13,23,int(245*(1-yy/240)**.65)))
 for yy in range(875,H):d.line((0,yy,W,yy),fill=(7,13,23,int(240*((yy-875)/(H-875))**.5)))
 text(d,(65,34),'JEPA  /  FROM WORDS TO CONTACT',23,TEAL,True,True)
 text(d,(65,91),title,57,INK,True)
 text(d,(67,164),kicker,22,GOLD,mono=True)
 text(d,(1338,35),'RTX 5090  ·  FROZEN LeWM',22,MUTE,mono=True)
 if t<8:
  text(d,(67,211),'Real Push-T rollout · UR5e robot visualization',27,MUTE)
  panel(d,(65,300,555,443));text(d,(92,323),'THE CHALLENGE',23,BLUE,mono=True)
  text(d,(92,365),'Move it. Rotate it. Align it.',31)
 if 8<=t<14:
  panel(d,(65,290,580,505));text(d,(94,315),'WHY A WORLD MODEL?',23,BLUE,mono=True)
  text(d,(94,363),'Imagine candidate pushes.',29)
  text(d,(94,407),'Pick one. Touch. Observe.',29)
  text(d,(94,452),'Repeat from the new state.',29)
 topdown(d,name,state,target if name!='far11_flat' else None,j,turn=name=='turn_cw')
 # Timeline of the real rollout, not a decorative success meter.
 panel(d,(1490,810,1860,911));text(d,(1510,829),f'ENV STEP  {int(j):03d} / {len(RUNS[name]["states"])-1:03d}',23,mono=True)
 d.rounded_rectangle((1510,879,1840,885),radius=3,fill='#2c3e4f');d.rounded_rectangle((1510,879,1510+max(3,330*u),885),radius=3,fill=TEAL if 'flat' not in name else RED)
 if 14<=t<29:
  panel(d,(65,293,580,466));text(d,(94,315),'FINAL IMAGE ONLY',23,RED,mono=True)
  text(d,(94,358),'One far destination.',32)
  text(d,(94,405),'No intermediate guidance.',27,MUTE)
  if u>.98:text(d,(67,550),'300 STEPS · NOT REACHED',29,RED,True)
 if 29<=t<49:
  panel(d,(65,293,580,474));text(d,(94,315),'DEMO BREADCRUMBS',23,GOLD,mono=True)
  text(d,(94,358),f'Hop {max(1,hop)}   /   next nearby pose',30)
  text(d,(94,410),'Plan ~25 steps. Observe again.',26,MUTE)
  if u>.98 and t>34:text(d,(67,550),'73 STEPS · GOAL REACHED',29,TEAL,True)
 if 49<=t<58:
  panel(d,(65,293,580,460));text(d,(94,315),'SECOND CAPTURE',23,BLUE,mono=True)
  text(d,(94,363),'Another held-out demo start.',27)
  text(d,(94,409),'99 steps to the goal.',30,TEAL,True)
 if 58<=t<66:
  panel(d,(65,285,640,700));text(d,(94,310),'SAVED EVALUATION · 50 TASKS',22,BLUE,mono=True)
  for yy,label,value,col in [(370,'Final goal only',10,RED),(500,'Breadcrumbs',29,TEAL)]:
   text(d,(94,yy),label,29);text(d,(490,yy),f'{value}/50',36,col,True)
   d.rounded_rectangle((94,yy+55,590,yy+77),radius=9,fill='#293c4b');d.rounded_rectangle((94,yy+55,94+496*value/50,yy+77),radius=9,fill=col)
  text(d,(94,625),'+38 percentage points',34,TEAL,True)
 if 66<=t<81:
  panel(d,(65,285,600,532));text(d,(94,308),'“Fit the T into the green outline.”',29)
  text(d,(94,365),'WORDS  >  goal region',27,BLUE,mono=True)
  text(d,(94,414),'GRAPH  >  next demo frame',27,GOLD,mono=True)
  text(d,(94,463),'LeWM   >  best push sequence',27,TEAL,mono=True)
  if t>76:text(d,(67,590),'125 STEPS · TARGET REACHED',28,TEAL,True)
 if 81<=t<106:
  st=RUNS[name]['states'][0];drift=float(np.linalg.norm(state[2:4]-np.array(st[2:4])))
  angle=math.degrees((state[4]-st[4]+np.pi)%(2*np.pi)-np.pi)
  panel(d,(65,285,580,560));text(d,(94,310),'TURN REQUEST  +90°',23,BLUE,mono=True)
  text(d,(94,363),f'{angle:+.1f}°',56,INK,True);text(d,(94,430),f'{drift:.1f} px drift',38,RED,True)
  text(d,(94,493),'Strict in-place limit: <15 px',25,MUTE)
  if 98<=t:
   panel(d,(65,592,580,737));text(d,(94,613),'SAVED TURN EVALUATION',22,BLUE,mono=True)
   text(d,(94,657),'8/20 loose · 0/20 strict',30,RED,True)
 if t>=106:
  panel(d,(65,280,660,581));text(d,(94,306),'WHAT WORKED',23,TEAL,mono=True)
  text(d,(94,351),'Nearby demo goals unlock longer tasks.',29)
  text(d,(94,403),'WHAT REMAINS',23,RED,mono=True)
  text(d,(94,445),'Precise turns. Unfamiliar goals.',29)
  text(d,(94,491),'Real-arm contact and transfer.',29)
  text(d,(67,630),'Next: teleoperated demos on the lab UR3.',27,MUTE)
 # Captions stay short enough to read while watching the motion.
 text(d,(67,934),caption,29,INK)
 footer='3D replay: recorded LeWM Push-T motion · UR5e follows by IK; arm dynamics/transfer not evaluated'
 text(d,(67,994),footer,20,MUTE)
 text(d,(67,1032),'Bashirov · Seyidov · Ogunwoye  /  Skoltech',20,MUTE)
 text(d,(1575,1032),f'{int(t)//60:02d}:{int(t)%60:02d}  /  01:56',21,MUTE,mono=True)
 d.rectangle((0,H-4,W*t/DUR,H),fill=TEAL)
 # Brief fades between experiments, while the current scene remains recognizable.
 local=t-a;fade=min(1,local/.32,(b-t)/.25)
 if fade<1:d.rectangle((0,0,W,H),fill=(6,11,19,int(255*(1-fade))))
 return Image.alpha_composite(im,ov).convert('RGB')

def soundtrack():
 sr=22050;t=np.arange(sr*DUR)/sr;audio=np.zeros_like(t)
 for k,hz in enumerate([130.8128,164.8138,195.9977,261.6256]):
  amp=.013*(.7+.3*np.sin(2*np.pi*.065*t+k))
  audio+=amp*np.sin(2*np.pi*hz*t+.08*np.sin(2*np.pi*.2*t))
 for start in [0,8,14,29,34,49,58,66,81,98,106]:
  x=t-start;mask=(x>=0)&(x<2)
  audio[mask]+=.025*np.exp(-3*x[mask])*np.sin(2*np.pi*(523.251 if start not in [14,81,98] else 391.995)*x[mask])
 audio*=np.minimum(1,t/2)*np.minimum(1,(DUR-t)/3)
 with wave.open(str(OUT/'soundtrack.wav'),'wb') as w:w.setparams((1,2,sr,0,'NONE','not compressed'));w.writeframes((audio*32767).astype('<i2').tobytes())

if __name__=='__main__':
 r=Robot()
 if len(sys.argv)>1 and sys.argv[1]=='stills':
  for t in [3,11,21,32,41,54,62,73,90,102,111]:frame(r,t).save(OUT/f'preview_{t:03}.jpg',quality=94)
 else:
  soundtrack()
  cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-i',str(OUT/'soundtrack.wav'),'-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-movflags','+faststart','-shortest',str(OUT/'JEPA_robot_demo.mp4')]
  proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
  for i in range(DUR*FPS):
   im=frame(r,i/FPS);proc.stdin.write(im.tobytes())
   if i%240==0:print(f'rendered {i/FPS:.0f}/{DUR}s',flush=True)
  proc.stdin.close();assert proc.wait()==0
 r.renderer.close()
