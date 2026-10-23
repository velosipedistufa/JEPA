"""60-second project demonstration using complete, time-resampled recorded runs."""
import render as full
from robot import *
from PIL import ImageDraw
import subprocess,sys
DURATION=60
CUTS=[(0,6,0,8),(6,9,8,14),(9,19,14,29),(19,22,29,34),(22,34,34,49),(34,40,58,66),(40,49,81,98),(49,54,98,106),(54,60,106,116)]

def frame(r,t):
 a,b,x,y=next(c for c in CUTS if c[0]<=t<c[1])
 old=x+(y-x)*(t-a)/(b-a)
 im=full.frame(r,old);d=ImageDraw.Draw(im)
 # Replace the original clock and progress line with the one-minute timeline.
 d.rectangle((1510,1025,1919,1074),fill='#080e17')
 full.text(d,(1575,1032),f'{int(t)//60:02d}:{int(t)%60:02d}  /  01:00',21,full.MUTE,mono=True)
 d.rectangle((0,1076,1919,1079),fill='#080e17');d.rectangle((0,1076,1920*t/DURATION,1079),fill=full.TEAL)
 return im

if __name__=='__main__':
 r=full.Robot()
 if len(sys.argv)>1 and sys.argv[1]=='stills':
  for t in [3,14,28,37,45,52,57]:frame(r,t).save(OUT/f'short_preview_{t:03}.jpg',quality=95)
 else:
  full.DUR=DURATION;full.soundtrack()
  cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','24','-i','-','-i',str(OUT/'soundtrack.wav'),'-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-movflags','+faststart','-shortest',str(OUT/'JEPA_robot_demo_60s.mp4')]
  p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
  for i in range(DURATION*24):
   p.stdin.write(frame(r,i/24).tobytes())
   if i%240==0:print(f'rendered {i/24:.0f}/{DURATION}s',flush=True)
  p.stdin.close();assert p.wait()==0
 r.renderer.close()
