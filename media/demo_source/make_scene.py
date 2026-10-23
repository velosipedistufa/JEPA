from pathlib import Path
import xml.etree.ElementTree as E
P=Path(__file__).resolve().parent
r=E.parse(P/'ur5e.xml').getroot()
r.find('worldbody/light').set('diffuse','.5 .5 .5')
r.find('worldbody/body').set('pos','-.10 .38 .75')
r.remove(r.find('keyframe'))
E.SubElement(r,'visual')
v=r.find('visual');E.SubElement(v,'global',offwidth='1920',offheight='1080');E.SubElement(v,'quality',shadowsize='4096',offsamples='4');E.SubElement(v,'headlight',diffuse='.45 .48 .55',ambient='.30 .30 .33',specular='.15 .15 .15');E.SubElement(v,'map',znear='.01',zfar='30',shadowscale='1.5')
a=r.find('asset')
E.SubElement(a,'texture',name='floor_tex',type='2d',builtin='checker',rgb1='.055 .071 .09',rgb2='.065 .083 .103',width='512',height='512')
E.SubElement(a,'material',name='floor_mat',texture='floor_tex',texrepeat='6 6',reflectance='.15')
w=r.find('worldbody')
E.SubElement(w,'geom',name='floor',type='plane',size='5 5 .01',material='floor_mat')
E.SubElement(w,'light',name='key',pos='1 -1 3',dir='-.2 .2 -1',diffuse='.85 .88 1',castshadow='true')
E.SubElement(w,'light',name='rim',pos='-1 1 2',dir='.2 -.2 -1',diffuse='.4 .65 .8',castshadow='false')
E.SubElement(w,'geom',type='box',pos='.25 .02 .71',size='.66 .62 .04',rgba='.16 .19 .23 1')
E.SubElement(w,'geom',type='box',pos='.40 -.08 .751',size='.232 .232 .001',rgba='.72 .75 .78 1')
for x in [-.28,.78]:
 for y in [-.43,.48]:E.SubElement(w,'geom',type='box',pos=f'{x} {y} .34',size='.035 .035 .34',rgba='.12 .15 .19 1')
# Workspace border and subtle metrology marks.
for x in [-.232,.232]:E.SubElement(w,'geom',type='box',pos=f'{.4+x} -.08 .753',size='.001 .232 .0005',rgba='.33 .40 .46 1')
for y in [-.232,.232]:E.SubElement(w,'geom',type='box',pos=f'.4 {-.08+y} .753',size='.232 .001 .0005',rgba='.33 .40 .46 1')
for name,color,z in [('goal','0.12 0.80 0.55 .65',.753),('crumb','1 .66 .15 .32',.756),('block','.72 .75 .80 1',.772)]:
 b=E.SubElement(w,'body',name=name,mocap='true',pos=f'.4 -.08 {z}')
 h=.018 if name=='block' else .0007
 E.SubElement(b,'geom',type='box',pos='0 -.0135 0',size=f'.054 .0135 {h}',rgba=color,contype='0',conaffinity='0')
 E.SubElement(b,'geom',type='box',pos='0 -.0675 0',size=f'.0135 .0405 {h}',rgba=color,contype='0',conaffinity='0')
b=r.find(".//body[@name='wrist_3_link']")
E.SubElement(b,'geom',type='capsule',fromto='0 .10 0 0 .22 0',size='.009',rgba='.8 .85 .89 1',contype='0',conaffinity='0')
E.SubElement(b,'geom',type='sphere',pos='0 .22 0',size='.0135',rgba='.13 .7 .85 1',contype='0',conaffinity='0')
E.SubElement(b,'site',name='push_tip',pos='0 .22 0',quat='-1 1 0 0',size='.002',rgba='0 0 0 0')
E.indent(r);E.ElementTree(r).write(P/'scene.xml')
