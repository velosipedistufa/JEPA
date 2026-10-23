"""Build the editable JEPA defence using saved results only. No experiments are run.
PYTHONPATH=/tmp/jepa_pptx_deps JEPA/.venv/bin/python deliverables/jepa_defence/build_defence.py
"""
from pathlib import Path
import json, math, hashlib
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.chart.data import CategoryChartData
from pptx.oxml.xmlchemy import OxmlElement
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'research/artifacts/5090'
AS=OUT/'assets';AS.mkdir(exist_ok=True)
C={'bg':'F7F6F2','navy':'142A3B','teal':'067F7C','mint':'D9EEEA','blue':'3567A0','orange':'B96643','purple':'77629D','muted':'61717B','gray':'AFB9C0','line':'DDE3E4','white':'FFFFFF','pale':'E9EDF0','red':'B24C43','panel':'1E3A4F'}
FONT='Arial'
prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
prs.core_properties.title='Language over demo breadcrumbs — Frozen LeWorldModel on Push-T'
prs.core_properties.author='Alexander Bashirov; Didar Seyidov; Faith Ogunwoye'
prs.core_properties.subject='Visual defence deck rebuilt from the lead-authored PDF and saved JEPA run artifacts'
records={}
def load(name):
 p=R/name;d=json.loads(p.read_text());records[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return d
D={k:load(v) for k,v in {
 'setup':'check0_official_cem_n50.json','d50f':'dataset50_flat_b300_n50.json','d50c':'dataset50_crumbs_b300_n50.json',
 'd100f':'dataset100_flat_b300_n50.json','d100c':'dataset100_crumbs_b300_n50.json','livef':'live_flat_b300_n20.json','livec':'live_crumbs_b300_n20.json',
 'wf':'words_flat_b300_n64.json','wc':'words_crumbs_b300_n64.json','parse':'parse_paraphrases_n120.json','language':'language_paraphrases_b300_n111.json',
 'head':'latent_subgoal_paraphrases_b300_n111.json','shuffle':'latent_subgoal_paraphrases_shuffled_b300_n111.json','train':'latent_subgoal_train.json',
 'dry':'dataset50_dry_n50.json','livedry':'live_dry_n20.json','wdry':'words_dry_n64.json'}.items()}
for key in ('d50f','d50c','d100f','d100c','livef','livec','wf','wc'):
 assert sum(bool(r['success']) for r in D[key]['episodes'])==D[key]['summary']['successes']
assert sum(r['llm_ok'] for r in D['parse']['rows'])==101
assert sum(r['keyword_ok'] for r in D['parse']['rows'])==31
assert len(D['language']['rows'])==111
assert sum(r['gold_run']['success'] for r in D['language']['rows'])==57
assert sum(r['parsed_run']['success'] for r in D['language']['rows'])==51
assert sum(r['success'] for r in D['head']['rows'])==18
assert sum(r['success'] for r in D['shuffle']['rows'])==10
paired={}
for pre in ('d50','d100','live','w'):
 a,b=D[pre+'f']['episodes'],D[pre+'c']['episodes'];assert len(a)==len(b)
 ident=('seed','command') if pre=='w' else ('seed',) if pre=='live' else ('episode','start')
 assert [tuple(x[k] for k in ident) for x in a]==[tuple(x[k] for k in ident) for x in b]
 paired[pre]={'both':sum(x['success'] and y['success'] for x,y in zip(a,b)), 'flat_only':sum(x['success'] and not y['success'] for x,y in zip(a,b)), 'crumbs_only':sum(not x['success'] and y['success'] for x,y in zip(a,b)), 'neither':sum(not x['success'] and not y['success'] for x,y in zip(a,b))}

def col(v):return RGBColor.from_string(C.get(v,v))
def shape(sl,x,y,w,h,fill='white',line=None,radius=False,kind=None):
 sp=sl.shapes.add_shape(kind or (MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE),Inches(x),Inches(y),Inches(w),Inches(h));sp.fill.solid();sp.fill.fore_color.rgb=col(fill)
 if line:sp.line.color.rgb=col(line);sp.line.width=Pt(1)
 else:sp.line.fill.background()
 if radius:
  try:sp.adjustments[0]=.13
  except:pass
 return sp

def text(sl,x,y,w,h,s,size=22,color='navy',bold=False,align=None):
 sp=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sp.text_frame;tf.clear();tf.word_wrap=True
 tf.margin_left=tf.margin_right=0;tf.margin_top=Pt(1);tf.margin_bottom=0
 for i,line in enumerate(str(s).split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name=FONT;p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=col(color);p.space_after=Pt(6)
  if align is not None:p.alignment=align
 return sp

def ln(sl,x1,y1,x2,y2,color='line',width=1.5,dash=False,arrow=False):
 sp=sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2));sp.line.color.rgb=col(color);sp.line.width=Pt(width)
 if dash:
  from pptx.enum.dml import MSO_LINE_DASH_STYLE
  sp.line.dash_style=MSO_LINE_DASH_STYLE.DASH
 if arrow:
  end=OxmlElement('a:tailEnd');end.set('type','triangle');end.set('w','med');end.set('len','med');sp.line._get_or_add_ln().append(end)
 return sp

def circle(sl,x,y,d,fill='teal',label=None):
 sp=shape(sl,x,y,d,d,fill,kind=MSO_SHAPE.OVAL)
 if label:text(sl,x,y+.04,d,d-.03,label,16,'white',True,PP_ALIGN.CENTER)
 return sp

def note(sl,body,sources='',original=None):
 out=body+'\n\nSOURCES (existing runs; no new experiment):\n'+sources
 if original:out+='\n\nOriginal PDF slide: '+str(original)
 sl.notes_slide.notes_text_frame.text=out

def base(kicker,title,sub=None,dark=False):
 sl=prs.slides.add_slide(prs.slide_layouts[6]);sl.background.fill.solid();sl.background.fill.fore_color.rgb=col('navy' if dark else 'bg')
 text(sl,.55,.30,11.9,.3,kicker.upper(),11,'mint' if dark else 'teal',True)
 text(sl,.55,.82,12.1,1.0,title,30,'white' if dark else 'navy',True)
 if sub:text(sl,.57,1.72,12.0,.54,sub,16,'mint' if dark else 'muted')
 ln(sl,.55,7.05,12.75,7.05,'muted' if dark else 'line',.7)
 text(sl,.57,7.13,10,.2,'FROZEN LeWM  /  PUSH-T  /  SKOLTECH',9,'mint' if dark else 'muted')
 text(sl,12.0,7.11,.75,.23,f'{len(prs.slides):02d}',10,'mint' if dark else 'muted',False,PP_ALIGN.RIGHT)
 tr=OxmlElement('p:transition');tr.set('spd','med');tr.append(OxmlElement('p:fade'));sl._element.append(tr)
 return sl

def pill(sl,x,y,w,s,color='teal'):
 shape(sl,x,y,w,.35,'mint' if color=='teal' else 'pale',radius=True);text(sl,x+.12,y+.03,w-.24,.25,s,11,color,True)

def block(sl,x,y,w,h,label,detail='',color='teal'):
 shape(sl,x,y,w,h,'white','line',True)
 text(sl,x+.18,y+.16,w-.36,.6,label,21,color,True)
 if detail:text(sl,x+.18,y+.83,w-.36,h-.91,detail,16,'muted')

def bar(sl,x,y,w,val,den,color,label=None):
 shape(sl,x,y,w,.22,'pale',radius=True)
 if val:shape(sl,x,y,w*val/den,.22,color,radius=True)
 if label:text(sl,x+w+.15,y-.09,1.3,.4,label,17,color,True)

def small_T(sl,x,y,scale=1,color='teal',rotation=0):
 # Native freeform; conceptual icon, not a measured trajectory.
 pts=[(-.6,-.45),(.6,-.45),(.6,-.17),(.15,-.17),(.15,.55),(-.15,.55),(-.15,-.17),(-.6,-.17)]
 a=math.radians(rotation);p=[(x+scale*(u*math.cos(a)-v*math.sin(a)),y+scale*(u*math.sin(a)+v*math.cos(a))) for u,v in pts]
 fb=sl.shapes.build_freeform(Inches(p[0][0]),Inches(p[0][1]));fb.add_line_segments([(Inches(u),Inches(v)) for u,v in p[1:]],close=True);sp=fb.convert_to_shape();sp.fill.solid();sp.fill.fore_color.rgb=col(color);sp.line.fill.background();return sp

# 1 — Lead's title, visual opening.
s=base('Course defence · 23 October 2026','Language over demo breadcrumbs',dark=True)
text(s,.6,1.68,10.8,.6,'Frozen LeWorldModel on Push-T',27,'mint')
text(s,.6,2.72,7.7,1.1,'Words name the goal.\nDemonstrations supply the steps.',31,'white',True)
text(s,.61,4.25,7.0,.62,'A language interface for a planner\nthat is reliable over short hops.',21,'mint')
text(s,.61,5.8,8.2,.6,'Alexander Bashirov   ·   Didar Seyidov   ·   Faith Ogunwoye',17,'white')
text(s,.61,6.47,10,.35,'Skoltech',15,'mint')
s.shapes.add_picture(str(AS/'pusht_logged_start.png'),Inches(9.0),Inches(2.22),width=Inches(3.35))
text(s,9.0,5.66,3.4,.4,'Push-T · logged start state',12,'mint',False,PP_ALIGN.CENTER)
note(s,'Opening: the lead’s narrative is retained. A frozen image-based world model is given nearby demonstration subgoals; language identifies the goal region. Cover image renders a logged start state from seed 1 of the clockwise turn task; it is not an experimental rollout animation.','defense_presentation.pdf; HANDOFF_5090.md',1)

# 2 — what this project adds (new in v2).
s=base('00 · What is new here','Four things this project adds to a frozen world model.','LeWM and its planner are used unchanged. Everything new sits around them.')
for i,(num,title,body,c) in enumerate([
 ('01','Words name the goal','A sentence becomes a goal region (on target, turn by θ, push by d) instead of one goal image. A small local LLM parses it: 101/120 unseen sentences.','teal'),
 ('02','Routes across many demos','A graph over 427,396 demo states links different demos where they pass close by, so a route exists from starts no demo began at.','blue'),
 ('03','Short hops reach far goals','The planner only ever chases a demo frame about 25 steps away. Goals 100 steps out: 10/50 → 29/50.','orange'),
 ('04','Words straight into the latent','A text-conditioned head predicts the next subgoal latent, with no graph at test time. First signal: 18/111 vs 10/111 with shuffled words.','purple')]):
 x=.6+(i%2)*6.15;y=2.42+(i//2)*2.05
 shape(s,x,y,5.95,1.85,'white','line',True);circle(s,x+.24,y+.26,.52,c,num)
 text(s,x+.98,y+.22,4.75,.45,title,20,'navy',True);text(s,x+.98,y+.74,4.75,1.0,body,14,'muted')
text(s,.65,6.48,12,.32,'Prior work (Hi-LeWM) used later frames of the same demo as subgoals. Every claim here was frozen in git before its scored run.',12,'muted')
note(s,'Say this slide as the promise of the talk: four additions, each shown later with numbers. (1) Language: commands become goal regions, not a single goal picture; Qwen2.5-1.5B parses 101/120 held-out sentences written by a different model. (2) Cross-demo routes: Hi-LeWM (Caselli et al. 2026) showed future frames of the SAME demo work as subgoals, which needs the demo the start came from; our graph links DIFFERENT demos, so new starts get routes too. (3) Short hops: the planner is strong at about 25 steps (47/50), so we only ever ask for that; offset-100 rises 10/50 to 29/50. (4) Words into the latent: a first test of feeding the sentence into LeWM latent space directly; it works mainly for target requests. Protocol: claims and kill numbers were committed before each scored run; test sentences came from Codex, not from the parser author.','HANDOFF_5090.md; research/artifacts/5090/*.json; scripts/demo_breadcrumbs.py; scripts/command_parser.py; scripts/latent_subgoal.py')

# 2 — problem and definitions.
s=base('01 · The problem','A short hop works. A distant goal is harder.','LeWM stays frozen: it encodes an image into 192 values and predicts latents under actions.')
for x,label,num,detail,color in [( .6,'SHORT DEMO GOAL','47/50','25 steps ahead · budget 50\n5090 setup check','teal'),(4.87,'LONG LIVE GOAL','0/20','Random live goals · budget 50\nEarlier 4090 run','orange'),(9.14,'IN-PLACE TURN','0/20','Rendered +90° or +180° goal\nBudget 50 · each angle','purple')]:
 shape(s,x,2.48,3.6,2.54,'white','line',True);text(s,x+.2,2.67,3.2,.36,label,13,color,True);text(s,x+.2,3.2,3.2,.82,num,49,color,True);text(s,x+.2,4.18,3.2,.62,detail,16,'muted')
text(s,.65,5.28,12.0,.7,'Course question: can a sentence name a goal\nthis existing planner can actually follow?',26,'navy',True)
text(s,.65,6.46,12,.31,'These are different goal settings, not a controlled horizon sweep. Earlier 4090 short-goal baseline: 45/50.',12,'muted')
note(s,'Explain what a latent is in ordinary language: a compact image representation. The three cards use different tasks/settings and must not be presented as an otherwise matched causal comparison. Short goal 47/50 is the 5090 setup check, 45/50 on the earlier 4090. Long live goal is 0/20 at budget 50. Oracle rendered in-place goals are 0/20 per angle at budget 50; +180 also failed at budget 100.','research/artifacts/5090/check0_official_cem_n50.json; research/artifacts/011/live_goal.json; reports/gpu_4090/logs/oracle_rotate.json',2)

# 3 — diagnostics.
s=base('02 · What we tried first','The angle is encoded. Control is still the obstacle.','Reading a state from a latent does not guarantee the planner can reach that state.')
block(s,.6,2.48,3.82,3.43,'1  Read the angle','Held-out circular error\n\n180° frames: 12.08°\nNear 0°: 14.58°','teal')
block(s,4.75,2.48,3.82,3.43,'2  Add a fixed Δz','One constant latent step\nper command fails to fit.\n\nResidual ratio: 0.991','orange')
block(s,8.9,2.48,3.82,3.43,'3  Plan the turn','Angle-cost +90°: 2/20\nStrict in-place gate: 0/20\n\nThree rendered stages: 1/20','purple')
shape(s,.61,6.2,12.1,.53,'navy',radius=True);text(s,.82,6.29,11.7,.32,'Decision: keep LeWM frozen and change the subgoal interface.',19,'white',True)
note(s,'The ridge angle readout uses the frozen encoder. Rot180 n=20; near-zero slice n=19. Residual ratio is training RMSE / target RMS, not an execution-success rate. The two angle-cost hits drifted over 100 px and are not in-place successes. Three rendered +30/+60/+90 stages passed the strict gate in 1/20. The conclusion is bounded to these planner settings.','data/probes/probe_metrics.json; data/probes/language_port_fit.json; reports/gpu_4090/logs/probe_cost_official.json; research/artifacts/012/staged.json',3)

# 4 — method.
s=base('03 · The idea','Make a long task a sequence of nearby goals.','Words choose a goal region. A graph of expert demonstrations supplies the intermediate frames.')
pill(s,.65,2.35,3.0,'1  SENTENCE → GOAL REGION');pill(s,4.78,2.35,3.0,'2  FIND A ROUTE');pill(s,9.0,2.35,3.0,'3  PLAN · ACT · REPEAT')
shape(s,.63,2.98,3.54,2.72,'white','line',True)
text(s,.86,3.2,3.1,.64,'“Turn the T clockwise.”',23,'navy',True)
text(s,.86,4.07,3.0,.8,'Qwen parser\n→ turn +90°',21,'teal',True)
text(s,.86,5.1,3.0,.35,'Region: angle + drift tolerance',12,'muted')
# Graph is an explicitly labelled schematic, with two demo tracks and a switch.
for y in (3.65,4.7):
 for x in (4.75,5.42,6.09,6.76,7.43):
  if x<7.43:ln(s,x+.12,y+.12,x+.79,y+.12,'gray',2)
  circle(s,x,y,.24,'gray')
ln(s,5.54,3.77,6.21,4.82,'teal',4,True)
ln(s,4.87,3.77,5.54,3.77,'teal',4)
ln(s,6.21,4.82,7.55,4.82,'teal',4)
for x,y in [(4.75,3.65),(5.42,3.65),(6.09,4.7),(6.76,4.7),(7.43,4.7)]:circle(s,x,y,.24,'teal')
text(s,4.7,3.04,3.4,.37,'Demo A',15,'muted');text(s,4.7,5.2,3.4,.54,'Switch demos near matching states.\nGraph drawing is schematic.',13,'muted')
block(s,8.84,3.0,3.87,2.72,'Next demo frame','Encode frame → frozen CEM\nExecute ≈25 steps\nRead live state again','teal')
ln(s,4.25,4.1,4.62,4.1,'teal',2.5,arrow=True);ln(s,7.95,4.1,8.7,4.1,'teal',2.5,arrow=True)
text(s,.66,6.1,12.0,.65,'427,396 nodes · one node every 5 demo steps · train split only\nRetrieval uses simulator state; LeWM is not trained with language.',16,'muted')
note(s,'Walk the audience left to right. Words select a region, graph retrieval chooses real demo frames, and the unchanged frozen model and CEM chase the next nearby frame. Replan after each hop. Graph: 427,396 nodes, 410,579 forward edges, 2,852,207 jump edges. Node stride 5; jump radius 15 in the combined state metric, not simply a 15-pixel block-radius rule. Retrieval uses simulator state. No physical-robot deployment is claimed. Diagram is conceptual, not a logged route.','scripts/demo_breadcrumbs.py; scripts/word_breadcrumbs.py; research/artifacts/5090/dataset50_crumbs_b300_n50.json',4)

# 5 — main empirical advantage, editable native chart.
s=base('04 · Demo breadcrumbs','The biggest gain appears on the farther demo goals.','Same planner and budget within each row. Only the intermediate subgoals change.')
text(s,10.5,2.22,2.1,.65,'+38 pp',33,'teal',True,PP_ALIGN.RIGHT)
text(s,10.1,2.89,2.5,.52,'at offset 100',16,'muted',False,PP_ALIGN.RIGHT)
cd=CategoryChartData();cd.categories=['Demo +50 · n=50','Demo +100 · n=50','Live goal · n=20','Word templates · n=64']
cd.add_series('Flat',[.48,.20,0,18/64]);cd.add_series('Breadcrumbs',[.68,.58,.15,34/64])
chart=s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED,Inches(.6),Inches(2.48),Inches(9.45),Inches(3.75),cd).chart
chart.has_legend=True;chart.legend.position=XL_LEGEND_POSITION.BOTTOM;chart.legend.font.name=FONT;chart.legend.font.size=Pt(13)
chart.chart_style=10;chart.category_axis.reverse_order=True
chart.category_axis.tick_labels.font.name=FONT;chart.category_axis.tick_labels.font.size=Pt(14)
chart.value_axis.minimum_scale=0;chart.value_axis.maximum_scale=1;chart.value_axis.major_unit=.25;chart.value_axis.tick_labels.number_format='0%';chart.value_axis.tick_labels.font.size=Pt(11)
chart.value_axis.has_major_gridlines=True
chart.value_axis.major_gridlines.format.line.color.rgb=col('line')
plot=chart.plots[0];plot.gap_width=85;plot.has_data_labels=True;plot.data_labels.position=XL_LABEL_POSITION.OUTSIDE_END;plot.data_labels.number_format='0%';plot.data_labels.font.size=Pt(13)
for series,c in zip(chart.series,['gray','teal']):series.format.fill.solid();series.format.fill.fore_color.rgb=col(c);series.format.line.fill.background()
text(s,10.43,3.58,2.05,.44,'PAIRED +100 TASKS',12,'muted',True)
text(s,10.43,4.16,2.2,.55,'23 gained',25,'teal',True)
text(s,10.43,4.91,2.2,.55,'4 lost',25,'orange',True)
text(s,10.43,5.66,2.2,.55,'19 net / 50',21,'navy',True)
text(s,.68,6.42,12,.39,'Budget 300. Rows have different frozen success criteria; compare methods within each row.',13,'muted')
note(s,'Primary result. All scored rows have execution budget 300. Offset-50 flat 24/50 versus crumbs 34/50; offset-100 10/50 versus 29/50; live 0/20 versus 3/20; seven templates 18/64 versus 34/64. Dataset rows score the original image-goal task, whereas word tasks score block-only command regions. Do not pool their rates. Live goal median block error fell 162.24→58.46 px, but 3/20 remains limited. Template tasks reuse ten reset seeds. Paired discordant outcomes: offset50 crumbs-only16 vs flat-only6; offset100 23 vs4. Improvements are descriptive; no significance test is added.','research/artifacts/5090/dataset{50,100}_{flat,crumbs}_b300_n50.json; live_{flat,crumbs}_b300_n20.json; words_{flat,crumbs}_b300_n64.json',5)

# 6 — language parsing vs execution.
s=base('05 · Words as the goal','The language interface works; execution is the larger gap.','Qwen2.5-1.5B-Instruct · local, greedy · 120 held-out sentences')
shape(s,.62,2.48,5.15,3.73,'white','line',True)
text(s,.85,2.7,4.7,.5,'101/120',38,'teal',True)
text(s,.85,3.35,4.7,.43,'sentences parsed correctly · 84.2%',19,'navy')
# Each tile is a real scored sentence, grouped only for count readability.
for i in range(120):shape(s,.87+(i%20)*.229,4.07+(i//20)*.19,.17,.13,'teal' if i<101 else 'pale',radius=True)
text(s,.85,5.52,4.7,.47,'Keyword baseline: 31/120 · 25.8%',17,'muted')
text(s,6.18,2.7,6.1,.45,'END-TO-END · SAME 111 ELIGIBLE TASKS',14,'muted',True)
text(s,6.18,3.43,2.85,.4,'Gold command',19,'navy',True);text(s,9.65,3.27,2.85,.7,'57/111',35,'blue',True)
bar(s,6.18,4.02,5.55,57,111,'blue')
text(s,6.18,4.64,2.85,.4,'Parsed command',19,'navy',True);text(s,9.65,4.48,2.85,.7,'51/111',35,'teal',True)
bar(s,6.18,5.23,5.55,51,111,'teal')
text(s,6.18,5.8,6.0,.48,'Parsing costs 6 net successes (−5.4 pp).',19,'navy',True)
text(s,.68,6.47,12,.31,'Nine pushes excluded: gold destination outside arena. Tasks reuse 10 reset seeds; budget 300.',12,'muted')
note(s,'The 120 test sentences were committed before the parser evaluation. L1 accuracy101/120; keyword31/120. L2 excludes9 invalid gold destinations, leaving111 sentence/start tasks on10 reset seeds, not111 independent starting states. Gold success57, parsed51. Paired outcomes8 gold-only and2 parsed-only, so net6. The claim of ≥90% parsing was not met; the ≤5pp execution-loss claim narrowly missed by0.4pp. Do not imply those preregistered thresholds passed. Gold commands still failed54/111, larger than the net parser loss.','research/artifacts/5090/parse_paraphrases_n120.json; language_paraphrases_b300_n111.json; HANDOFF_5090.md',6)

# 7 — actual measured drift of the successful turn rows.
turns=[r for r in D['wc']['episodes'] if r['command'].startswith('turn')]
assert sum(r['success'] for r in turns)==8 and sum(r.get('gate_012',False) for r in turns)==0
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(7.7,3.8),facecolor='#F7F6F2');ax.set_facecolor('#F7F6F2')
ax.axvspan(0,15,color='#D9EEEA',alpha=.8);ax.axvspan(15,40,color='#E9EDF0',alpha=.9)
succ=[r for r in turns if r['success']];fail=[r for r in turns if not r['success']]
# Positions within rows are just a beeswarm for legibility, not a second measured variable.
for rows,y,color,marker in [(succ,1,'#067F7C','o'),(fail,0,'#B96643','x')]:
 xs=[r['block_moved_px'] for r in rows];ys=[y+(i%4-1.5)*.075 for i in range(len(rows))]
 ax.scatter(xs,ys,s=65,c=color,marker=marker,zorder=3)
ax.axvline(15,color='#067F7C',ls='--',lw=1.4);ax.axvline(40,color='#61717B',ls='--',lw=1.4)
ax.text(7.5,1.42,'<15 px',ha='center',color='#067F7C',weight='bold');ax.text(27.5,1.42,'<40 px',ha='center',color='#61717B',weight='bold')
ax.set(xlim=(-3,112),ylim=(-.38,1.65),xlabel='Block drift at the recorded end state (pixels)',yticks=[0,1],yticklabels=['Did not pass\nturn task','Passed\nturn task'])
ax.grid(axis='x',alpha=.16);fig.tight_layout(pad=.7)
fig.savefig(AS/'turn_drift.png',dpi=220);fig.savefig(AS/'turn_drift.svg');plt.close(fig)
s=base('06 · Turns are not in-place','The T turns, but it slides while turning.','A looser task success must not be presented as an in-place rotation.')
s.shapes.add_picture(str(AS/'turn_drift.png'),Inches(.62),Inches(2.54),width=Inches(8.14))
text(s,9.23,2.65,3.1,.45,'BREADCRUMBS',13,'teal',True)
text(s,9.23,3.2,3.1,.76,'8/20',42,'teal',True)
text(s,9.23,3.97,3.1,.63,'turn tasks reached\nwith <40 px drift',18,'navy')
text(s,9.23,4.86,3.1,.73,'0/20',42,'orange',True)
text(s,9.23,5.63,3.1,.62,'pass the stricter\n<15 px in-place gate',18,'navy')
text(s,.68,6.48,12,.3,'Flat: 4/20 turns, 1/20 strict. In-place turns exist in demos: 8,040 nodes across 1,691 episodes.',12,'muted')
note(s,'Every plotted point is a saved end-of-run drift from the 20 crumbs turn tasks, not a generated trajectory. Y separates the saved success flag; vertical jitter is visual only. The eight successful runs all stop with drift between15 and40px. Gate012 is success AND final drift<15px;0/20 crumbs,1/20 flat. Main turn task also requires angle error<20°. Source demo coverage8,040/427,396 nodes in1,691/16,817 episodes is not execution success. No claim of strict in-place control.','research/artifacts/5090/words_crumbs_b300_n64.json; words_flat_b300_n64.json; scripts/word_breadcrumbs.py',7)

# 8 — learned head and meaningful control ablation.
s=base('07 · A head instead of the graph','Lower latent error did not deliver better control.','A small head predicts the next latent from start latent, current latent and a frozen text embedding.')
block(s,.64,2.5,4.05,2.38,'Text → subgoal head','At test time:\nno graph · no command parser','purple')
text(s,.85,5.05,3.75,.76,'0.67×',41,'purple',True)
text(s,.85,5.89,3.65,.7,'validation MSE vs copying\nthe current latent',17,'muted')
rows=[('Graph + gold command',57,'blue'),('Graph + parsed command',51,'teal'),('Learned subgoal head',18,'purple'),('Head + shuffled text',10,'gray')]
for i,(label,v,c) in enumerate(rows):
 y=2.55+i*.95;text(s,5.18,y,6.8,.38,label,18,'navy',True);bar(s,5.18,y+.48,5.45,v,111,c,f'{v}/111')
text(s,5.18,6.48,7.2,.32,'Same eligible sentences and starts · budget 300',12,'muted')
note(s,'The frozen Qwen embedding is mean-pooled last layer. Head learns next nearby subgoal latent. Training:22,420 examples from5,521 plans, validation2,210 examples; valMSE122.858 versus copy184.718, ratio0.6651. Regression split is separate from111-task control evaluation; do not confuse MSE with success. At test no graph/parser. Head18/111 versus shuffled10/111; net8 does not meet the frozen +10 claim. Main benefit is target requests11/30 vs2/30; turns3/45 vs3/45, pushes4/36 vs5/36. No general language-control result is claimed.','research/artifacts/5090/latent_subgoal_train.json; latent_subgoal_paraphrases_b300_n111.json; latent_subgoal_paraphrases_shuffled_b300_n111.json; language_paraphrases_b300_n111.json',8)

# 9 — Lead’s closing message, with bounded implications.
s=base('08 · What the defence should take','Words can name routes the frozen planner can follow.',dark=True)
for y,num,title,body in [(2.45,'01','Nearby demo frames make distant goals easier.','Offset 100: 10/50 → 29/50 with breadcrumbs.'),(3.77,'02','Language can select the route.','101/120 parsed correctly; 57 → 51 successes after parsing.'),(5.09,'03','Reachability is still the limiting factor.','Strict in-place turns: 0/20. Learned head: 18/111.')]:
 circle(s,.66,y,.47,'teal',num);text(s,1.4,y-.04,10.6,.47,title,24,'white',True);text(s,1.4,y+.52,10.6,.45,body,18,'mint')
text(s,.68,6.54,12,.28,'Frozen LeWM throughout · descriptive results on the saved task sets · no new runs for this deck',12,'mint')
note(s,'Close on the lead’s original conclusion. Same frozen model can be more useful when given reachable intermediate targets. The graph+parser is the strongest tested language interface here, not a general solution to language-grounded manipulation. Short-hop competence is not long-horizon or strict in-place competence. The next slide concludes the talk; three backup slides follow for questions.','defense_presentation.pdf; HANDOFF_5090.md; all main-slide result sources',9)

# 11 — conclusion and next steps (new in v2).
s=base('09 · Conclusion','A frozen world model can follow words, one reachable hop at a time.',dark=True)
for x,head,items in [(.6,'WHAT WE SHOWED',['Demo breadcrumbs lift success by 20–38 points on demo goals.','Free sentences work end to end: 51/111 (57 with perfect parsing).','Words can enter the latent directly: a first signal, mainly for “put it on the target”.']),
 (6.75,'WHAT COMES NEXT',['Strict in-place turns (0/20 today)','Random live goals (3/20 today)','Latent head: more data, relative commands','The lab UR3 arm with teleoperated demos'])]:
 shape(s,x,2.35,5.95,3.5,'panel',radius=True);text(s,x+.3,2.55,5.4,.35,head,13,'mint',True)
 for j,it in enumerate(items):
  yy=3.05+j*(.62 if len(items)==4 else .88);circle(s,x+.32,yy+.07,.16,'teal');text(s,x+.66,yy,5.0,.8,it,16,'white')
text(s,.6,6.12,12.1,.6,'Thank you. Questions welcome.',28,'white',True)
note(s,'Close in three sentences: breadcrumbs make the frozen planner useful on goals it cannot reach alone; a small language model turns everyday sentences into those goals with a small loss; feeding words straight into the latent is possible but still weak. Then name the honest limits as next steps rather than as failures: strict in-place turns, random live goals, the latent head. End with the robot plan: Faith and Didar will repeat the method on the lab UR3 with a camera over a table, a 3D-printed T with a tag, teleoperated demos, and LeWM retrained on those. Backup slides A–C follow for questions.','HANDOFF_5090.md; research/artifacts/5090/*.json')

# Appendix A — paired outcomes, more revealing than two bars alone.
s=base('Backup A · Paired evidence','Where did the extra successes come from?','Matched tasks let us count gains and losses directly. These are descriptive counts.')
headers=['TASK','BOTH SUCCEED','FLAT ONLY','CRUMBS ONLY','NEITHER']
xs=[.7,4.05,6.12,8.2,10.28]
for x,h in zip(xs,headers):text(s,x,2.55,2.7,.46,h,12,'muted',True)
for i,(pre,label,n) in enumerate([('d50','Demo +50 · n=50',50),('d100','Demo +100 · n=50',50),('live','Live goals · n=20',20),('w','Templates · n=64',64)]):
 y=3.25+i*.7;shape(s,.63,y-.06,12.05,.62,'white',radius=True);text(s,.81,y,3.1,.43,label,18,'navy',True)
 for x,key in zip(xs[1:],['both','flat_only','crumbs_only','neither']):text(s,x,y,1.5,.44,str(paired[pre][key]),21,'teal' if key=='crumbs_only' else 'navy',key=='crumbs_only')
text(s,.7,6.41,12,.36,'Offset 100: 23 gained, 4 lost → 19 net extra successes out of 50 (+38 percentage points).',16,'teal',True)
note(s,'Counts recomputed directly from matched episode rows. Matching keys are episode/start for dataset tasks, seed for live, seed/command for templates. No new inferential test is added. Runs within a row share initial states and planner settings; different rows have different task criteria.','research/artifacts/5090/{dataset50,dataset100,live,words} matched flat/crumbs JSONs')

# Appendix B — text effect is narrow, not general control.
s=base('Backup B · Action breakdown','The head’s text effect is mainly on target requests.','Same 111 eligible tasks; target n=30, turns n=45, pushes n=36.')
for i,(label,n,graph,head,shuf) in enumerate([('PUT ON TARGET',30,20,11,2),('TURN',45,17,3,3),('PUSH',36,14,4,5)]):
 x=.67+i*4.2;shape(s,x,2.55,3.82,3.59,'white','line',True);text(s,x+.2,2.78,3.3,.4,label,16,'navy',True)
 for j,(nm,v,c) in enumerate([('Graph + parser',graph,'teal'),('Head',head,'purple'),('Shuffled text',shuf,'gray')]):
  y=3.5+j*.77;text(s,x+.2,y,2.2,.34,nm,15,'muted');text(s,x+2.6,y,1.0,.4,f'{v}/{n}',18,c,True);bar(s,x+.2,y+.4,3.25,v,n,c)
text(s,.7,6.48,12,.32,'These are successes on the saved command sets, not evidence of general language understanding.',13,'muted')
note(s,'Graph parsed:target20/30,turn17/45,push14/36. Head11/30,3/45,4/36. Shuffled2/30,3/45,5/36. Gold graph reference23/30,18/45,16/36. Same held-out sentences, same reset seeds. Text shuffling is a control for sentence contribution, not a separate language model.','research/artifacts/5090/language_paraphrases_b300_n111.json; latent_subgoal_paraphrases_b300_n111.json; latent_subgoal_paraphrases_shuffled_b300_n111.json')

# Appendix C — actual logged block endpoint reconstruction.
s=base('Backup C · A measured example','A successful quarter-turn can miss the in-place gate.','Seed 1 · clockwise turn template · breadcrumbs · requested change: +90°')
text(s,.82,2.4,3.4,.4,'START',14,'muted',True)
text(s,4.55,2.4,3.4,.4,'RECORDED END',14,'muted',True)
s.shapes.add_picture(str(AS/'turn_seed1_start.png'),Inches(.72),Inches(2.95),width=Inches(3.1))
s.shapes.add_picture(str(AS/'turn_seed1_end.png'),Inches(4.47),Inches(2.95),width=Inches(3.1))
ln(s,3.98,4.48,4.31,4.48,'teal',3,arrow=True)
text(s,8.1,2.95,4.6,.65,'77.2°',40,'teal',True)
text(s,8.1,3.65,4.6,.43,'actual clockwise turn',20,'navy')
text(s,8.1,4.4,4.6,.65,'22.5 px',40,'orange',True)
text(s,8.1,5.1,4.6,.65,'block drift\npasses <40 px, fails <15 px',19,'navy')
text(s,.75,6.44,11.9,.34,'Reconstructed from logged block poses; pusher omitted. Green T = requested goal. No trajectory was saved.',12,'muted')
note(s,'Concrete example from the saved seed1 turn_cw row: success=true, drift22.4685790448px, signed turn77.1700203478deg. Requested+90deg; angleerror12.82998deg is within20deg. Drift meets40px task gate and fails15px strict gate. Images use the existing simulator renderer with saved block start/end states; no env.step or planner call. Pusher is omitted rather than fabricating its final position. Goal is reconstructed from the command, not a measured intermediate frame. These are two endpoints, not a rollout video. Fixed-backbone protocol: frozen192dimLeWM, CEM300samples/30iterations/top30, horizon5/actionblock5, executionbudget300. State retrieval/trainsplit as slide4.', 'research/artifacts/5090/words_crumbs_b300_n64.json, seed1/turn_cw; scripts/word_breadcrumbs.py')

# Appendix D — demo graph coverage (new in v2).
g=D['dry']['graph'];ipt=D['wdry']['in_place_turns']
assert D['dry']['summary']['with_path']==50 and D['livedry']['summary']['with_path']==13 and D['wdry']['summary']['with_route']==56
s=base('Backup D · The demo graph','The graph covers demo goals, not random live goals.','Training split only. One node every 5 demo steps; nearby states from different demos are linked.')
for i,(v,lab) in enumerate([(g['nodes'],'demo states (nodes)'),(g['forward_edges'],'“next frame” links'),(g['jump_edges'],'links between demos')]):
 x=.6+i*4.13;text(s,x,2.4,3.9,.7,f'{v:,}',36,'teal' if i<2 else 'blue',True);text(s,x,3.12,3.9,.4,lab,16,'muted')
for i,(head,big,c,body) in enumerate([
 ('DEMO GOALS, +50 STEPS','50/50','teal','have a route · median 0.5 demo switches\nstart lies 9 state units from a demo state'),
 ('WORD COMMANDS','56/64','teal','start/command pairs have a route\nevery command has matching demo states'),
 ('RANDOM LIVE GOALS','13/20','orange','have a route, but start and goal lie\n37 and 39 state units from any demo state')]):
 x=.6+i*4.13;shape(s,x,3.85,3.85,2.35,'white','line',True)
 text(s,x+.22,4.03,3.4,.35,head,12,'muted',True);text(s,x+.22,4.42,3.4,.75,big,36,c,True);text(s,x+.22,5.25,3.45,.85,body,13,'navy')
text(s,.65,6.45,12,.32,f'In-place turns do occur in the demos: {ipt["nodes"]:,} nodes in {ipt["episodes"]:,} of {ipt["of_episodes"]:,} episodes. State units: block/pusher px, angle at 60 per radian.',12,'muted')
note(s,'Use this if asked why live goals are hard. The graph is dense around what experts actually did, so held-out demo goals always have a route and the start is close (median 9 state units). Random live goals are far from every demo state (median 37 at the start, 39 at the goal), so the first and last hops are long and the planner leaves its reliable range. The state distance mixes pusher xy, block xy (px) and block angle (60 units per radian). Jump links join states from different demos closer than 15 units.','research/artifacts/5090/dataset50_dry_n50.json; live_dry_n20.json; words_dry_n64.json')

# Appendix E — every pre-registered check (new in v2).
s=base('Backup E · Every check','Each claim and kill line was committed before its scored run.','Verdict: holds = claim met; between = claim missed, kill line not reached. Budget 300 on all scored runs.')
rowsE=[('Check','Claim (frozen first)','Kill line','Result','Verdict'),
 ('0 · Setup reproduces the 4090','≈ 45/50','≤ 40/50','47/50','holds'),
 ('1a · Routes exist, demo goals','≥ 35/50','< 35/50','50/50','holds'),
 ('2b · Breadcrumbs, +50 steps','flat + 10 pp','≤ flat + 2 pp','68% vs 48%','holds'),
 ('2c · Breadcrumbs, +100 steps','gap grows','crumbs ≤ flat','58% vs 20%','holds'),
 ('2d · Random live goals','≥ 4/20','≤ flat + 1','3/20 vs 0/20','between'),
 ('3b · Word templates','flat + 10 pp','≤ flat + 1 ep','34 vs 18 of 64','holds'),
 ('3c · Turns (40 px drift allowed)','≥ 5/20','≤ 2/20','8/20','holds'),
 ('L1 · Sentence parsing','≥ 90%','≤ 75%','84.2%','between'),
 ('L2 · Parsed vs gold, end to end','≥ gold − 5 pp','≤ gold − 15 pp','−5.4 pp','between'),
 ('S1 · Head, offline error','≤ 0.7× copy','≥ 0.9× copy','0.67×','holds'),
 ('S2 · Head, end to end','≥ 34/111','≤ 11/111','18/111','between'),
 ('S3 · Words matter for the head','shuffled + 10','≤ shuffled + 3','+8','between')]
tb=s.shapes.add_table(len(rowsE),5,Inches(.6),Inches(2.35),Inches(12.1),Inches(4.0)).table
for j,wd in enumerate([4.1,2.3,2.2,2.1,1.4]):tb.columns[j].width=Inches(wd)
for i,row in enumerate(rowsE):
 tb.rows[i].height=Inches(.305)
 for j,v in enumerate(row):
  cell=tb.cell(i,j);cell.fill.solid();cell.fill.fore_color.rgb=col('navy' if i==0 else ('white' if i%2 else 'pale'))
  cell.margin_top=cell.margin_bottom=Pt(2);cell.margin_left=Pt(8);cell.vertical_anchor=MSO_ANCHOR.MIDDLE
  p=cell.text_frame.paragraphs[0];p.text=v;p.font.name=FONT;p.font.size=Pt(12);p.font.bold=(i==0 or j==4)
  p.font.color.rgb=col('white' if i==0 else ('teal' if v=='holds' else 'orange' if v=='between' else 'navy'))
note(s,'Use this if the panel asks how honest the numbers are. Every row was written into HANDOFF_5090.md and pushed to git before the scored run, with the number that would kill it. Seven checks hold; five land between claim and kill; none hit a kill line. 3c holds only on the frozen 40 px drift gate; on the stricter 15 px in-place gate it is 0/20 (slide 8).','HANDOFF_5090.md results tables; research/artifacts/5090/*.json')

# References (new in v2).
s=base('References','Sources')
refs=['LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. OpenReview, v0.9.2.',
 'Assran, M., et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA). CVPR.',
 'Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., Balestriero, R. (2026). LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. arXiv:2603.19312.',
 'Caselli, Massafra, Punzo, Lo Sardo, Pantelidis, Bhethanabhotla (2026). Hi-LeWM. arXiv:2607.12547.',
 'Florence, P., et al. (2021). Implicit Behavioral Cloning. CoRL. (Push-T task)',
 'Rubinstein, R. Y. (1999). The Cross-Entropy Method for Combinatorial and Continuous Optimization. Methodology and Computing in Applied Probability.',
 'Yang, A., et al. (2024). Qwen2 Technical Report. arXiv:2407.10671. Model used: Qwen2.5-1.5B-Instruct (Qwen Team, 2024).',
 'Myers, V., et al. (2023). Goal Representations for Instruction Following (GRIF). CoRL. arXiv:2307.00117.',
 'Chen, D., et al. (2025). VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language. arXiv:2512.10942.']
y=1.8
for i,r in enumerate(refs):
 two=len(r)>130;text(s,.6,y,.5,.4,f'[{i+1}]',13,'teal',True);text(s,1.15,y,11.5,.62 if two else .4,r,13,'navy');y+=.66 if two else .44
text(s,.6,6.5,12,.32,'Code, frozen claims and all result files: github.com/velosipedistufa/JEPA · branch 5090-run-demo-breadcrumbs',12,'muted')
note(s,'[1] JEPA idea. [2] I-JEPA, the first image JEPA. [3] LeWorldModel, the frozen model and planner we use. [4] Hi-LeWM, closest prior work: same-demo future frames as subgoals. [5] Push-T task. [6] Cross-entropy method used by the planner. [7] Qwen, the parser and text encoder. [8] GRIF and [9] VL-JEPA, related language-to-goal work.','reports/refs.bib; HANDOFF_5090.md; Qwen2.5-1.5B-Instruct model card')

# Set descriptive alternative text and save fully editable slides.
for slide in prs.slides:
 for sp in slide.shapes:
  for eff in sp._element.xpath('./p:style/a:effectRef'):eff.set('idx','0')
  nv=sp._element.find('.//{http://schemas.openxmlformats.org/drawingml/2006/picture}cNvPr')
  # Native text and charts remain accessible through ordinary PowerPoint objects.
  if sp.has_text_frame:
   nodes=sp._element.xpath('.//p:cNvPr')
   if nodes:nodes[0].set('descr',sp.text[:600])
path=OUT/'JEPA_defence_visual.pptx';prs.save(path)
# Artifact-level audit and exact source hashes are kept outside slides for compact notes.
(OUT/'RESULTS_AUDIT.json').write_text(json.dumps({'source_sha256':records,'paired_outcomes':paired,'slides':len(prs.slides),'main_slides':11,'backup_slides':5,'references_slide':1,'new_experiments':False},indent=2)+'\n')
print(path)
print('17 slides: 11 main + 5 backup + references; verified saved totals and matched-pair counts.')
