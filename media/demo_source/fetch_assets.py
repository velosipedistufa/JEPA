"""Fetch licensed robot meshes; verify exact assets used in the reviewed film."""
from pathlib import Path
import json,hashlib,urllib.request,concurrent.futures
p=Path(__file__).resolve().parent
m=json.loads((p/'asset_manifest.json').read_text())
base='https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/'+m['revision']+'/universal_robots_ur5e/'
def fetch(name):
 f=p/name;f.parent.mkdir(parents=True,exist_ok=True)
 if not f.exists():urllib.request.urlretrieve(base+name,f)
 assert hashlib.sha256(f.read_bytes()).hexdigest()==m['files'][name],name
 return name
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 for name in pool.map(fetch,m['files']):print('verified',name)
