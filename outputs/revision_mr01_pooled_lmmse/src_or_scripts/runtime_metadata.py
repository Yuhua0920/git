"""Metadata-only supplement: installed profile tables and runtime identities.

No channel generation, model fit, outcome table, or frozen code is changed.
"""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,platform,sys,winreg
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.root).resolve();out=Path(a.out).resolve();sys.path.insert(0,str(root))
from sionna.phy.channel.tr38901 import models
import torch
paths={name:Path(models.parameter_file('CDL-'+name+'.json','19.2')) for name in ('B','C')}
profiles={name:dict(path=str(v),bytes=v.stat().st_size,sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for name,v in paths.items()}
try:
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
except OSError as e:cpu='Unavailable: '+str(e)
r=dict(profiles=profiles,spec_version='19.2',profile_to_conditions=dict(B=['Profile','Composed'],C=['ID','Mobility','Boundary']),generation_count=0,source_scope='Supplement to first-ID provenance; authenticates actual installed B/C parameter tables separately',CPU=cpu,Python=sys.version,executable=sys.executable,platform=platform.platform(),dependencies={x:importlib.metadata.version(x) for x in ('numpy','scipy','pytest','torch','sionna-no-rt')},CUDA_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,CUDA_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None)
with (out/'reports/condition_parameter_runtime.json').open('x',encoding='utf8') as f:json.dump(r,f,indent=2)
print(json.dumps(dict(metadata='PASS',profile_tables=list(profiles),CPU=cpu,CUDA=r['CUDA_device'],H_generated=0)))
