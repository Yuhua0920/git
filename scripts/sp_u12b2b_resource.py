"""Finalize engineering worker policy before execution freeze / any formal H."""
from pathlib import Path
import sys,os,json
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts import sp_u12b2b_common as b
b.need(not (b.TECH/'execution_freeze.json').exists() and not b.SEAL.exists(),'RESOURCE_AMENDMENT_AFTER_FREEZE')
cfg=b.config();old_hash=b.sha(b.CFG);memory=b.memory_available()
workers=min(2,max(1,(os.cpu_count() or 1)//2),max(1,int(memory['available_physical_bytes']/2**30)-2))
b.write(b.TECH/'resource_config_before.json',cfg)
cfg.update(worker_count=workers,memory_preflight=memory,worker_policy='At most2CPU workers;1GiB/worker plus2GiB generator/headroom reservation;BLAS1unchanged;single default-device generator.')
b.write(b.CFG,cfg)
b.write(b.TECH/'resource_amendment.json',dict(before_sha256=old_hash,after_sha256=b.sha(b.CFG),worker_count=workers,resource_policy=cfg['worker_policy'],memory=memory,formal_H_generated=0,scientific_design_changes=0,formal_seed_changes=0,reason='Bounded engineering concurrency following serial/parallel synthetic byte-equivalence preflight',created_utc=b.timestamp()))
for suffix in ('json','txt','xml'):
 path=b.TECH/('pytest_preflight.'+suffix)
 if path.exists():path.replace(b.TECH/('pytest_preflight_attempt1.'+suffix))
b.emit(stage='resource_preflight',status='PASS',worker_count=workers,formal_H_generated=0,outcomes_viewed=False)
