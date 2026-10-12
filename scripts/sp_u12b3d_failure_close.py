"""Technical-only failure closure, frozen before formal entry. No retry/compute."""
from pathlib import Path
import sys,json,os,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u12b3d_common as b
from scripts.sp_u12b3d_build import core
def main():
 b.need((b.TECH/'PERMANENT_STOP.json').exists(),'FAILURE_CLOSURE_REQUIRES_STOP')
 f=b.read(b.TECH/'execution_freeze.json');stop=b.read(b.TECH/'PERMANENT_STOP.json')
 exceptions=[]
 for path in sorted(b.TECH.glob('*exception*.json')):exceptions.append(dict(path=str(path.resolve()),sha256=b.sha(path),envelope=b.read(path)))
 b.need(bool(exceptions),'FAILURE_WITHOUT_FORENSICS')
 # Count only metadata/committed receipts; never read scientific arrays or tables.
 counts={c:len(list((b.SEAL/'raw'/c).glob('*.json'))) for c in b.CONDITIONS}
 write=lambda n,o:b.write(ROOT/'outputs'/('sp_u12b3d_run2_'+n+'.json'),o)
 for name in ('integrity_audit','both_replay_audit','bootstrap_audit'):write(name,dict(PASS=False,complete=False,formal_run_stopped=True,completed_trajectories=sum(counts.values()),completed_per_condition=counts,outcomes_viewed=False,reason='TECHNICAL_FAILURE_BEFORE_COMPLETE_GATE'))
 write('io_audit',dict(PASS=False,formal_exception=True,forensics=exceptions,retries=0,automatic_resume=False,formal_patch=False,outcomes_viewed=False))
 write('resource_log',dict(PASS=False,parent_pid=os.getpid(),parent_peak=b.peak(),CPU_workers=2,BLAS_threads=1,retries=0,resumes=0,new_H_generated=0,sealed_disk_bytes=sum(p.stat().st_size for p in b.SEAL.rglob('*') if p.is_file()),runtime_claim=False,outcomes_viewed=False))
 leaves={p.relative_to(b.SEAL).as_posix():dict(sha256=b.sha(p),bytes=p.stat().st_size) for p in sorted(b.SEAL.rglob('*')) if p.is_file()}
 m=dict(status='B3_RUN2_TECHNICAL_FAILURE_WITH_FORENSICS',stop=stop,mentor_authorization=b.read(b.TECH/'mentor_authorization.json'),historical_failure_disclosure_sha256=b.sha(ROOT/'docs/sp_u12b3d_historical_failure_disclosure.md'),scientific_core_sha256_before=f['scientific_core_sha256'],scientific_core_sha256_after=core()[0],io_wrapper_sha256=f['io_wrapper_sha256'],execution_freeze_sha256=b.sha(b.TECH/'execution_freeze.json'),formal_seed_registry_sha256=f['formal_seed_registry_sha256'],bootstrap_seed_registry_sha256=f['bootstrap_seed_registry_sha256'],Run1_use='PROHIBITED',B2_seal=b.read(ROOT/'outputs/sp_u12b3d_run2_b2_seal_audit.json'),Run1_archive=b.read(ROOT/'outputs/sp_u12b3d_run2_run1_preservation.json'),preservation=b.read(ROOT/'outputs/sp_u12b3d_run2_preservation.json'),raw_partial_tree=b.tree(leaves),files=leaves,outcomes_viewed=False,scientific_figures_viewed=False,retries=0)
 write('manifest',m)
 cert='# SP-U12B3D technical failure seal certificate\n\n'+json.dumps({k:v for k,v in m.items() if k!='files'},indent=2)+'\n\nFull forensic records: '+', '.join(e['path'] for e in exceptions)+'\nNo continuation, retry, patch or Run3. STOP for mentor review.\n'
 b.atomic(ROOT/'docs/sp_u12b3d_run2_seal_certificate.md',cert.encode());b.atomic(ROOT/'docs/sp_u12b3d_run2_execution_log.md',cert.encode())
 b.write(b.TECH/'decision.json',dict(verdict=m['status'],STOP=True,outcomes_viewed=False));b.emit(stage='failure_close',verdict=m['status'],outcomes_viewed=False)
if __name__=='__main__':main()
