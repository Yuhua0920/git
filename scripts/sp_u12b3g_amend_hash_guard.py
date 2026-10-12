"""Freeze opaque provenance hashing compatible with the active outcome firewall."""
from pathlib import Path
import sys,shutil,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u12b3g_schema as s
from scripts.sp_u12b3g_preserve import read,write,sha,OUT
old=read(s.FREEZE_PATH);oldsha=sha(s.FREEZE_PATH)
assert oldsha=='2c8e3a679e82da2254c998be08b79dbc8069c94f956b144fd29178d74a1fa14c'
archive=OUT/'pre_gate_hash_guard_amendment';assert not archive.exists();archive.mkdir()
names=['outputs/sp_u12b3g_next_execution_freeze.json','outputs/sp_u12b3g/pytest.txt','outputs/sp_u12b3g/pytest.xml','outputs/sp_u12b3g/pytest.json','outputs/sp_u12b3g_wrapper_diff_audit.json']+[n for n in old['hashes'] if n.startswith(('scripts/sp_u12b3g','tests/test_sp_u12b3g')) or n in ('scripts/run_sp_u12b3f.py','scripts/sp_u12b3e_io.py')]
leaves={}
for n in names:
 dest=archive/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest);assert sha(dest)==sha(ROOT/n);leaves[n]=dict(sha256=sha(dest),archive_path=dest.relative_to(ROOT).as_posix())
p=ROOT/'scripts/sp_u12b3g_schema.py';text=p.read_text(encoding='utf8');before='def sha(p):\n h=hashlib.sha256()'
after="def sha(p):\n # Hash bytes through the existing opaque-read scope once the firewall is active.\n b=sys.modules.get('scripts.sp_u12b3e_common')\n if b is not None and getattr(b,'_guard',False):return b.sha(p)\n h=hashlib.sha256()"
assert text.count(before)==1;p.write_text(text.replace(before,after),encoding='utf8')
p=ROOT/'scripts/sp_u12b3g_audit.py';text=p.read_text(encoding='utf8');before="print(json.dumps(dict(PASS=True,B2_write_blocked_before_OS=True,B2_bytes_changed=False,expected_policy_denial=True)),flush=True);return"
after="s.validate_execution_freeze(s.load(s.FREEZE_PATH),verify_leaves=False)\n  print(json.dumps(dict(PASS=True,B2_write_blocked_before_OS=True,B2_bytes_changed=False,expected_policy_denial=True,opaque_hash_validation_PASS=True)),flush=True);return"
assert text.count(before)==1;p.write_text(text.replace(before,after),encoding='utf8')
new=dict(old);new['candidate_revision']=3;new['previous_candidate_freeze_sha256']=oldsha;new['technical_wrapper_sha256']=s.wrapper_hash();new['hashes']={n:sha(ROOT/n) for n in old['hashes']};new['hashes']['scripts/sp_u12b3g_amend_hash_guard.py']=sha(Path(__file__))
write(s.FREEZE_PATH,new)
diff=read(ROOT/'outputs/sp_u12b3g_wrapper_diff_audit.json');diff['new_wrapper_sha256']=s.wrapper_hash();diff['new_bundle_files']={n:sha(ROOT/n) for n in s.WRAPPER_FILES};diff['schema_helper_opaque_hash_scope']='uses existing b.sha only when outcome firewall active;no outcome decoding or write allowance';write(ROOT/'outputs/sp_u12b3g_wrapper_diff_audit.json',diff)
write(archive/'manifest.json',dict(files=leaves,old_candidate_sha256=oldsha,new_candidate_sha256=sha(s.FREEZE_PATH),reason='Static integration audit:post-initializer validation must hash sealed provenance through the existing opaque read scope;add active-firewall regression to the readonly probe',scientific_trajectories=0,exact_run_gate_started=False))
print(json.dumps(dict(stage='hash_guard_amendment',new_wrapper_sha256=s.wrapper_hash(),new_candidate_sha256=sha(s.FREEZE_PATH),scientific_trajectories=0)),flush=True)
