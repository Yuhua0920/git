"""Preserve pre-test candidate and amend only an exception-inspection test helper."""
from pathlib import Path
import sys,json,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u12b3g_schema as s
from scripts.sp_u12b3g_preserve import read,write,sha,OUT
old=read(s.FREEZE_PATH);oldsha=sha(s.FREEZE_PATH)
assert oldsha=='b6351a4d8af8edb6788fb461b936d951abca8b45e0de5fb493bff08c167d29a8'
archive=OUT/'pretest_failure01';assert not archive.exists();archive.mkdir()
names=['outputs/sp_u12b3g_next_execution_freeze.json','outputs/sp_u12b3g/pytest.txt','outputs/sp_u12b3g/pytest.xml','outputs/sp_u12b3g/pytest.json']+[n for n in old['hashes'] if n.startswith(('scripts/sp_u12b3g','tests/test_sp_u12b3g')) or n in ('scripts/run_sp_u12b3f.py','scripts/sp_u12b3e_io.py')]
leaves={}
for n in names:
 dest=archive/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest);assert sha(dest)==sha(ROOT/n);leaves[n]=dict(sha256=sha(dest),archive_path=dest.relative_to(ROOT).as_posix())
p=ROOT/'scripts/sp_u12b3g_audit.py';text=p.read_text(encoding='utf8')
before="need('B2_WRITE_VIOLATION' in str(exc) and not p.exists(),'READONLY_PROBE_FAILED')"
after="need(exc.args and 'B2_WRITE_VIOLATION' in exc.args[0] and not p.exists(),'READONLY_PROBE_FAILED')"
assert text.count(before)==1;p.write_text(text.replace(before,after),encoding='utf8')
new=dict(old);new['candidate_revision']=2;new['previous_candidate_freeze_sha256']=oldsha;new['hashes']={n:sha(ROOT/n) for n in old['hashes']};new['hashes']['scripts/sp_u12b3g_amend.py']=sha(Path(__file__))
write(s.FREEZE_PATH,new)
schema=read(ROOT/'outputs/sp_u12b3g_execution_freeze_schema.json');schema['optional'].update(candidate_revision='pre-scientific technical candidate revision',previous_candidate_freeze_sha256='archived rejected/pre-test candidate hash');write(ROOT/'outputs/sp_u12b3g_execution_freeze_schema.json',schema)
write(archive/'manifest.json',dict(files=leaves,old_candidate_sha256=oldsha,new_candidate_sha256=sha(s.FREEZE_PATH),reason='Readonly guard worked;OSError filename formatting hides reason in str(exc);inspect preserved exc.args instead',scientific_trajectories=0,B2_write_reached_OS=False,wrapper_unchanged=True))
print(json.dumps(dict(stage='technical_test_amendment',old_candidate_sha256=oldsha,new_candidate_sha256=sha(s.FREEZE_PATH),scientific_trajectories=0,wrapper_sha256=s.wrapper_hash())),flush=True)
