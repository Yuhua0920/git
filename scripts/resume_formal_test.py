"""T20B external-interruption recovery: unchanged scientific code, exact prefix replay."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import io
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_formal_test as original

OUT=ROOT/'outputs'
BASE_PREFLIGHT=OUT/'t20b_formal_resume_preflight.json'
PREFLIGHT=OUT/'t20b_formal_resume_preflight_v2.json'
RESULT=OUT/'t20b_formal_resume_recovery.json'
ARCHIVED_STOP=OUT/'t20b_resume_directory_failure'/'STOP.json'
SOURCES=('scripts/resume_formal_test.py','tests/test_formal_resume.py')
JOURNALS=('manifest','trajectory_metrics','frame_metrics','input_hashes','weights')


class ResumeOutputPath(type(Path())):
    """Allow only the already-audited raw directory to exist at resume."""
    def mkdir(self, mode=0o777, parents=False, exist_ok=False):
        if self.resolve()==(OUT/'t20b_formal_data').resolve() and self.is_dir():
            return
        return super().mkdir(mode=mode,parents=parents,exist_ok=exist_ok)


def load_native_h(path):
    """Restore the observed frozen torch export's [tx,rx,time,k] storage."""
    with np.load(path,allow_pickle=False) as archive:
        if archive.files!=['H']:raise ValueError('Existing H archive schema drift')
        saved=archive['H']
    if saved.shape!=(2,2,64,160) or saved.dtype!=np.complex128 or not np.isfinite(saved).all():
        raise ValueError('Existing H archive is incomplete')
    return np.ascontiguousarray(saved.transpose(1,0,3,2)).transpose(1,0,3,2)


def archive_directory_stop():
    """Preserve exactly the researcher-authorized pre-generation I/O failure."""
    path=OUT/'t20b_formal_STOP.json'
    if ARCHIVED_STOP.exists():
        if path.exists():raise ValueError('New STOP exists; no further retry permitted')
        return
    stop=original.load(path)
    if (stop['error_type']!='FileExistsError' or stop['generated_formal_trajectories']!=0
        or stop['completed_trajectories']!=0 or stop['observed_trajectory_results']
        or 't20b_formal_data' not in stop['message']
        or original.frozen.sha(path)!=original.load(OUT/'t20b_resume_STOP_audit.json')['stop_record_sha256']):
        raise ValueError('Only the authorized pre-generation directory STOP can be archived')
    before=original.frozen.sha(path)
    ARCHIVED_STOP.parent.mkdir(exist_ok=True)
    path.replace(ARCHIVED_STOP)
    if original.frozen.sha(ARCHIVED_STOP)!=before:raise ValueError('Archived STOP bytes changed')


def csv_line(fields,row=None):
    stream=io.StringIO(newline='')
    writer=original.csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
    if row is None:writer.writeheader()
    else:writer.writerow(row)
    return stream.getvalue().encode('utf8')


class ReplayJournal:
    """Existing bytes must match exact replay; append only beyond the frozen EOF."""
    instances=[]

    def __init__(self,path,fields):
        self.path=Path(path);self.fields=fields;self.old_size=self.path.stat().st_size
        self.reader=self.path.open('rb')
        if self.reader.readline()!=csv_line(fields):
            self.reader.close();raise ValueError('Journal header mismatch: '+str(path))
        self.stream=self.path.open('ab')
        self.replayed=0;self.appended=0
        self.instances.append(self)

    def append(self,row):
        line=csv_line(self.fields,row)
        if self.reader.tell()<self.old_size:
            old=self.reader.readline()
            if old!=line:
                raise ValueError('STOP exact historical formal row mismatch: '+str(self.path)+':'+str(self.replayed+2))
            self.replayed+=1
        else:
            self.stream.write(line);self.stream.flush();self.appended+=1

    def close(self):
        if not self.reader.closed:self.reader.close()
        self.stream.close()


def snapshot():
    interruption=original.load(OUT/'t20b_formal_interruption.json')
    original.frozen.verify_hashes(interruption['current_metric_manifest_unseal_sha256'])
    original.verify_preparation()
    original.governance.environment()
    if (OUT/'t20b_formal_STOP.json').exists():
        raise ValueError('A scientific/implementation STOP cannot be recovered by this entry point')
    raw=OUT/'t20b_formal_data'
    files=sorted(raw.glob('*.npz'))
    h=[p for p in files if not p.stem.endswith('_communication')]
    communication=[p for p in files if p.stem.endswith('_communication')]
    if len(h)!=2561 or len(communication)!=2561:
        raise ValueError('Interruption inventory differs')
    rows=original.readcsv(OUT/'t20b_formal_trajectory_metrics.csv')
    groups={}
    for row in rows:groups.setdefault((row['environment'],int(row['trajectory_id'])),[]).append(row)
    if len(groups)!=2560 or any(len(v)!=16 for v in groups.values()):
        raise ValueError('Incomplete prior trajectory metric group')
    expected={(name,i) for name in original.metrics.CONDITIONS[:2] for i in range(600,1600)}
    expected|={(original.metrics.CONDITIONS[2],i) for i in range(600,1160)}
    if set(groups)!=expected:
        raise ValueError('Prior metrics do not form the fixed execution prefix')
    for values in groups.values():
        if {(int(v['snr_db']),v['method']) for v in values}!={(s,m) for s in original.metrics.SNRS for m in original.metrics.METHODS}:
            raise ValueError('Incomplete paired method/SNR group')
    manifests=original.readcsv(OUT/'t20b_formal_manifest.csv')
    if len(manifests)!=25610:
        raise ValueError('Prior manifest count mismatch')
    raw_hashes={p.relative_to(ROOT).as_posix():original.frozen.sha(p) for p in files}
    for row in manifests:
        if raw_hashes[row['raw_artifact']]!=row['raw_artifact_sha256']:
            raise ValueError('Prior raw channel file bytes changed')
    prefixes={}
    for name in JOURNALS:
        p=OUT/f't20b_formal_{name}.csv'
        if not p.read_bytes().endswith(b'\n'):
            raise ValueError('Truncated journal line')
        prefixes[p.relative_to(ROOT).as_posix()]=dict(bytes=p.stat().st_size,sha256=original.frozen.sha(p))
    return dict(task='T20B recovery',timestamp=datetime.now(timezone.utc).isoformat(),
        original_unseal_sha256=original.frozen.sha(original.UNSEAL),interruption_record_sha256=original.frozen.sha(OUT/'t20b_formal_interruption.json'),
        original_frozen_execution_code_unchanged=True,original_generated_H_count=2561,
        original_completed_trajectory_count=2560,original_metric_rows=40960,
        existing_raw_sha256=raw_hashes,journal_prefixes=prefixes,
        resume_source_sha256={p:original.frozen.sha(ROOT/p) for p in SOURCES},
        all_checks_passed=True,new_unseal=False,new_seeds=False,new_estimator=False,
        recovery_policy='exact replay of frozen prefix and append missing suffix; no existing file truncation or raw overwrite')


def preflight():
    if PREFLIGHT.exists():raise ValueError('Recovery preflight already frozen')
    archive_directory_stop()
    proof=snapshot()
    base=original.load(BASE_PREFLIGHT)
    if proof['existing_raw_sha256']!=base['existing_raw_sha256'] or proof['journal_prefixes']!=base['journal_prefixes']:
        raise ValueError('Original interruption boundary changed')
    # One existing canonical realization checks serialization/layout only.
    # No performance, aggregate or endpoint is evaluated here.
    name=original.metrics.CONDITIONS[0];i=600
    native=original.FormalCDL(name,original.seed_index()).generate(i)
    saved=load_native_h(OUT/'t20b_formal_data'/f'{name}_{i}.npz')
    if not np.array_equal(native,saved) or native.strides!=saved.strides:
        raise ValueError('Saved H cannot reproduce the frozen numerical memory layout')
    proof.update(previous_preflight_sha256=original.frozen.sha(BASE_PREFLIGHT),
        archived_directory_stop_sha256=original.frozen.sha(ARCHIVED_STOP),
        saved_h_native_values_and_strides_exact=True,native_h_strides=list(native.strides),
        existing_H_loaded_without_redraw=True,scientific_interpretation='original preregistered formal test')
    original.write(PREFLIGHT,proof)
    print('Recovery inventory PASS: 2561 raw / 2560 completed; unseal/code/environment exact',flush=True)


def verify_prefixes(proof):
    import hashlib
    for path,info in proof['journal_prefixes'].items():
        with (ROOT/path).open('rb') as stream:content=stream.read(info['bytes'])
        if len(content)!=info['bytes'] or hashlib.sha256(content).hexdigest()!=info['sha256']:
            raise ValueError('Original journal prefix mutated: '+path)
    original.frozen.verify_hashes(proof['existing_raw_sha256'])
    if original.frozen.sha(original.UNSEAL)!=proof['original_unseal_sha256']:
        raise ValueError('Original unseal bytes mutated')


def preserve_savez(save_function,path,**arrays):
    path=Path(path)
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            if set(old.files)!=set(arrays):raise ValueError('Raw archive schema mismatch')
            for name,value in arrays.items():
                saved=old[name]
                if saved.shape!=value.shape or saved.dtype!=value.dtype or original.array_hash(saved)!=original.array_hash(value):
                    raise ValueError('STOP exact raw formal tensor replay mismatch: '+str(path)+'/'+name)
        return
    save_function(path,**arrays)


def validate_population(rows):
    expected={(n,i,s,m) for n in original.metrics.CONDITIONS for i in range(600,1600)
              for s in original.metrics.SNRS for m in original.metrics.METHODS}
    keys=[(r['environment'],r['trajectory_id'],r['snr_db'],r['method']) for r in rows]
    if len(keys)!=80000 or len(set(keys))!=80000 or set(keys)!=expected:
        raise ValueError('STOP aggregate requires exactly all 5000 canonical trajectories')


@contextmanager
def replay_bindings(proof):
    """Only opening/persistence/output metadata adapters; scientific functions unchanged."""
    old_open=original.open_formal;old_journal=original.Journal;old_savez=np.savez
    old_write=original.write;old_out=original.OUT;old_adapter=original.FormalCDL
    old_aggregate=original.metrics.aggregate
    ReplayJournal.instances=[]
    def existing_open():
        original.verify_preparation()
        original.frozen.verify_hashes(proof['resume_source_sha256'])
        verify_prefixes(proof)
        summary=original.frozen.passed_summary(OUT/'t20b_resume_preflight_pytest.txt',1)
        print('Resume same original unseal; exact replay prefix; no new opening. '+summary,flush=True)
        return original.load(original.UNSEAL)
    def journal(path,fields):
        path=Path(path)
        if path.exists() and path.name in {f't20b_formal_{n}.csv' for n in JOURNALS}:
            return ReplayJournal(path,fields)
        return old_journal(path,fields)
    def augmented_write(path,value):
        if Path(path).name=='t20b_formal_numerical_results.json':
            value=dict(value,formal_execution_interrupted_and_resumed=True,
                formal_test_untouched_at_resume=False,original_unseal_preserved=True,
                new_unseal_record_created=False,original_execution_code_unchanged=True,
                recovery_orchestration_added_after_unseal=True,method_modified_after_unseal=False,
                pre_interruption_stage_runtime_available=False,
                runtime_scope='full deterministic completion pass including exact prefix replay; interrupted pass stage timings unavailable',
                recovery_preflight_sha256=original.frozen.sha(PREFLIGHT),recovery_source_sha256=proof['resume_source_sha256'])
        old_write(path,value)
    class SavedOrFrozenCDL(old_adapter):
        def generate(self,trajectory):
            path=OUT/'t20b_formal_data'/f'{self.name}_{trajectory}.npz'
            if path.exists():
                if original.frozen.sha(path)!=proof['existing_raw_sha256'].get(path.relative_to(ROOT).as_posix()):
                    raise ValueError('Existing H bytes differ from recovery preflight')
                h=load_native_h(path)
                if list(h.strides)!=proof['native_h_strides']:raise ValueError('Native H strides drift')
                return h
            return super().generate(trajectory)
    def complete_aggregate(rows,*args,**kwargs):
        validate_population(rows)
        return old_aggregate(rows,*args,**kwargs)
    original.open_formal=existing_open;original.Journal=journal
    np.savez=lambda path,**arrays:preserve_savez(old_savez,path,**arrays)
    original.write=augmented_write;original.OUT=ResumeOutputPath(OUT);original.FormalCDL=SavedOrFrozenCDL
    original.metrics.aggregate=complete_aggregate
    try:yield
    finally:
        original.open_formal=old_open;original.Journal=old_journal;np.savez=old_savez
        original.write=old_write;original.OUT=old_out;original.FormalCDL=old_adapter
        original.metrics.aggregate=old_aggregate


def run():
    if RESULT.exists() or (OUT/'t20b_formal_numerical_results.json').exists():
        raise ValueError('Completed recovery must not be rerun')
    proof=original.load(PREFLIGHT)
    if (OUT/'t20b_formal_STOP.json').exists():raise ValueError('STOP record prohibits recovery')
    with replay_bindings(proof):original.run()
    verify_prefixes(proof)
    journals=[dict(path=j.path.relative_to(ROOT).as_posix(),original_bytes=j.old_size,
                   exact_replayed_rows=j.replayed,new_appended_rows=j.appended) for j in ReplayJournal.instances]
    original.write(RESULT,dict(task='T20B deterministic recovery',all_checks_passed=True,
        recovery_preflight_sha256=original.frozen.sha(PREFLIGHT),original_unseal_sha256=proof['original_unseal_sha256'],
        original_raw_files_verified=len(proof['existing_raw_sha256']),original_raw_file_mismatches=0,
        original_journal_prefix_mismatches=0,journals=journals,
        original_method_and_code_unchanged=True,scientific_rules_unchanged=True,all_5000_fixed_trajectories_completed=True,
        archived_directory_stop_sha256=proof['archived_directory_stop_sha256'],
        scientific_interpretation='original preregistered formal test',existing_H_loaded_without_redraw=True))
    print('Recovery PASS; all original bytes/prefix rows exact; all 5000 fixed trajectories complete',flush=True)


def audit():
    proof=original.load(PREFLIGHT)
    original.verify_preparation();original.frozen.verify_hashes(proof['resume_source_sha256'])
    verify_prefixes(proof)
    if original.frozen.sha(ARCHIVED_STOP)!=proof['archived_directory_stop_sha256']:raise ValueError('Archived STOP changed')
    print('Original unseal/code/raw files/journal prefix integrity PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('preflight','run','audit'))
    stage=parser.parse_args().stage
    {'preflight':preflight,'run':run,'audit':audit}[stage]()
