"""Recovery persistence adapters, exact evidence checks; no changed scientific code."""
import ast
import csv
import inspect
from pathlib import Path

import numpy as np
import pytest

from scripts import resume_formal_test as recovery


def test_exact_journal_prefix_is_preserved_and_suffix_appended(tmp_path):
    path=tmp_path/'journal.csv';fields=('id','energy')
    path.write_bytes(b'id,energy\n1,1.25\n2,2.5\n')
    old=path.read_bytes()
    j=recovery.ReplayJournal(path,fields)
    j.append(dict(id=1,energy=1.25));j.append(dict(id=2,energy=2.5))
    assert path.read_bytes()==old
    j.append(dict(id=3,energy=3.75));j.close()
    assert path.read_bytes()==old+b'3,3.75\n'
    assert j.replayed==2 and j.appended==1


def test_changed_replay_row_stops_without_overwriting_existing_evidence(tmp_path):
    path=tmp_path/'journal.csv';path.write_bytes(b'id,energy\n1,1.25\n')
    old=path.read_bytes();j=recovery.ReplayJournal(path,('id','energy'))
    with pytest.raises(ValueError,match='exact historical'):j.append(dict(id=1,energy=1.26))
    j.close();assert path.read_bytes()==old


def test_header_mismatch_is_rejected(tmp_path):
    path=tmp_path/'journal.csv';path.write_bytes(b'wrong,header\n')
    with pytest.raises(ValueError,match='header'):recovery.ReplayJournal(path,('id','energy'))


def test_existing_raw_zip_bytes_preserved_for_exact_array_replay(tmp_path):
    path=tmp_path/'H.npz';a=np.array([[1+2j,3-4j]],np.complex128)
    np.savez(path,H=a);before=path.read_bytes()
    recovery.preserve_savez(np.savez,path,H=a.copy())
    assert path.read_bytes()==before
    with pytest.raises(ValueError,match='tensor replay'):recovery.preserve_savez(np.savez,path,H=a+1)
    assert path.read_bytes()==before


def test_missing_raw_archive_uses_original_save_without_transform(tmp_path):
    path=tmp_path/'H.npz';a=np.array([[1+2j]],np.complex128)
    recovery.preserve_savez(np.savez,path,H=a)
    with np.load(path) as archive:np.testing.assert_array_equal(archive['H'],a)


def test_recovery_does_not_replace_scientific_functions():
    tree=ast.parse(inspect.getsource(recovery.replay_bindings))
    replaced=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Attribute):replaced.add(ast.unparse(target))
    assert replaced=={'ReplayJournal.instances','original.open_formal','original.Journal','np.savez','original.write','original.OUT','original.FormalCDL','original.metrics.aggregate'}
    assert 'original.run()' in inspect.getsource(recovery.run)


def test_resume_existing_directory_and_retains_files(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'OUT',tmp_path)
    root=tmp_path/'t20b_formal_data';root.mkdir();(root/'original').write_bytes(b'preserved')
    recovery.ResumeOutputPath(root).mkdir()
    assert (root/'original').read_bytes()==b'preserved'
    other=tmp_path/'other';other.mkdir()
    with pytest.raises(FileExistsError):recovery.ResumeOutputPath(other).mkdir()


def test_loaded_h_preserves_native_values_and_strides(tmp_path):
    native=np.arange(40960,dtype=np.float64).reshape(2,2,160,64).astype(np.complex128).transpose(1,0,3,2)
    path=tmp_path/'H.npz';np.savez(path,H=native)
    loaded=recovery.load_native_h(path)
    assert np.array_equal(native,loaded) and native.strides==loaded.strides


def test_incomplete_h_is_rejected(tmp_path):
    path=tmp_path/'H.npz';np.savez(path,H=np.zeros((2,2,64,159),np.complex128))
    with pytest.raises(ValueError,match='incomplete'):recovery.load_native_h(path)


def test_aggregate_population_gate_rejects_missing_and_duplicate_keys():
    m=recovery.original.metrics
    rows=[dict(environment=n,trajectory_id=i,snr_db=s,method=label)
          for n in m.CONDITIONS for i in range(600,1600) for s in m.SNRS for label in m.METHODS]
    recovery.validate_population(rows)
    with pytest.raises(ValueError,match='5000'):recovery.validate_population(rows[:-1])
    rows[-1]=rows[0]
    with pytest.raises(ValueError,match='5000'):recovery.validate_population(rows)


def test_original_run_reaches_generation_with_existing_directory(tmp_path,monkeypatch):
    """Exercise the actual entry path that the first recovery tests missed."""
    run=recovery.original;root=tmp_path/'t20b_formal_data';root.mkdir()
    monkeypatch.setattr(recovery,'OUT',tmp_path)
    monkeypatch.setattr(run,'OUT',recovery.ResumeOutputPath(tmp_path))
    monkeypatch.setattr(run,'open_formal',lambda:{})
    monkeypatch.setattr(run,'verify_preparation',lambda:{'historical_comparison_hashes':[]})
    monkeypatch.setattr(run,'seed_index',lambda:{})
    monkeypatch.setattr(run,'frozen_method',lambda:({},None,None))
    monkeypatch.setattr(run,'cv_family',lambda *_:None)
    class Sentinel:
        def __init__(self,*_):pass
        def generate(self,*_):raise RuntimeError('generation boundary reached')
    class Statistics:
        def prepare(self,*_):return None
    monkeypatch.setattr(run,'FormalCDL',Sentinel)
    monkeypatch.setattr(run,'frozen_statistics',lambda *_:Statistics())
    with pytest.raises(RuntimeError,match='generation boundary reached'):run.run()
    assert run.load(tmp_path/'t20b_formal_STOP.json')['error_type']=='RuntimeError'


def test_actual_recovery_preflight_when_present():
    if not recovery.PREFLIGHT.exists():return
    proof=recovery.original.load(recovery.PREFLIGHT)
    assert proof['original_generated_H_count']==2561 and proof['original_completed_trajectory_count']==2560
    assert proof['all_checks_passed'] and proof['original_frozen_execution_code_unchanged']
    recovery.original.verify_preparation()
    recovery.original.frozen.verify_hashes(proof['resume_source_sha256'])
    recovery.verify_prefixes(proof)


def test_actual_recovery_completed_when_present():
    if not recovery.RESULT.exists():return
    result=recovery.original.load(recovery.RESULT)
    assert result['all_checks_passed'] and result['original_raw_file_mismatches']==result['original_journal_prefix_mismatches']==0
    assert result['original_raw_files_verified']==5122
    expected={'manifest':25610,'trajectory_metrics':40960,'frame_metrics':245760,'input_hashes':51200,'weights':30720}
    for j in result['journals']:
        name=Path(j['path']).stem.replace('t20b_formal_','')
        assert j['exact_replayed_rows']==expected[name]
    recovery.audit()
