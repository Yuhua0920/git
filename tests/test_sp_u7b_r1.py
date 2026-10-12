"""R1 regression/invariant tests: synthetic archives only, no scientific CDL calls."""
import copy,hashlib,inspect,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from scripts import sp_u7b_r1_common as c
from scripts import sp_u7b_common as original
from scripts import run_sp_u7b_r1 as runner
from scripts import analyze_sp_u7b_r1 as analysis
from tests.test_sp_u7b import fake_proof,fake_reference

@pytest.fixture(autouse=True)
def no_scientific_CDL(monkeypatch):
    from sionna.phy.channel.tr38901 import CDL
    def forbidden(*a,**kw):raise AssertionError('no scientific CDL-A calls in Phase A-R')
    monkeypatch.setattr(CDL,'__call__',forbidden)

@pytest.fixture
def ledger(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path)
    c.write(c.state_path(),dict(phase='A_IMPLEMENTED',UNSEAL=False,scientific_H_calls=0,matched_ref_H_calls=0,eval_H_calls=0))
    return tmp_path

def h_fixture(i=0):
    # Deliberately non-C strides to exercise actual persisted-layout recovery.
    h=np.empty((2,2,160,64),complex).transpose(0,1,3,2)
    h[:]=i+1+1j*(i+2);return h

def test_actual_counter_defect_2matched_3eval(ledger):
    for group,n in (('MATCHED_REF',2),('EVAL',3)):
        for i in range(n):c.commit_h(group,i,100+i,{'synthetic':True},lambda i=i:h_fixture(i))
    s=c.read(c.state_path());r=c.reconcile(strict=True,expected=(2,3))
    assert s['scientific_H_calls']==5 and s['matched_ref_H_calls']==2 and s['eval_H_calls']==3
    assert r['witness_count']==5 and r['archive_count']==5

def test_actual_process_crash_reload_no_redraw(ledger):
    for i in range(2):c.commit_h('MATCHED_REF',i,100+i,{'synthetic':True},lambda i=i:h_fixture(i))
    code="""from pathlib import Path
import sys,numpy as np
from scripts import sp_u7b_r1_common as c
c.OUT=Path(sys.argv[1])
def simulated_exit(**kwargs):raise SystemExit(77)
c.reconcile=simulated_exit
c.commit_h('EVAL',0,300,{'synthetic':True},lambda:np.full((2,2,64,160),3+4j,dtype=np.complex128))
"""
    child=subprocess.run([sys.executable,'-c',code,str(ledger)],cwd=c.ROOT,capture_output=True,text=True)
    assert child.returncode==77,child.stderr
    assert c.read(c.state_path())['scientific_H_calls']==2
    c.reconcile(strict=False)
    assert c.read(c.state_path())['scientific_H_calls']==3
    def no_redraw():raise AssertionError('completed trajectory redrawn')
    h,record=c.commit_h('EVAL',0,300,{'synthetic':True},no_redraw)
    assert np.all(h==3+4j);assert c.reconcile(strict=True)['witness_count']==3

def test_noncontiguous_resume_strides_exact(ledger):
    h=h_fixture();c.commit_h('MATCHED_REF',0,12,{'synthetic':True},lambda:h)
    restored,_=c.commit_h('MATCHED_REF',0,12,{'synthetic':True},lambda:pytest.fail('redraw'))
    assert restored.strides==h.strides and np.array_equal(restored,h)

@pytest.mark.parametrize('corruption',['archive_deleted','witness_deleted','archive_changed','witness_ID_changed','seed_resume_changed'])
def test_archive_witness_corruption_fails(ledger,corruption):
    c.commit_h('EVAL',0,123,{'synthetic':True},h_fixture)
    folder=ledger/'calls/EVAL/0000'
    if corruption=='archive_deleted':(folder/'H.npz').unlink()
    elif corruption=='witness_deleted':(folder/'call.json').unlink()
    elif corruption=='archive_changed':
        with (folder/'H.npz').open('ab') as f:f.write(b'corruption')
    elif corruption=='witness_ID_changed':
        w=c.read(folder/'call.json');w['trajectory_id']=1;(folder/'call.json').write_text(json.dumps(w),encoding='utf8')
    else:
        with pytest.raises(RuntimeError):c.commit_h('EVAL',0,999,{'synthetic':True},h_fixture)
        return
    with pytest.raises(RuntimeError):c.reconcile(strict=True)

def test_state_integer_count_alone_not_trusted(ledger):
    c.commit_h('EVAL',0,123,{'synthetic':True},h_fixture)
    state=c.read(c.state_path());state['scientific_H_calls']=0;c.replace_state(state)
    with pytest.raises(RuntimeError,match='counter mismatch'):c.reconcile(strict=True)

def test_publication_snapshot_never_bypasses_phase_hash_check(ledger):
    c.commit_h('EVAL',0,123,{'synthetic':True},h_fixture)
    with (ledger/'calls/EVAL/0000/H.npz').open('ab') as f:f.write(b'corrupt')
    c.commit_h('EVAL',1,124,{'synthetic':True},lambda:h_fixture(1))
    with pytest.raises(RuntimeError):c.transition('A_IMPLEMENTED','A_FROZEN')

def test_missing_cached_archive_never_redrawn(ledger):
    c.commit_h('EVAL',0,123,{'synthetic':True},h_fixture)
    (ledger/'calls/EVAL/0000/H.npz').unlink()
    with pytest.raises(RuntimeError):c.commit_h('EVAL',0,123,{'synthetic':True},lambda:pytest.fail('redraw'))

def test_phase_transition_requires_reconciled_counts(ledger):
    c.commit_h('EVAL',0,123,{'synthetic':True},h_fixture)
    state=c.read(c.state_path());state['eval_H_calls']=0;c.replace_state(state)
    with pytest.raises(RuntimeError):c.transition('A_IMPLEMENTED','A_FROZEN')

def test_atomic_pair_publication_crash(ledger,monkeypatch):
    def fail(*a,**kw):raise RuntimeError('synthetic failure before witness fsync')
    monkeypatch.setattr(c,'write',fail)
    with pytest.raises(RuntimeError):c.commit_h('EVAL',0,123,{'synthetic':True},h_fixture)
    assert not (ledger/'calls/EVAL/0000').exists()
    with pytest.raises(RuntimeError,match='incomplete call transaction'):c.reconcile()

def test_old_raw_access_machine_rejected():
    with pytest.raises(PermissionError,match='invalidated-run raw quarantine'):
        np.load(c.OLD/'evaluation/raw_0000.npz',allow_pickle=False)

def test_old_H_access_machine_rejected():
    with pytest.raises(PermissionError):np.load(c.OLD/'matched_reference/H_0000.npz',allow_pickle=False)

def test_renamed_and_copied_payload_fingerprint(tmp_path):
    p=tmp_path/'synthetic.npz';np.savez(p,synthetic=np.arange(5))
    st=p.stat();ids={(st.st_dev,st.st_ino)};hashes={c.sha(p)};sizes={st.st_size}
    renamed=tmp_path/'renamed.bin';p.rename(renamed)
    assert c.fingerprint_quarantined(renamed,ids,hashes,sizes)
    copied=tmp_path/'copied.data';shutil.copyfile(renamed,copied)
    assert c.fingerprint_quarantined(copied,set(),hashes,sizes)

def test_old_tree_unchanged():assert c.quarantine_check()['old_tree_unchanged']

def test_new_namespace_derivation_and_zero_overlap():
    audit=c.seed_audit();assert audit['namespace']=='SP_U7B_R1_OOD_CDLA120_V2'
    assert audit['new_scientific_seed_count']==34500 and audit['invalidated_old_scientific_seed_overlap']==0
    row=c.seed_rows()[0];msg='|'.join(str(row[k]) for k in ('namespace','group','trajectory_id','frame_id','role','counter'))
    assert int.from_bytes(hashlib.sha256(msg.encode('ascii')).digest()[:4],'big')==int(row['seed'])
    assert int(c.seed_rows()[-1]['seed'])==764389531

def test_reused_numerical_components_exact_identity():
    assert c.evaluate is original.evaluate and c.infer_pilots is original.infer_pilots
    assert c.communication is original.communication and c.load_deployables is original.load_deployables
    c.verify(c.read(c.ROOT/'outputs/sp_u7a_protected_inputs.json')['hashes'])

def test_archive_run_ID_is_metadata_only(ledger,monkeypatch):
    record=c.archive(ledger/'fake.npz',error=np.ones((2,7)))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('numeric loading forbidden'))
    assert runner.archive_identity(ledger/'fake.npz')==c.RUN_ID
    assert record['members']['error']['shape']==[2,7]

def test_analysis_guard_is_new_root_and_requires_unseal(ledger,monkeypatch):
    monkeypatch.setattr(c,'verify_frozen',lambda:{'hashes':{}})
    c.write(ledger/'unseal.json',dict(UNSEAL=False))
    with pytest.raises(RuntimeError):analysis.assert_unseal()

def test_failure_marking_does_not_need_broken_reconcile(ledger,monkeypatch):
    monkeypatch.setattr(c,'reconcile',lambda **k:pytest.fail('failure state must not depend on reconciliation'))
    runner.invalidate(c.read(c.state_path()),RuntimeError('synthetic'))
    s=c.read(c.state_path());assert s['phase']=='R1_INVALIDATED' and not s['UNSEAL'] and not s['R2_allowed']

def test_complete_15_gate_pipeline_synthetic_only(ledger,monkeypatch):
    """Exercise the actual C/unseal reader chain on a complete synthetic population."""
    real_index=c.seed_index();seed_audit=c.seed_audit()
    condition=dict(c.read(c.ROOT/'outputs/sp_u7b_r1/phase_a_plan.json')['condition'])
    monkeypatch.setattr(c,'seed_index',lambda:real_index)
    monkeypatch.setattr(c,'seed_audit',lambda:seed_audit)
    monkeypatch.setattr(c,'quarantine_check',lambda:dict(old_tree_unchanged=True,old_performance_payload_reads=0))
    monkeypatch.setattr(c,'verify_frozen',lambda:dict(hashes={},R1_FREEZE=True))
    c.write(ledger/'phase_a_plan.json',dict(condition=condition))
    c.write(ledger/'historical_snapshot_at_freeze.json',dict(inventory={},historical_H_hashes=[]))
    c.write(ledger/'freeze.json',dict(hashes={},R1_FREEZE=True))
    proof_hashes={};ref_hashes={}
    raw_arrays=dict(error=np.ones((2,7)),truth=np.ones(2),bit_errors=np.zeros((2,7),np.int64),total_bits=np.full(2,23808,np.int64),
       frame_error=np.ones((2,7,6)),frame_truth=np.ones((2,6)),frame_bit_errors=np.zeros((2,7,6),np.int64),frame_total_bits=np.full((2,6),3968,np.int64),
       coordinates=np.zeros((12,16)),power=np.ones(3))
    for group,n in (('MATCHED_REF',500),('EVAL',1000)):
        cond=dict(condition,matched_group=group)
        folder=ledger/('matched_reference' if group=='MATCHED_REF' else 'evaluation');folder.mkdir()
        for i in range(n):
            pair=ledger/'calls'/group/f'{i:04d}';pair.mkdir(parents=True)
            hr=c.archive(pair/'H.npz',H=np.full((2,2,64,160),i+(0 if group=='EVAL' else 1000)+1,dtype=np.complex128))
            array_hash=hashlib.sha256(f'synthetic-{group}-{i}'.encode()).hexdigest()
            call=dict(group=group,trajectory_id=i,condition=cond,sionna_seed=c.consumed(real_index,group,i)['sionna_seed'],timestamp_utc=c.utc(),
              output_archive_path=hr['path'],archive_SHA256=hr['sha256'],archive=hr,call_success=True,states_generated=160,H_array_sha256=array_hash,export_strides_bytes=[163840,81920,2560,16])
            c.write(pair/'call.json',call)
            proof=fake_proof(i) if group=='EVAL' else fake_reference(i)
            proof.update(condition=cond,H=hr,H_array_sha256=array_hash,channel_seeds=c.consumed(real_index,group,i))
            if group=='EVAL':
                proof['communication_seeds']=[c.consumed(real_index,group,i,j) for j in range(10)]
                proof['communication']=c.archive(folder/f'comm_{i:04d}.npz',bits=np.zeros((10,2,64,16,2),np.uint8),symbols=np.ones((10,2,64,16),complex),unit_noise=np.ones((10,2,64,16),complex))
                proof['raw']=c.archive(folder/f'raw_{i:04d}.npz',**raw_arrays)
            p=folder/f'proof_{i:04d}.json';c.write(p,proof)
            (proof_hashes if group=='EVAL' else ref_hashes)[p.relative_to(ledger).as_posix()]=c.sha(p)
    sr=c.archive(ledger/'matched_reference/statistics.npz',mean_p=np.zeros(64,complex),mean_d=np.zeros(3968,complex),rpp=np.eye(64,dtype=complex),rdp=np.zeros((3968,64),complex),
          train_trajectory_ids=np.arange(500),samples=np.array(5000),W_matched_10=np.zeros((3968,64),complex),W_matched_20=np.zeros((3968,64),complex))
    c.write(ledger/'reference_complete.json',dict(reference_proof_hashes=ref_hashes,statistics=sr,required_finite=True,frames=5000,process_id=1))
    c.write(ledger/'raw_complete.json',dict(evaluation_proof_hashes=proof_hashes,process_id=2,counts=dict(H_archives=1000,communication_archives=1000,trajectory_metrics=14000,frame_metrics=84000,inputs=20000,coordinates=12000)))
    state=c.read(c.state_path());state.update(phase='B_RAW_COMPLETE',scientific_H_calls=1500,matched_ref_H_calls=500,eval_H_calls=1000);c.replace_state(state)
    # integrity installs np.load prohibition; monkeypatch restores it after test.
    monkeypatch.setattr(np,'load',np.load)
    runner.integrity()
    manifest=c.read(ledger/'pre_unseal_manifest.json');assert len(manifest['gates'])==15 and all(g['PASS'] for g in manifest['gates'].values())
    assert not manifest['numeric_outcome_columns_read']
    runner.unseal();assert c.read(ledger/'unseal.json')['UNSEAL']
    assert c.read(c.state_path())['scientific_H_calls']==1500
