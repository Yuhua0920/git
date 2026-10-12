"""Four-phase, fail-closed SP-U7B sealed execution. No aggregate generation in B/C."""
import argparse
import sys
import os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import sp_u7b_r1_common as c
import numpy as np
import subprocess
import time
import zipfile
from src.cdl_benchmark import OfficialCDL,ARRAY,frames,array_hash,StreamingStatistics,observation_geometry,FrozenStatistics

class SealedCDL(OfficialCDL):
    def __init__(self,group,index):
        if group not in ('EVAL','MATCHED_REF'):raise ValueError('namespace boundary')
        super().__init__()  # preserve frozen default-CUDA initialization entry point
        from sionna.phy.channel.tr38901 import CDL
        self.group=group;self.index=index;self.calls=[]
        if c.state_path().exists():
            self.calls=[w['trajectory_id'] for w in c._witnesses(True) if w['group']==group]
            if self.calls!=list(range(len(self.calls))):raise RuntimeError('non-prefix completed-call ledger')
        self.model=CDL(model='A',delay_spread=100e-9,carrier_frequency=3.5e9,ut_array=self.array,bs_array=self.array,
            direction='downlink',ut_orientation=ARRAY['ut_orientation_radians'],bs_orientation=ARRAY['bs_orientation_radians'],
            ut_velocity=[120/3.6,0.,0.],min_speed=120/3.6,max_speed=120/3.6,normalize_delays=True,
            precision='double',device=self.device,spec_version='19.2')

    def generate(self,i):
        if i!=len(self.calls):raise ValueError('one ascending call per trajectory; no retry/redraw')
        import src.cdl_benchmark as adapter
        old=adapter.seeds
        adapter.seeds=lambda trajectory,frame=0:c.consumed(self.index,self.group,trajectory,frame)
        try:h=super().generate(i)
        finally:adapter.seeds=old
        self.calls.append(i)
        return h

    def metadata(self):
        from sionna.phy.channel.tr38901 import models
        audit=c.read(c.ROOT/'outputs/sp_u7a_model_audit.json')
        actual=dict(device=self.device,powers=self.model._powers.cpu().tolist(),delays=self.model._delays.cpu().tolist(),
                    positions=self.array.ant_pos.cpu().tolist(),condition=c.read(c.PROTOCOL)['condition'],model_table_sha256=c.sha(models.parameter_file('CDL-A.json','19.2')))
        expected=audit['profiles']['A']
        # Compare against the table and pinned SDK; no coefficient call occurs here.
        table=c.read(models.parameter_file('CDL-A.json','19.2'))
        db=np.array(table['powers']);powers=10**(db/10);powers/=powers.sum()
        delays=np.array(table['delays'])*100e-9
        if not np.array_equal(np.array(actual['powers']),powers) or not np.array_equal(np.array(actual['delays']),delays):
            # Torch and NumPy power calculation may differ by a few ulps; authoritative
            # metadata is the frozen U7A constructor output, not a tolerance on H.
            if not np.allclose(actual['powers'],powers,rtol=2e-15,atol=0) or not np.array_equal(np.array(actual['delays']),delays):raise ValueError('model table mismatch')
        if audit['device']!=self.device or actual['condition']['sample_rate_hz']!=64*30000:raise ValueError('device/sample-rate drift')
        actual.update(profile='CDL-A',speed_kmh=120,nominal_DS_ns=100,los=False,clusters=23,
          normalize_H=False,states_per_call=160,state_sampling_hz=24000.,frames=10,scored_frames=list(range(4,10)),
          shape=[2,2,64,160],dtype='complex128',state_mapping=[[j,16*j,16*j+15] for j in range(10)],
          array_semantics='frozen PanelArray and orientation; fixed x-axis UT velocity',matched_group=self.group)
        return actual

def progress(group,completed,total,start):
    print(c.utc(),group,'complete',completed,'/',total,'seconds',round(time.perf_counter()-start,2),'RAM',c.resources(),flush=True)

archive=c.archive

def headers(path):
    """NPY headers only. C never reads scientific numeric payloads."""
    result={}
    with zipfile.ZipFile(path) as z:
        for item in z.infolist():
            with z.open(item) as f:
                version=np.lib.format.read_magic(f)
                if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(f)
                elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(f)
                else:raise ValueError('unsupported archive header')
            result[Path(item.filename).stem]=dict(shape=list(shape),dtype=str(dtype))
    return result

def archive_identity(path):
    """Decode only the scalar Unicode identity member; scientific payloads untouched."""
    schema=headers(path)
    member=schema.get('run_id')
    if member!=dict(shape=[],dtype=str(np.asarray(c.RUN_ID).dtype)):raise ValueError('archive run-ID schema')
    with zipfile.ZipFile(path) as z:
        with z.open('run_id.npy') as f:identity=str(np.lib.format.read_array(f,allow_pickle=False))
    if identity!=c.RUN_ID:raise ValueError('archive run-ID value')
    return identity

def allocate():c.allocate_seeds()

def certify_recovery(proofs,rp,index):
    reconciled=c.reconcile(strict=True,verify_archives=True,expected=(500,1000))
    ledger=c._witnesses(True);indexed={(w['group'],w['trajectory_id']):w for w in ledger}
    for p in proofs+rp:
        if p['run_id']!=c.RUN_ID:raise ValueError('proof run-ID mixture')
        w=indexed[p['group'],p['trajectory_id']]
        if w['condition']!=p['condition'] or w['sionna_seed']!=c.consumed(index,p['group'],p['trajectory_id'])['sionna_seed']:
            raise ValueError('ledger condition/seed pairing')
        if w['archive']!=p['H'] or w['H_array_sha256']!=p['H_array_sha256']:raise ValueError('proof/ledger archive identity')
        archive_identity(c.OUT/w['output_archive_path'])
        if p['group']=='EVAL':
            for member in ('communication','raw'):archive_identity(c.OUT/p[member]['path'])
    ref=c.read(c.OUT/'reference_complete.json');archive_identity(c.OUT/ref['statistics']['path'])
    quarantine=c.quarantine_check();seed=c.seed_audit()
    if seed['invalidated_old_scientific_seed_overlap']!=0:raise ValueError('old/new scientific seed overlap')
    return dict(run_id=c.RUN_ID,reconciliation=reconciled,old_run_isolation=quarantine,old_new_scientific_seed_overlap=0,
        ledger_hashes={f"calls/{w['group']}/{w['trajectory_id']:04d}/call.json":c.sha(c.OUT/f"calls/{w['group']}/{w['trajectory_id']:04d}/call.json") for w in ledger},
        H_archive_hashes={w['output_archive_path']:w['archive_SHA256'] for w in ledger},
        all_scientific_archive_hashes={**{w['output_archive_path']:w['archive_SHA256'] for w in ledger},
           **{p[k]['path']:p[k]['sha256'] for p in proofs for k in ('communication','raw')},
           ref['statistics']['path']:ref['statistics']['sha256']})

def invalidate(state,exc):
    # Failure recording must not call a potentially failing reconciliation path.
    state.update(phase='R1_INVALIDATED',UNSEAL=False,updated_utc=c.utc(),OOD_execution_branch='STOP',R2_allowed=False)
    c.replace_state(state)

def prepare():
    if c.state_path().exists():raise RuntimeError('exclusive R1 state required')
    c.OUT.mkdir(exist_ok=True)
    c.quarantine_check()
    protected=c.read(c.ROOT/'outputs/sp_u7a_protected_inputs.json');c.verify(protected['hashes'])
    delivery=c.read(c.ROOT/'outputs/sp_u7a_delivery_manifest.json');c.verify(delivery['file_hashes'])
    seeds=c.seed_audit();history=c.historical_snapshot()
    c.write(c.OUT/'seed_audit.json',seeds);c.write(c.OUT/'historical_snapshot.json',history)
    adapter=SealedCDL('MATCHED_REF',c.seed_index());metadata=adapter.metadata()
    import torch,platform,importlib.metadata
    c.write(c.OUT/'phase_a_plan.json',dict(task='SP-U7B-R1',created_utc=c.utc(),phase='A',scientific_H_calls=0,
        condition=metadata,branch_status={'Theory':'FROZEN','MFA':'STOPPED','A-MMSE':'STOPPED','Offline-statistics':'FROZEN','Pilot-density':'FROZEN','Final LR4':'FROZEN','T20B':'USED / REPORTING-ONLY','SP-U7A':'PASS / E0'},
        output_packing='one metrics NPZ and proof JSON per evaluation trajectory; logical 14 trajectory rows/84 frame rows/20 input rows/12 coordinate rows each',
        reference_isolation='separate reference process/object/path; clean reference arrays only prepare Matched; infer_pilots signature cannot accept truth or Matched',
        resource_plan={'BLAS_threads':1,'torch_cpu_threads':torch.get_num_threads(),'device':adapter.device,'RAM':c.resources(),'free_disk_GiB':__import__('shutil').disk_usage(c.ROOT).free/2**30},
        environment={'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'sionna-no-rt':importlib.metadata.version('sionna-no-rt'),'GPU':torch.cuda.get_device_name(0)},
        realized_power_definition='frames4..9 useful-data received signal energy divided by 6*2*992 samples; SNR=power/sigma2; unit atom energy separately recorded; no rescaling',
        figure_contract={'backend':'python','A':'4 panels: full seven-method trajectory ECDF at10/20; paired Source-LR4 full gain ECDF at10/20; no trimming',
                         'B':'2 panels: pooled BER with descriptive percentile CI allsevenmethods at10/20, linear axes',
                         'replicate_unit':'1000 independent trajectories; allsixframes pooled withintrajectory',
                         'fixed_order':list(c.METHODS),'exports':['PDF','SVG','TIFF','PNG'],'glyph_floor_pt':5,'alignment_tolerance_pt':1.5},
        forbidden_next_work=['U7C','rescue','new profile','method tuning','paper changes','Git commit']))
    c.write(c.state_path(),dict(phase='A_IMPLEMENTED',created_utc=c.utc(),UNSEAL=False,scientific_H_calls=0,matched_ref_H_calls=0,eval_H_calls=0))
    print('PHASE A plan written; scientific H calls=0; seed audit PASS',flush=True)

def freeze():
    if c.read(c.state_path())['phase']!='A_IMPLEMENTED':raise RuntimeError('A phase required')
    junit=c.OUT/'pytest_phase_a.xml'
    import xml.etree.ElementTree as ET
    tree=ET.parse(junit);suites=tree.getroot().iter('testsuite')
    records=[x.attrib for x in suites]
    if not records or any(int(x.get(k,'0')) for x in records for k in ('failures','errors','skipped')):raise RuntimeError('all Phase A pytest checks must pass')
    cases=list(tree.getroot().iter('testcase'))
    old_cases=[x for x in cases if x.get('classname')=='tests.test_sp_u7b']
    new_cases=[x for x in cases if x.get('classname')=='tests.test_sp_u7b_r1']
    if len(old_cases)!=42 or not new_cases:raise RuntimeError('all42 old scoped tests plus R1 tests required')
    if not c.read(c.OUT/'synthetic_end_to_end.json')['PASS']:raise RuntimeError('synthetic engineering check required')
    from scripts.plot_sp_u7b import validate_synthetic
    validate_synthetic(c.OUT/'synthetic_figure_QA')
    protected=c.read(c.ROOT/'outputs/sp_u7a_protected_inputs.json');c.verify(protected['hashes'])
    hashes=dict(protected['hashes'])
    for p in (*c.NEW_SOURCES,'scripts/sp_u7a_statistics.py','scripts/sp_u7a_metadata.py','configs/sp_u7a_protocol.json',
              'outputs/sp_u7a_future_seed_manifest.csv','outputs/sp_u7a_historical_seed_union.json',
              'outputs/sp_u7a_protected_inputs.json','outputs/sp_u7a_model_audit.json','outputs/sp_u7a_delivery_manifest.json',
              'outputs/sp_u7b_r1/seed_manifest.csv','outputs/sp_u7b_r1/sp_u7b_invalidated_quarantine.json','outputs/sp_u7b_r1/sp_u7b_r1_root_cause.json'):
        hashes[p]=c.sha(c.ROOT/p)
    sdkroot=c.ROOT/'.venv-cdl/Lib/site-packages/sionna'
    for p in sdkroot.rglob('*.py'):
        hashes[p.relative_to(c.ROOT).as_posix()]=c.sha(p)
    hashes.update({p:c.sha(c.ROOT/p) for p in c.read(c.PROTOCOL)['deployable']['objects']})
    for p in sdkroot.rglob('CDL-*.json'):hashes[p.relative_to(c.ROOT).as_posix()]=c.sha(p)
    c.seed_audit()
    # Rerun the fresh metadata projection now that Phase A source is final.
    c.write(c.OUT/'historical_snapshot_at_freeze.json',c.historical_snapshot())
    arrays,family,cvs=c.load_deployables()
    prefix_hashes={f'{m}_{s}_{part}':array_hash(getattr(family[m,s,4],part)) for m in ('P','M') for s in c.SNRS for part in ('A','Q')}
    c.write(c.OUT/'freeze.json',dict(created_utc=c.utc(),scientific_H_calls=0,hashes=hashes,
           R1_FREEZE=True,pytest=records,old_scoped_tests=len(old_cases),new_R1_tests=len(new_cases),synthetic_engineering_PASS=True,analysis_frozen=True,source_algorithms_modified=False,rank4_prefix_hashes=prefix_hashes))
    c.transition('A_IMPLEMENTED','A_FROZEN',freeze_sha256=c.sha(c.OUT/'freeze.json'))
    print('PHASE A FROZEN; hashes',len(hashes),'scientific H calls=0',flush=True)

def reference():
    c.verify_frozen();c.guard();index=c.seed_index();adapter=SealedCDL('MATCHED_REF',index)
    c.transition('A_FROZEN','B_REFERENCE_STARTED',first_scientific_start_utc=c.utc())
    folder=c.OUT/'matched_reference';folder.mkdir();g=observation_geometry();acc=StreamingStatistics(g.selected,g.data)
    evidence=[];start=time.perf_counter();condition=adapter.metadata()
    for i in range(500):
        h,hrecord=c.commit_h('MATCHED_REF',i,c.consumed(index,'MATCHED_REF',i)['sionna_seed'],condition,lambda:adapter.generate(i))
        acc.update(frames(h).reshape(10,-1),[i])
        proof=dict(group='MATCHED_REF',trajectory_id=i,channel_seeds=c.consumed(index,'MATCHED_REF',i),H_array_sha256=array_hash(h),
                   H=hrecord,condition=condition,finite=True,coefficient_calls=1,frames=10,states=160,reference_only=True)
        c.write(folder/f'proof_{i:04d}.json',proof);evidence.append(proof)
        if (i+1)%25==0:progress('MATCHED_REF',i+1,500,start)
    stats=acc.freeze();values={k:np.asarray(getattr(stats,k)) for k in ('mean_p','mean_d','rpp','rdp')}
    values.update(train_trajectory_ids=np.array(stats.train_trajectory_ids),samples=np.array(stats.samples))
    for s in c.SNRS:values[f'W_matched_{s}']=stats.prepare(c.VARIANCES[s]).weight
    if any(not np.isfinite(v).all() for v in values.values()):raise ValueError('nonfinite reference')
    record=archive(folder/'statistics.npz',**values)
    c.write(c.OUT/'reference_complete.json',dict(group='MATCHED_REF',trajectories=500,frames=5000,ids=list(range(500)),
        process_id=os.getpid(),completed_utc=c.utc(),condition=condition,statistics=record,required_finite=True,
        reference_proof_hashes={f'matched_reference/proof_{i:04d}.json':c.sha(folder/f'proof_{i:04d}.json') for i in range(500)},
        runtime_seconds=time.perf_counter()-start,resource=c.resources()))
    c.transition('B_REFERENCE_STARTED','B_REFERENCE_FROZEN',reference_sha256=c.sha(c.OUT/'reference_complete.json'))
    print('REFERENCE FROZEN;500 trajectories;5000 clean frames;finite PASS',flush=True)

def matched():
    p=c.OUT/'matched_reference/statistics.npz'
    meta=c.read(c.OUT/'reference_complete.json')
    if c.sha(p)!=meta['statistics']['sha256']:raise ValueError('reference hash drift')
    with np.load(p,allow_pickle=False) as f:
        arrays=[f[k].copy() for k in ('mean_p','mean_d','rpp','rdp')]
        ids=tuple(map(int,f['train_trajectory_ids']));n=int(f['samples'])
        weights={s:f[f'W_matched_{s}'].copy() for s in c.SNRS}
    if ids!=tuple(range(500)) or n!=5000:raise ValueError('reference budget')
    stats=FrozenStatistics(*arrays,ids,n)
    from src.cdl_benchmark import EmpiricalEstimator
    return {s:EmpiricalEstimator(stats,weights[s],c.VARIANCES[s],0.) for s in c.SNRS}

def evaluation():
    c.verify_frozen();c.guard();index=c.seed_index();adapter=SealedCDL('EVAL',index)
    arrays,family,cvs=c.load_deployables();reference_estimators=matched()
    if c.read(c.OUT/'reference_complete.json')['process_id']==os.getpid():raise ValueError('separate reference process required')
    c.transition('B_REFERENCE_FROZEN','B_EVALUATION_STARTED')
    folder=c.OUT/'evaluation';folder.mkdir();condition=adapter.metadata();start=time.perf_counter()
    for i in range(1000):
        h,hr=c.commit_h('EVAL',i,c.consumed(index,'EVAL',i)['sionna_seed'],condition,lambda:adapter.generate(i))
        comm=[c.communication(c.consumed(index,'EVAL',i,j)) for j in range(10)]
        cr=archive(folder/f'comm_{i:04d}.npz',bits=np.stack([x[0] for x in comm]),symbols=np.stack([x[1] for x in comm]),unit_noise=np.stack([x[2] for x in comm]))
        result,evidence=c.evaluate(h,comm,arrays,family,cvs,reference_estimators)
        rr=archive(folder/f'raw_{i:04d}.npz',**result)
        evidence.update(group='EVAL',trajectory_id=i,channel_seeds=c.consumed(index,'EVAL',i,0),communication_seeds=[c.consumed(index,'EVAL',i,j) for j in range(10)],
            H=hr,communication=cr,raw=rr,H_array_sha256=array_hash(h),condition=condition,coefficient_calls=1,frames=10,states=160)
        c.write(folder/f'proof_{i:04d}.json',evidence)
        if (i+1)%10==0:progress('EVAL',i+1,1000,start)
    c.write(c.OUT/'raw_complete.json',dict(trajectories=1000,completed_utc=c.utc(),process_id=os.getpid(),condition=condition,
        counts=dict(H_archives=1000,communication_archives=1000,trajectory_metrics=14000,frame_metrics=84000,inputs=20000,coordinates=12000),
        evaluation_proof_hashes={f'evaluation/proof_{i:04d}.json':c.sha(folder/f'proof_{i:04d}.json') for i in range(1000)},
        runtime_seconds=time.perf_counter()-start,resource=c.resources()))
    c.transition('B_EVALUATION_STARTED','B_RAW_COMPLETE')
    print('PHASE B RAW COMPLETE; no aggregate generated; UNSEAL FALSE',flush=True)

def structural_validate(proofs,reference,expected_n=1000,expected_ref=500):
    """Dataflow witnesses + shapes/counts/identities only, never scientific values."""
    if [p['trajectory_id'] for p in proofs]!=list(range(expected_n)):raise ValueError('missing/duplicate eval identity')
    if [p['trajectory_id'] for p in reference]!=list(range(expected_ref)):raise ValueError('missing/duplicate reference identity')
    for p in proofs:
        if p['group']!='EVAL' or p['frames']!=10 or p['states']!=160 or p['coefficient_calls']!=1:raise ValueError('continuous identity')
        if p['counts']!=dict(trajectory_metrics=14,frame_metrics=84,inputs=20,coordinates=12):raise ValueError('logical counts')
        if p['target_clean_CSI_passed_to_deployables'] or p['matched_reference_passed_to_deployables']:raise ValueError('taint')
        if not p['required_finite'] or not p['paired_inputs_immutable']:raise ValueError('finite/paired')
        if len(p['inputs'])!=20 or len(p['outputs'])!=84 or len(p['windows'])!=12:raise ValueError('proof cardinality')
        ii={(r['snr_db'],r['frame_id']):r for r in p['inputs']}
        if set(ii)!={(s,j) for s in c.SNRS for j in range(10)}:raise ValueError('input identities')
        oo={(r['snr_db'],r['frame_id'],r['method']) for r in p['outputs']}
        if oo!={(s,j,m) for s in c.SNRS for j in range(4,10) for m in c.METHODS}:raise ValueError('output identities')
        for j in range(10):
            if any(ii[10,j][k]!=ii[20,j][k] for k in ('H_sha256','bits_sha256','symbols_sha256','unit_noise_sha256')):raise ValueError('SNR pairing')
        for r in p['outputs']:
            inp=ii[r['snr_db'],r['frame_id']]
            if r['shared_Y_sha256']!=inp['Y_sha256'] or r['shared_pilot_sha256']!=inp['pilot_sha256']:raise ValueError('method pairing')
        for r in p['windows']:
            if r['frames']!=list(range(r['frame_id']-4,r['frame_id']+1)) or r['frame_id'] not in range(4,10):raise ValueError('causal windows')
            if r['rank_P']!=4 or r['rank_M']!=4 or r['clipped'] or r['regularized'] or not r['solve_success'] or not r['dense_lr4_same_c']:raise ValueError('rank/fallback')
    for p in reference:
        if p['group']!='MATCHED_REF' or not p['reference_only'] or not p['finite'] or p['coefficient_calls']!=1 or p['frames']!=10 or p['states']!=160:raise ValueError('reference isolation/completeness')
    return True

def integrity():
    freeze=c.verify_frozen();c.guard('integrity')
    def forbid_numeric_npz(*args,**kwargs):raise PermissionError('C: no numeric scientific archive loading')
    np.load=forbid_numeric_npz
    if c.read(c.state_path())['phase']!='B_RAW_COMPLETE':raise RuntimeError('complete raw generation required')
    ref=c.read(c.OUT/'reference_complete.json');raw=c.read(c.OUT/'raw_complete.json')
    c.verify({str((c.OUT/p).relative_to(c.ROOT)):h for p,h in {**ref['reference_proof_hashes'],**raw['evaluation_proof_hashes']}.items()})
    proofs=[c.read(c.OUT/f'evaluation/proof_{i:04d}.json') for i in range(1000)]
    rp=[c.read(c.OUT/f'matched_reference/proof_{i:04d}.json') for i in range(500)]
    structural_validate(proofs,rp)
    condition=c.read(c.OUT/'phase_a_plan.json')['condition']
    for p in proofs+rp:
        actual=dict(p['condition']);expected=dict(condition);actual.pop('matched_group');expected.pop('matched_group')
        if actual!=expected:raise ValueError('exact condition')
    index=c.seed_index();seed=c.seed_audit()
    for p in proofs+rp:
        group=p['group'];i=p['trajectory_id']
        if p['channel_seeds']!=c.consumed(index,group,i,0):raise ValueError('consumed channel seeds')
        if group=='EVAL' and p['communication_seeds']!=[c.consumed(index,group,i,j) for j in range(10)]:raise ValueError('consumed communication seeds')
    historical=c.read(c.OUT/'historical_snapshot_at_freeze.json')
    # Byte-only inventory delta; any external research-state change blocks unseal.
    for path,record in historical['inventory'].items():
        p=c.ROOT/path
        if not p.exists() or p.stat().st_size!=record['size'] or p.stat().st_mtime_ns!=record['mtime_ns']:raise ValueError('historical metadata changed during sealed run: '+path)
    hs={p['H_array_sha256'] for p in proofs+rp}
    if len(hs)!=1500 or hs&set(historical['historical_H_hashes']):raise ValueError('H identity overlap')
    for p in proofs+rp:
        for label in ('H','communication','raw') if p['group']=='EVAL' else ('H',):
            r=p[label];path=c.OUT/r['path']
            if not path.exists() or path.stat().st_size!=r['bytes'] or c.sha(path)!=r['sha256'] or headers(path)!=r['members']:raise ValueError('raw file hash/header/completeness')
            expected={'H':{'H':([2,2,64,160],'complex128')},
                'communication':{'bits':([10,2,64,16,2],'uint8'),'symbols':([10,2,64,16],'complex128'),'unit_noise':([10,2,64,16],'complex128')},
                'raw':{'error':([2,7],'float64'),'truth':([2],'float64'),'bit_errors':([2,7],'int64'),'total_bits':([2],'int64'),
                       'frame_error':([2,7,6],'float64'),'frame_truth':([2,6],'float64'),'frame_bit_errors':([2,7,6],'int64'),'frame_total_bits':([2,6],'int64'),
                       'coordinates':([12,16],'float64'),'power':([3],'float64')}}[label]
            expected['run_id']=([],str(np.asarray(c.RUN_ID).dtype))
            if r['members']!={k:dict(shape=v[0],dtype=v[1]) for k,v in expected.items()}:raise ValueError('expected archive schema drift')
    r=ref['statistics'];path=c.OUT/r['path']
    if c.sha(path)!=r['sha256'] or headers(path)!=r['members'] or not ref['required_finite'] or ref['frames']!=5000 or ref['process_id']==raw['process_id']:raise ValueError('reference statistics/process isolation')
    expected_stats={'mean_p':([64],'complex128'),'mean_d':([3968],'complex128'),'rpp':([64,64],'complex128'),'rdp':([3968,64],'complex128'),
                    'train_trajectory_ids':([500],'int64'),'samples':([],'int64'),'W_matched_10':([3968,64],'complex128'),'W_matched_20':([3968,64],'complex128')}
    expected_stats['run_id']=([],str(np.asarray(c.RUN_ID).dtype))
    if r['members']!={k:dict(shape=v[0],dtype=v[1]) for k,v in expected_stats.items()}:raise ValueError('reference archive schema drift')
    if len(list((c.OUT/'evaluation').glob('*.npz')))!=2000 or len(list((c.OUT/'calls/MATCHED_REF').glob('*/H.npz')))!=500 or len(list((c.OUT/'calls/EVAL').glob('*/H.npz')))!=1000:raise ValueError('extra/missing archives')
    recovery=certify_recovery(proofs,rp,index)
    gate_names=c.read(c.PROTOCOL)['pre_unseal']['gates']+['state_witness_archive_reconciliation','invalidated_run_isolation']
    evidence={
      'exact_condition':'all1500 per-trajectory metadata match Phase A constructor/table/SDK/device; one160state call',
      'trajectory_count':'1000 exact EVAL IDs0..999 and500 MATCHED_REF IDs0..499',
      'seed_uniqueness':'34501 immutable unique role seed values; consumed maps exact',
      'no_historical_overlap':'frozen831190-value union+fresh seed-column/AST audit; historical inventory unchanged; 1500 unique H hashes disjoint historical H manifests',
      'method_artifact_hashes':'six frozen NPZ byte hashes and factor prefix paths verified before B and C',
      'frozen_code_hashes':str(len(freeze['hashes']))+' frozen source/SDK/protocol/seed/artifact files verified',
      'no_target_clean_CSI_in_deployables':'pilot-only signature; synthetic taint tests; every12000 runtime boundary witness false for cleanCSI',
      'matched_reference_isolated':'separate process IDs/object/signature/path and independent seeds; 5000frame statistics frozen before EVAL',
      'all_trajectories_complete':'all1000 complete10frameH/comm and84 method-frame outputs each; no dropping',
      'causal_windows':'12000 exact j-4..j witnesses,4<=j<=9',
      'finite_outputs':'all prediction,data-ZF,normal-equation,energy flags checked during B; C reads flags only',
      'no_rank_or_fallback_drift':'rank4 prefixes; original real solve; no pinv/ridge/clip/recompression; Dense/LR4 same c object',
      'raw_file_completeness':'all3501 scientific NPZ byte hashes and NPY headers verified; logical counts exact'}
    evidence.update(state_witness_archive_reconciliation='state=ledger=archive1500; MATCHED_REF500/EVAL1000; every identity/seed/hash verified',invalidated_run_isolation='old tree unchanged; old raw decoding blocked; old/new scientific seeds disjoint; no pooling')
    manifest=dict(created_utc=c.utc(),UNSEAL=False,verdict='PASS',gates={k:dict(PASS=True,evidence=evidence[k]) for k in gate_names},
       logical_counts=raw['counts'],reference_trajectories=500,reference_clean_frames=5000,
       proof_hashes={**ref['reference_proof_hashes'],**raw['evaluation_proof_hashes']},statistics_sha256=r['sha256'],
       freeze_sha256=c.sha(c.OUT/'freeze.json'),all_code_hashes=freeze['hashes'],numeric_outcome_columns_read=False,
       seed_audit=seed,H_unique=1500,historical_H_overlap=0,recovery_integrity=recovery)
    c.write(c.OUT/'pre_unseal_manifest.json',manifest)
    c.transition('B_RAW_COMPLETE','C_ALL_PASS',pre_unseal_manifest_sha256=c.sha(c.OUT/'pre_unseal_manifest.json'))
    print('PHASE C:15/15 PASS; numeric outcome columns read=0; UNSEAL remains FALSE',flush=True)

def unseal():
    frozen=c.verify_frozen();s=c.read(c.state_path());p=c.read(c.OUT/'pre_unseal_manifest.json')
    expected=set(c.read(c.PROTOCOL)['pre_unseal']['gates'])|{'state_witness_archive_reconciliation','invalidated_run_isolation'}
    if s['phase']!='C_ALL_PASS' or p['run_id']!=c.RUN_ID or p['verdict']!='PASS' or not all(v['PASS'] is True for v in p['gates'].values()) or set(p['gates'])!=expected or s['pre_unseal_manifest_sha256']!=c.sha(c.OUT/'pre_unseal_manifest.json'):raise RuntimeError('unseal gate denied')
    c.write(c.OUT/'unseal.json',dict(UNSEAL=True,timestamp_utc=c.utc(),pre_unseal_manifest_sha256=c.sha(c.OUT/'pre_unseal_manifest.json'),all_code_hashes=frozen['hashes']))
    c.transition('C_ALL_PASS','D_UNSEALED',UNSEAL=True,data_use='USED validation evidence, permanently unavailable for development')
    print('UNSEAL TRUE; all15 gates PASS',flush=True)

def synthetic():
    """Engineering only: analytic synthetic H, never Sionna coefficient generator."""
    arrays,family,cvs=c.load_deployables();idx=np.arange(2*2*64*160).reshape(2,2,64,160)
    h=(np.sin(idx*.017)+1j*np.cos(idx*.019)).astype(np.complex128)
    comm=[c.communication(dict(bits_seed_tx0=42+j*3,bits_seed_tx1=43+j*3,noise_seed=44+j*3)) for j in range(10)]
    from src.cdl_benchmark import EmpiricalEstimator
    g=observation_geometry();acc=StreamingStatistics(g.selected,g.data);acc.update(frames(h).reshape(10,-1),[0]);stats=acc.freeze()
    ref={s:stats.prepare(c.VARIANCES[s]) for s in c.SNRS}
    result,proof=c.evaluate(h,comm,arrays,family,cvs,ref)
    if result['frame_error'].shape!=(2,7,6) or proof['counts']['frame_metrics']!=84:raise ValueError('synthetic count')
    c.write(c.OUT/'synthetic_end_to_end.json',dict(PASS=True,fixture='analytic sin/cos H; engineering seeds only, not candidate/manifest scientific trajectories',
        scientific_H_calls=0,raw_shapes={k:list(v.shape) for k,v in result.items()},complete_seven_method_ZF=True,
        target_CSI_taint_tests='pytest',aggregate_scientific_outcomes_generated=False))
    print('synthetic end-to-end PASS; scientific H calls=0',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['allocate','prepare','synthetic','freeze','reference','evaluate','integrity','unseal']);args=parser.parse_args()
    try:globals()[{'evaluate':'evaluation'}.get(args.phase,args.phase)]()
    except Exception as exc:
        if c.state_path().exists():
            s=c.read(c.state_path())
            if s['phase'].startswith(('B_','C_')):
                c.write(c.OUT/'sealed_failure.json',dict(status='R1_INVALIDATED',UNSEAL=False,timestamp_utc=c.utc(),
                    phase=s['phase'],exception_type=type(exc).__name__,message=str(exc),aggregate_outcomes_accessed=False,
                    stop=True,no_patch_continue=True))
                invalidate(s,exc)
        raise

if __name__=='__main__':main()
