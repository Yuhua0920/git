"""Actual selection/calibration tests; no formal target inputs."""
import os,sys
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
from scripts.sp_u12b5b_common import *
firewall()
from scripts.sp_u12b5b_model import *
import pytest

def test_split_is_trajectory_level_disjoint_exact_counts():
 split=read(ROOT/'outputs/sp_u12b5b_preparation_split_manifest.json')
 assert split['unit']=='trajectory' and split['train_trajectories']==1200 and split['validation_trajectories']==300
 assert split['target_IDs_used']==0
 for name,parts in split['origin_partitions'].items():
  assert parts['train_ids']==list(range(400)) and parts['validation_ids']==list(range(400,500))
  assert not set(parts['train_ids'])&set(parts['validation_ids'])

def test_existing_preparation_namespace_and_origin_ids_only():
 cfg=read(CONFIG);proof=read(OUT/'preparation_channel_proofs.json')
 assert cfg['canonical_origins']==['Source/CDL-C-30','CDL-B-30','CDL-C-120']
 assert len(proof)==1500 and all(v['preparation_only'] for v in proof)
 assert {(v['origin'],v['trajectory_id']) for v in proof}=={(n,i) for n in cfg['canonical_origins'] for i in range(500)}

def test_grid_order_and_seeds_exactly_match_B5A():
 cfg=read(CONFIG);fid=check_fidelity();source=fid['frozen_policy']['future_formal_seeds']['registered']
 assert [(r['K'],r['rank']) for r in cfg['candidate_order']]==[(K,L) for K in (1,2,4) for L in (4,8,16)]
 assert len(cfg['candidate_order'])==9
 for a,b in zip(cfg['candidate_order'],source):
  assert a['seed']==b['candidate_training_initialization_seed'] and a['final_seed']==b['full_retraining_initialization_seed']
  assert a['seed']!=a['final_seed']

def test_validation_crn_is_one_frozen_input_for_every_candidate():
 inputs=read(OUT/'input_manifest.json');cfg=read(CONFIG)
 assert sha(OUT/'validation_pilots.npy')==inputs['files']['validation_pilots.npy']
 assert cfg['noise_variances']=={'10':0.9941134822523973,'20':0.09941134822523973}
 pilots=np.load(OUT/'validation_pilots.npy',mmap_mode='r',allow_pickle=False)
 assert pilots.shape==(3,2,100,6,64) and pilots.dtype==np.complex128
 assert len(read(OUT/'validation_noise_registry.json'))==1800

def test_validation_frames4_to9_only_training_all10():
 cfg=read(CONFIG);assert cfg['frames']['validation']==list(range(4,10)) and cfg['frames']['training']==list(range(10))
 truth=np.load(OUT/'validation_truth.npy',mmap_mode='r',allow_pickle=False)
 assert truth.shape==(3,100,6,3968)

def test_linear_pooled_ratio_is_not_mean_of_ratios_or_dB():
 e=np.array([1.,9.]);t=np.array([1.,3.]);actual=pooled_ratio(e,t)
 assert actual==2.5 and actual!=float(np.mean(e/t))
 assert actual!=float(10**np.mean(10*np.log10(e/t)/10))

def test_equal_six_cell_macro_mean_before_log():
 cells=[.01,.01,.01,.01,.01,1.]
 assert macro_score(cells)==sum(cells)/6
 assert macro_score(cells)!=float(10**(np.mean(10*np.log10(cells))/10))

def rows(scores=None):
 return [dict(K=K,rank=L,macro_selection_score=1. if scores is None else scores[j]) for j,(K,L) in enumerate((K,L) for K in (1,2,4) for L in (4,8,16))]

def test_automatic_argmin_no_cost_penalty():
 r=rows([9.,8.,7.,6.,5.,4.,3.,2.,1.]);r[-1]['runtime']=1e9
 assert automatic_select(r)['K']==4 and automatic_select(r)['rank']==16

def test_exact_tie_smaller_K_then_smaller_rank():
 assert (automatic_select(rows())['K'],automatic_select(rows())['rank'])==(1,4)
 r=rows();r[1]['macro_selection_score']=np.nextafter(1.,0.)
 assert automatic_select(r)['rank']==8

def test_incomplete_grid_cannot_select_survivors():
 with pytest.raises(ValueError):automatic_select(rows()[:8])

def test_integer1_vs_float1_0_regression():
 from scripts.sp_u12b5a_cases import known_model
 m=known_model(2);A=np.eye(6)[[0,2,5]];y=np.zeros((1,3),complex);Cn=.37*np.eye(3)
 full=MFAReferenceAdapter(m,A).estimate(y,Cn);hard=m.estimate(y,Cn,A,1)
 assert np.max(abs(full-hard))>1e-6

def test_final_serialization_and_parameter_hashes_calibration(tmp_path):
 from scripts.sp_u12b5a_cases import known_model
 m=known_model(2,D=8,L=2);m.L_all=[-10.,-9.];initial=parameter_hashes(m)
 a=save_model(m,tmp_path,4190123501,'toy_input_hash',initial);loaded,b=load_model(tmp_path)
 assert a['parameter_hash']==b['parameter_hash']==tree(parameter_hashes(loaded))
 assert parameter_validity(loaded)['PASS']
 A=np.eye(8,dtype=complex);y=np.zeros((1,8),complex);Cn=.37*A
 assert np.array_equal(MFAReferenceAdapter(m,A).estimate(y,Cn),MFAReferenceAdapter(loaded,A).estimate(y,Cn))

def test_target_firewall_blocks_before_open():
 for root in ('sp_u12b2b_sealed','sp_u12b3g_next_sealed','t20b_formal_data'):
  with pytest.raises(PermissionError): (ROOT/f'outputs/{root}/forbidden_probe.csv').open('r')

def test_B2_B3_and_B5A_hashes_intact():
 assert sha(p.B2M)==p.B2SHA and sha(p.B3M)==p.B3SHA
 before=read(OUT/'preservation_before.json');assert before['seals']['B2']['PASS'] and before['seals']['B3']['PASS']
 assert sha(FID)==FID_SHA

def test_maxiter_status_and_no_restart_policy():
 cfg=read(CONFIG)
 assert cfg['policy']['maxiter']==100 and cfg['policy']['tol']==1e-4 and cfg['restart_count']==0
 assert 'MAX_ITER_REACHED is valid' in cfg['mentor_amendment']
