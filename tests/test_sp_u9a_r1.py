"""Evidence/provenance and scope gates only; never imports a channel simulator."""
import hashlib,importlib.util,json,csv
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_sp_u9a_r1 import decide,preservation
from sp_u9a_r1_data import FIELDS,MANDATORY_QUERIES,OZ_FIELDS

@pytest.fixture(scope='module')
def ledger():return json.loads((ROOT/'outputs/sp_u9a_r1_source_ledger.json').read_text(encoding='utf-8'))
@pytest.fixture(scope='module')
def records(ledger):return {s['id']:s for s in ledger['sources']}

def test_frozen_science_manuscript_and_old_audits_preserved():
 audit=preservation()
 assert audit['checked_files']==728
 assert audit['all_unchanged'],audit['changed_files']

def test_actual_source_files_and_pdfs_match_recorded_hashes(ledger):
 for row in ledger['retained_source_files']:
  p=Path(row['path']);assert p.is_file()
  assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 for s in ledger['sources']:
  if s['local_pdf']:
   p=Path(s['local_pdf']);assert p.read_bytes().startswith(b'%PDF-')
   assert hashlib.sha256(p.read_bytes()).hexdigest()==s['local_pdf_sha256']
   assert isinstance(s['pdf_pages'],int) and s['pdf_pages']>0

def test_missing_original_methods_stay_unknown(records):
 for id in ('P4','P5'):
  s=records[id]
  assert s['full_text_status']=='FULL_TEXT_UNAVAILABLE'
  assert all(v['status']=='UNKNOWN' for v in s['fingerprint'].values())
  for f in ('source_anchor','correction_only_lowrank','unrestricted_coordinates','cross_pilot_validation','online_combination_weight_update','current_pilots','history'):
   assert s['ozdemir_fields'][f]['status']=='UNKNOWN'
 assert records['P4']['ozdemir_fields']['subspace_tracking']['status']=='YES'
 assert all(v['status']=='UNKNOWN' for v in records['P5']['ozdemir_fields'].values())

def test_ozdemir_primary_bibliography_not_incomplete_secondary(records):
 for id in ('P4','P5'):
  s=records[id]
  authors=s['crossref']['authors']
  assert len(authors)==3 and authors[-1]['family']=='Arvas'
 assert records['P5']['doi']=='10.1109/WAMIC.2005.1528390'
 assert records['P5']['year']==2005

def test_gui_versions_and_equation_limits_remain_separate(records):
 assert len(records['P6']['authors'])==4 and len(records['P6pre']['authors'])==2
 assert records['P6']['pdf'] is None
 assert records['P6']['full_text_status']=='PARTIAL_PUBLISHED_TEXT'
 assert records['P6']['fingerprint']['unrestricted_affine']['status']=='UNKNOWN'
 assert records['P6pre']['quality']=='FULL_TEXT_PREPRINT'
 assert records['P6pre']['doi'] is None
 assert 'not silently' in json.loads((ROOT/'outputs/sp_u9a_r1_decision.json').read_text(encoding='utf-8'))['source_limit_disclosure'][1]

def test_prior_components_are_not_erased(records):
 for id in ('P1','P2','P3'):
  assert records[id]['fingerprint']['unrestricted_affine']['status']=='NO'
  assert len(records[id]['gmm_fields'])==13
 assert records['LS2010']['fingerprint']['unrestricted_affine']['status']=='YES'
 assert 'temporal' in records['LS2010']['fingerprint']['five_frame_causal']['reason'].lower()
 assert records['Mei']['fingerprint']['cross_pilot']['status']=='YES'
 assert 'full rank' in records['P3']['gmm_fields']['covariance']
 assert 'ARE' in records['P2']['gmm_fields']['temporal_history']

def test_direct_collision_cannot_be_bounded_away_by_source_disclosure():
 fake=[dict(id='P1',severity='DIRECT_COLLISION'),dict(id='P4',severity='STRONG_COMPONENT_OVERLAP')]
 assert decide(fake)['final_readiness']=='NOT_READY_DIRECT_COLLISION'
 assert decide(fake)['R1']=='DIRECT_COLLISION_RISK'
 fake=[dict(id='P4',severity='DIRECT_COLLISION')]
 assert decide(fake)['R2']=='DIRECT_COLLISION_RISK'
 assert decide(fake)['final_readiness']=='NOT_READY_DIRECT_COLLISION'

def test_unavailable_source_is_not_a_detected_direct_collision():
 d=decide([dict(id='P4',severity='STRONG_COMPONENT_OVERLAP')])
 assert d['final_readiness']=='READY_WITH_DISCLOSURES'
 assert d['R2']=='OPEN_SOURCE_LIMIT'
 assert d['R3']=='DISCLOSE_AS_LIMITATION'

def test_required_queries_and_row_level_evidence(ledger,records):
 assert ledger['queries']['mandatory']==MANDATORY_QUERIES
 assert ledger['queries']['total']==45
 for p in (ROOT/'outputs/sp_u9a_r1').glob('web_retrieval*.json'):
  assert json.loads(p.read_text(encoding='utf-8')),f'Empty retrieval archive: {p}'
 with (ROOT/'outputs/sp_u9a_r1_collision_matrix.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 for id in records:
  subset=[r for r in rows if r['work']==id and r['section']=='fingerprint']
  assert {r['field'] for r in subset}==set(FIELDS)
  assert all(r['locator'] and r['detail'] and r['quality'] for r in subset)
 for id in ('P4','P5'):
  assert {r['field'] for r in rows if r['work']==id and r['section']=='Ozdemir'}==set(OZ_FIELDS)

def test_outputs_scope_and_evidence_roles():
 d=json.loads((ROOT/'outputs/sp_u9a_r1_decision.json').read_text(encoding='utf-8'))
 for k in ('method_changes','manuscript_changes','bibliography_changes','new_figures','new_channel_runs','new_bootstrap','new_thresholds'):assert d[k]==0
 assert d['MFA_performance']==d['AMMSE_performance']=='NOT TESTED'
 assert d['evidence_roles']['T20B']=='USED formal; reporting-only firewall'
 assert d['evidence_roles']['U4/U5']=='development'
 assert d['evidence_roles']['U8A']=='post-hoc diagnostic'
 assert d['git_commit'] is False
 for tier in ('conservative_contributions','moderate_contributions'):
  assert 3<=len(d[tier])<=4
  text=' '.join(d[tier]).lower()
  for phrase in ('first','novel mixing','unprecedented','universal','outperforms mfa','outperforms a-mmse'):assert phrase not in text
 for p in (ROOT/'scripts').glob('*sp_u9a_r1*.py'):
  if 'search_capture' in p.name or 'capture_' in p.name:continue
  text=p.read_text(encoding='utf-8')
  assert 'import src.' not in text and 'from src.' not in text
