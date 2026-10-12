"""Editorial deliverable checks; no scientific experiment imports or execution."""
import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/sp_u9a'
def load(name):return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def test_claim_schema_and_mandatory_families():
 rows=load('outputs/sp_u9a_claim_evidence_matrix.json')
 expected={'Claim ID','Candidate claim','Allowed wording','Forbidden wording','Evidence level','Evidence source','Condition','Metric','Primary/secondary','Independent unit','Sample size','Statistical support','Known counter-evidence','Boundary/limitation','Reviewer attack','Needed manuscript location','Figure/table support','Status'}
 assert {f'C{i}' for i in range(1,13)} <= {r['Claim ID'] for r in rows}
 assert len({r['Claim ID'] for r in rows})==len(rows)
 for row in rows:
  assert set(row)==expected and all(str(v).strip() for v in row.values())
  assert row['Status'] in {'SUPPORTED','SUPPORTED_WITH_LIMITS','MIXED','REJECTED','NOT_TESTED'}

def test_csv_json_lossless_rendering():
 for filename,data in [('outputs/sp_u9a_claim_evidence_matrix.csv',load('outputs/sp_u9a_claim_evidence_matrix.json'))]:
  with (ROOT/filename).open(encoding='utf-8-sig') as f:assert list(csv.DictReader(f))==data

def test_all_risk_fields_and_closed_action_vocabulary():
 rows=list(csv.DictReader((ROOT/'outputs/sp_u9a_reviewer_risk_register.csv').open(encoding='utf-8-sig')))
 assert {r['Risk ID'] for r in rows}=={f'R{i}' for i in range(1,19)}
 for r in rows:
  assert all(str(v).strip() for v in r.values())
  assert r['Action'] in {'ANSWER_WITH_EXISTING_EVIDENCE','DISCLOSE_AS_LIMITATION','PAPER_RISK_OPEN'}
  assert r['Severity'] in {'HIGH','MEDIUM','LOW'}
 decision=load('outputs/sp_u9a_decision.json')
 high=[r['Risk ID'] for r in rows if r['Severity']=='HIGH' and r['Action']=='PAPER_RISK_OPEN']
 assert decision['HIGH_PAPER_RISK_OPEN']==high
 if high:assert decision['verdict']=='NOT_READY'

def test_frozen_sources_and_manuscript_untouched():
 before=load('outputs/sp_u9a/preservation_before.json')
 assert any(p.startswith('paper/') for p in before)
 assert all((ROOT/p).is_file() and digest(ROOT/p)==value for p,value in before.items())
 for row in load('outputs/sp_u9a/full_source_inventory.json'):
  assert row['exists'] and digest(ROOT/row['path'])==row['sha256']

def test_review_reports_immutable_and_consensus_not_votes():
 packet=digest(OUT/'immutable_packet.txt')
 for i in (1,2,3):
  lock=load(f'outputs/sp_u9a/reviewer{i}_lock.json')
  assert lock['packet_sha256']==packet
  files=lock['files']
  if isinstance(files,dict):files=[{'path':p if '/' in p else f'outputs/sp_u9a/{p}','sha256':h} for p,h in files.items()]
  for row in files:assert digest(ROOT/row['path'])==row['sha256']
 synthesis=load('outputs/sp_u9a/reviewer_synthesis.json')
 assert synthesis['reports_frozen_before_comparison']
 assert synthesis['consensus_blocking_concerns']==[]
 for item in synthesis['other_consensus_major_concerns']:
  assert len({x.split('-')[0] for x in item['ids']})>=2

def test_reporting_only_and_invalidated_exclusion():
 provenance=load('outputs/sp_u9a_evidence_provenance.json')
 assert all(p['whether reusable for tuning'] is False for p in provenance)
 records={p['task ID']:p for p in provenance}
 assert records['T20B']['used/development/formal status']=='USED formal'
 assert records['SP-U7B-R1']['used/development/formal status']=='USED validation'
 assert records['SP-U8A']['evidence class']=='D'
 assert records['old SP-U7B']['used/development/formal status']=='NEVER SCIENTIFIC EVIDENCE'
 claims={r['Claim ID']:r for r in load('outputs/sp_u9a_claim_evidence_matrix.json')}
 assert claims['C11-P']['Status']=='NOT_TESTED'
 assert claims['C12-N']['Status']=='REJECTED'
 assert 'old SP-U7B' not in ''.join(r['Evidence source'] for r in claims.values())

def test_script_does_not_execute_scientific_or_manuscript_operations():
 allowed={'csv','hashlib','json','pathlib','subprocess'}
 for name in ('prepare_sp_u9a.py','build_sp_u9a.py'):
  tree=ast.parse((ROOT/'scripts'/name).read_text(encoding='utf-8'))
  imported=set()
  for node in ast.walk(tree):
   if isinstance(node,ast.Import):imported.update(x.name.split('.')[0] for x in node.names)
   elif isinstance(node,ast.ImportFrom):imported.add(node.module.split('.')[0])
  assert imported <= allowed
 decision=load('outputs/sp_u9a_decision.json')
 for k in ['new_scientific_experiments','new_bootstrap','new_statistical_threshold','method_changes','manuscript_changes','new_figures','new_literature_search']:
  assert decision[k]==0
 assert not decision['git_commit']

def test_no_same_estimand_conflict_suppressed():
 conflicts=load('outputs/sp_u9a/evidence_conflicts.json')
 assert load('outputs/sp_u9a_decision.json')['evidence_conflicts']==len(conflicts)
 for row in conflicts:
  assert row['status']=='EVIDENCE_CONFLICT' and 'value1' in row and 'value2' in row

def test_required_deliverables_and_display_budget():
 for name in ['master_claim_evidence_matrix','reviewer_red_team','manuscript_architecture_audit','figure_table_registry','negative_evidence_ledger','writing_language_rules','invalidated_excluded_evidence']:
  assert (ROOT/f'docs/sp_u9a_{name}.md').stat().st_size>500
 registry=load('outputs/sp_u9a/figure_table_registry.json')
 assert len([r for r in registry if r['ID'] in {f'F{i}' for i in range(1,7)}])==6
 assert len([r for r in registry if r['ID'] in {f'T{i}' for i in range(1,6)}])==5
 assert all((ROOT/path).is_file() for row in registry for path in row['source files'])
