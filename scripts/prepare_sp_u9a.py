"""Read-only source packet and preservation baseline. No scientific computation."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/sp_u9a'
OUT.mkdir(exist_ok=True)
SOURCE_NAMES = [
 'outputs/t20b_formal_results.json','outputs/t20b_formal_numerical_results.json',
 'outputs/t20b_formal_nmse.csv','outputs/t20b_formal_bootstrap.csv','outputs/t20b_formal_ber.csv',
 'outputs/t20b_formal_unseal.json','outputs/t20b_formal_equivalence_audit.json',
 'outputs/sp_u2a_theory_validation.json','outputs/sp_u2a_delivery_manifest.json',
 'outputs/sp_u2b_analysis.json','outputs/sp_u2b_table3_hypothesis_audit.csv',
 'outputs/sp_u2c_analysis.json','outputs/sp_u2c_table3_hypothesis_audit.csv',
 'outputs/sp_u2d_analysis.json','outputs/sp_u2d_table4_hypotheses.csv',
 'outputs/sp_u3a_baseline_gate.json','outputs/sp_u3b_resource_gate.json',
 'outputs/sp_u3b_r1_decision.json','outputs/sp_u3b_r1_medium_gate.json','outputs/sp_u3b_r1_A_gate.json',
 'outputs/sp_u4a_analysis.json','outputs/sp_u4b_analysis.json','outputs/sp_u4b_summary.csv',
 'outputs/sp_u5a_analysis.json','outputs/sp_u5b_analysis.json',
 'outputs/sp_u6a_decision.json','outputs/sp_u6a_resource_gate.json','outputs/sp_u6a_ammse_version_ledger.json',
 'outputs/sp_u6a_source_freeze.json','outputs/sp_u7b_r1/analysis.json','outputs/sp_u7b_r1/table_A.csv',
 'outputs/sp_u7b_r1/table_B.csv','outputs/sp_u7b_r1/freeze.json','outputs/sp_u7b_r1/unseal.json',
 'outputs/sp_u7b_r1/final_verification.json','outputs/sp_u7b_r1/delivery_manifest.json',
 'outputs/sp_u8a/analysis.json','outputs/sp_u8a/decision.json','outputs/sp_u8a/delivery_manifest.json',
 'docs/sp_u2a_prior_art_amendment.md','docs/sp_u2a_equation_inventory.md',
 'docs/sp_u2a_coordinate_theory.md','docs/sp_u2a_assumption_audit.md',
 'docs/sp_u1_signal_processing_venue_profile.md',
 'configs/sp_u4a_protocol.json','configs/sp_u5a_protocol.json','configs/sp_u5b_protocol.json',
]

def digest(p):
 return hashlib.sha256(p.read_bytes()).hexdigest()

def prune(value):
 # Remove bookkeeping trace bulk only. Scientific values remain unchanged.
 if isinstance(value, dict):
  return {k:prune(v) for k,v in value.items() if k not in {
   'access','read_access','read_log','access_log','formal_artifact_sha256','source_sha256',
   'artifact_hashes','preserved_files','frozen_files','files','baseline_files'}}
 if isinstance(value,list): return [prune(v) for v in value]
 return value

def main():
 source_paths = [ROOT / n for n in SOURCE_NAMES]
 source_paths += sorted((ROOT/'paper').rglob('*.tex'))
 inventory = []
 for p in source_paths:
  rel=p.relative_to(ROOT).as_posix()
  inventory.append({'path':rel,'absolute_path':str(p),'exists':p.exists(),
   'sha256':digest(p) if p.exists() else None})
 preserved={}
 for folder in ('paper','src','configs','scripts','tests','docs'):
  for p in (ROOT/folder).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and 'sp_u9a' not in p.name:
    preserved[p.relative_to(ROOT).as_posix()]=digest(p)
 for row in inventory:
  if row['exists']: preserved[row['path']]=row['sha256']
 (OUT/'preservation_before.json').write_text(json.dumps(preserved,indent=2),encoding='utf-8')
 (OUT/'source_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
 blocks=[]
 for row in inventory:
  if not row['exists']: continue
  p=ROOT/row['path']
  # Delivery manifests and gates remain available in allowlist, not duplicated in text.
  if any(x in p.name for x in ('delivery_manifest','final_verification','freeze','unseal','resource_gate')):continue
  text=p.read_text(encoding='utf-8-sig')
  if p.suffix=='.json': text=json.dumps(prune(json.loads(text)),ensure_ascii=False,indent=2)
  blocks.append('SOURCE '+row['path']+' SHA256 '+row['sha256']+'\n'+text)
 packet='SP-U9A immutable scientific source packet.\nAssessment boundary: existing unfinished manuscript plus frozen scientific/engineering outputs. No manuscript edits, experiments, rebootstrap or tuning. Existing outputs are reporting-only. Old invalidated U7B outcomes are excluded. No review/concern ledger is included. Review independently under Signal Processing scope; Nature-style axes are generic assessment aids, not this journal\'s acceptance policy.\n\n'+'\n\n'.join(blocks)
 (OUT/'immutable_packet.txt').write_text(packet,encoding='utf-8')
 briefs={'reviewer1':'Originality, positioning and relevance',
 'reviewer2':'Technical validity, statistics and information fairness',
 'reviewer3':'Readability, claim architecture and practical scope'}
 setup={'packet_sha256':digest(OUT/'immutable_packet.txt'),'source_inventory_sha256':digest(OUT/'source_inventory.json'),
 'preassigned_emphases':briefs,'mutual_blindness':'fork_turns=none; same packet; no peer reports or root synthesis supplied',
 'git_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'missing_sources':[r['path'] for r in inventory if not r['exists']]}
 (OUT/'review_setup.json').write_text(json.dumps(setup,indent=2),encoding='utf-8')
 print(json.dumps({'sources':len(inventory),'preserved':len(preserved),'packet_bytes':len(packet.encode()),**setup},indent=2))

if __name__=='__main__':main()
