"""Generate bounded literature/positioning artifacts. No channel code or outcomes imported."""
import csv,hashlib,json
from pathlib import Path
from sp_u9a_r1_data import (FIELDS,FINGERPRINT,FINGERPRINT_LOCATORS,SOURCES,GMM_FIELDS,
 OZ_MATRIX,MANDATORY_QUERIES,ACCESS_QUERIES,MODERN_DISCLOSURE,TITLE,SAFE_MISSING,
 CONSERVATIVE,MODERATE,NOVELTY,BLUEPRINT)
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/sp_u9a_r1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write_doc(name,text):(ROOT/'docs'/name).write_text(text.rstrip()+'\n',encoding='utf-8')
def preservation():
 baseline=json.loads((OUT/'preservation_before.json').read_text(encoding='utf-8'))
 changed=[name for name,h in baseline.items() if not (ROOT/name).is_file() or sha(ROOT/name)!=h]
 return dict(checked_files=len(baseline),changed_files=changed,all_unchanged=not changed,baseline_sha256=sha(OUT/'preservation_before.json'))
def decide(sources):
 direct=[s['id'] for s in sources if s['severity']=='DIRECT_COLLISION']
 return dict(final_readiness='NOT_READY_DIRECT_COLLISION' if direct else 'READY_WITH_DISCLOSURES',
  R1='DIRECT_COLLISION_RISK' if any(p in direct for p in ['P1','P2','P3']) else 'CLOSED_BY_POSITIONING',
  R2='DIRECT_COLLISION_RISK' if any(p in direct for p in ['P4','P5']) else 'OPEN_SOURCE_LIMIT',
  R3='DISCLOSE_AS_LIMITATION',direct_collisions=direct)

def main():
 OUT.mkdir(exist_ok=True)
 audit=preservation()
 if not audit['all_unchanged']:raise RuntimeError('Frozen input bytes changed: '+str(audit['changed_files']))
 records=json.loads(json.dumps(SOURCES,ensure_ascii=False))
 meta={r['id']:r for r in json.loads((OUT/'metadata_audit.json').read_text(encoding='utf-8'))}
 for s in records:
  s['local_pdf']=None;s['local_pdf_sha256']=None;s['pdf_pages']=None
  if s['pdf']:
   p=ROOT/s['pdf']
   if not p.read_bytes().startswith(b'%PDF-'):raise ValueError('Invalid PDF: '+str(p))
   from pypdf import PdfReader
   s.update(local_pdf=str(p),local_pdf_sha256=sha(p),pdf_pages=len(PdfReader(p).pages))
  if s['id'] in meta:
   m=meta[s['id']];s['crossref']=dict(url=m['url'],sha256=m['sha256'],title=m['title'],authors=m['authors'],published=m['published'],page=m['page'])
  elif s['id']=='LS2010':
   p=OUT/'crossref_LS2010.json';m=json.loads(p.read_text(encoding='utf-8'))['message']
   s['crossref']=dict(url='https://api.crossref.org/works/10.1109%2FISCAS.2010.5538026',sha256=sha(p),title=m['title'],authors=m['author'],published=m['published'],page=m['page'])
  s['bibliographic_verification']='Primary Crossref + original publisher/author manuscript' if s.get('crossref') else 'Original author arXiv metadata/manuscript; A-MMSE reuses frozen U6A audit'
  s['full_text_status']='FULL_TEXT_UNAVAILABLE' if s['id'] in ('P4','P5') else ('PARTIAL_PUBLISHED_TEXT' if s['id']=='P6' else 'FULL_TEXT_AVAILABLE')
  s['si_downloaded']=False
  s['gmm_fields']=GMM_FIELDS.get(s['id'])
  s['ozdemir_fields']=OZ_MATRIX.get(s['id'])
 rows=[]
 for s in records:
  for f,v in s['fingerprint'].items():
   rows.append(dict(work=s['id'],section='fingerprint',field=f,status=v['status'],detail=v['reason'],locator=v['locator'],quality=s['quality'],severity=s['severity']))
  for f,v in GMM_FIELDS.get(s['id'],{}).items():
   rows.append(dict(work=s['id'],section='GMM_MFA',field=f,status='DOCUMENTED',detail=v,locator=s['locator'],quality=s['quality'],severity=s['severity']))
  for f,v in OZ_MATRIX.get(s['id'],{}).items():
   rows.append(dict(work=s['id'],section='Ozdemir',field=f,status=v['status'],detail=v['reason'],locator=v['locator'],quality=s['quality'],severity=s['severity']))
 with (ROOT/'outputs/sp_u9a_r1_collision_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['work','section','field','status','detail','locator','quality','severity']);w.writeheader();w.writerows(rows)
 evidence_files=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix.lower() in ('.json','.html','.pdf') and p.name not in ('delivery_manifest.json','execution_results.json','integrity_audit.json'):
   evidence_files.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
 ledger=dict(task='SP-U9A-R1',date='2026-10-05',sources=records,
  source_quality_policy='The five task labels identify what was actually read; P6 retains PUBLISHER_ABSTRACT with separately marked publisher method excerpts, not a fabricated full-text label.',
  SI_policy='Default main text only; add specific SI only if original main text delegates a collision-changing detail. No WHY SI NEEDED trigger found; no SI downloaded.',
  queries=dict(mandatory=MANDATORY_QUERIES,targeted_access_and_metadata=ACCESS_QUERIES,total=len(MANDATORY_QUERIES)+len(ACCESS_QUERIES),count_unit='distinct queries; final two P6 queries repeated to repair an empty retrieval archive, not new search scope'),
  search_boundary='Six mandatory works plus bounded exact collision concepts, source access and closest-result checking. No generic surveys or hundreds-paper sweep. Search is closed for this task.',
  screened_not_equation_audited=[dict(title='Pilot Design for Gaussian Mixture Channel Estimation in Massive MIMO',authors='Yujie Gu; Yimin D. Zhang',year=2018,
   url='https://yiminzhang.com/pdf_r/icassp18_pilot.pdf',status='Indexed author-PDF lead; actual download 403; exact DOI not verified in this audit; not used for exact equations or whole-method clearance.')],
  source_limit=dict(P4='Original publisher abstract available; no legal public full text located.',P5='Original metadata only; secondary-hosted abstract; no public full text located.',
   P6='Original publisher prose excerpts available, equation images/full manuscript unavailable. Separate related 2014 author preprint fully read; correspondence not proven.'),
  actual_access_attempts=json.loads((OUT/'access_attempts.json').read_text(encoding='utf-8')),
  downloader_manifests=[dict(path=str(p),sha256=sha(p)) for p in sorted(OUT.rglob('manifest.json'))],
  retained_source_files=evidence_files,
  no_author_messages=True,no_institution_credentials_used=True,no_paywall_bypass=True,
  preservation=audit)
 write_json(ROOT/'outputs/sp_u9a_r1_source_ledger.json',ledger)
 decision=decide(records)
 decision.update(task='SP-U9A-R1',closest_work_1='P3 MFA2023',closest_work_2='P1 GMM2022',
  additional_closest_by_field=dict(fitting='Mei2021',affine_algebra='LS2010',temporal_validation='ET2026 preprint'),
  safe_statement=SAFE_MISSING,recommended_title=TITLE,modern_baseline_disclosure=MODERN_DISCLOSURE,
  source_limit_disclosure=['P4/P5 full text unavailable; method fields UNKNOWN, no priority claim.','P6 published equation constraints unverified; related 2014 author preprint is not silently substituted.'],
  conservative_contributions=CONSERVATIVE,moderate_contributions=MODERATE,
  novelty_scope='Whole frozen interface and evidence within tested conditions; no component-level priority claim.',
  method_changes=0,manuscript_changes=0,bibliography_changes=0,new_figures=0,new_channel_runs=0,new_bootstrap=0,new_thresholds=0,
  MFA_performance='NOT TESTED',AMMSE_performance='NOT TESTED',experiments='FROZEN',
  evidence_roles={'T20B':'USED formal; reporting-only firewall','U7B-R1':'USED sealed validation','U4/U5':'development','U8A':'post-hoc diagnostic'},
  decision_delta='SP-U9A NOT_READY high risks are now bounded by source-grounded positioning and the mentor-frozen disclosure policy. Scientific outcomes unchanged; neither unavailable sources nor untested modern baselines are treated as a detected direct collision.',
  next_action='STOP and await mentor decision on SP-U9B; no manuscript upgrade executed.',git_commit=False,
  integrity=audit,execution_validation='PENDING actual scoped pytest and task-run receipt')
 write_json(ROOT/'outputs/sp_u9a_r1_decision.json',decision)

 lines=['# SP-U9A-R1 nearest-neighbor matrix','',
 'The fingerprint is compared as an entire operational interface. NO means an observed difference in the audited construction, not a claim about every variant of an author\'s work. UNKNOWN is never converted to NO. All source locators and row-level reasons are retained in the CSV and source ledger.','',
 '## Frozen interface','']
 for f,v in FINGERPRINT.items():lines.append(f'- {f}: {v} Code: {FINGERPRINT_LOCATORS[f]}.')
 lines+=['','In the five-frame fit G is 640 x 2, z is 640, and the normal equation is 2 x 2. Each pilot prediction kernel is 16 x 48 and remains dense. Deployment output is 3968, input 64; each correction uses 3968 x 4 and 4 x 64 factors. The pilot-LS observation is Source-centered. No pilot-design modification or OLS covariance theorem is introduced.','',
 '## Whole-method comparison','', '| Work | Source | Prepared directions | Analytical fitting | Unrestricted affine | Cross-pilot | Exact five-frame | Corrections only low rank | No online truth/labels | Severity |','|---|---|---|---|---|---|---|---|---|---|']
 for s in records:lines.append('| '+s['id']+' | '+' | '.join(s['fingerprint'][f]['status'] for f in FIELDS)+' | '+s['severity']+' |')
 for s in records:
  lines+=['',f"## {s['id']} — {s['title']}",'',f"Quality: {s['quality']}. {s['note']}",f"Locator: {s['locator']}"]
  for f,v in s['fingerprint'].items():lines.append(f"- {f}: {v['status']}. {v['reason']}")
 lines+=['','## GMM/MFA thirteen-field audit','']
 for id,fields in GMM_FIELDS.items():
  lines.append('### '+id);lines.append('')
  lines.extend(f'- {f}: {v}' for f,v in fields.items());lines.append('')
 lines+=['Mixture responsibilities and conditional affine LMMSE estimators already cover pilot/observation-dependent statistical mixing. Choosing a reference component can algebraically rewrite a probability mixture as base plus differences, but the probability constraints, component means and posterior objective remain. A reparameterization alone is not a novelty distinction.','',
 '## Affine and selector positioning','',
 'LS2010 already has unrestricted affine K-filter combinations, K-1-coordinate normal equations, temporal windows and base+difference rewriting. Gui combines online-adaptive sparse LMS estimates and adjusts its practical combiner by SG. The separate 2014 author version presents this explicitly. In the 2015 original publisher prose SG is confirmed, but equation-image constraints remain UNKNOWN.','',
 'Mei evaluates noisy leave-one-out predictions and selects a discrete CCF candidate. Therefore cross-prediction/heldout pilot residuals are not a standalone originality claim. Discrete selection, posterior probability mixing and unrestricted continuous coordinates are different interfaces, although all can be represented as linear combinations of candidate outputs.','',
 'ET2026 shares anchors, corrections, heldout validation and windows, but its physical-state anchor, clipped complex scalar alignment, iterative sparse vector correction and action selection do not instantiate the frozen Source/P/M/rank4-operator interface. No performance claim is imported.','',
 'Closest statistical-method works: P3 (#1) and P1 (#2), with P2 in the same top tier. Closest fitting neighbor: Mei; closest affine algebra: LS2010. No DIRECT_COLLISION identified in inspected full methods. P4/P5 remain unresolved at method level; P6 published-version constraints are not cleared.']
 write_doc('sp_u9a_r1_nearest_neighbor_matrix.md','\n'.join(lines))

 lines=['# SP-U9A-R1 Özdemir audit','',
 'Verdict: OPEN_SOURCE_LIMIT. P4/P5 FULL_TEXT_UNAVAILABLE after bounded legal public retrieval. Do not assert that their online variables, objectives or low-rank construction differ from ours.','',
 'Primary Crossref and original IEEE metadata confirm three authors on both records. P4: DOI 10.1109/TWC.2006.04713, TWC 5(10):2675-2678, 2006. P5: DOI 10.1109/WAMIC.2005.1528390, WAMICON 2005:87-91. Do not reuse the incomplete two-author dblp entry. P5 exact calendar day is not asserted.','',
 'P4 original publisher abstract confirms comb-pilot MIMO-OFDM generalized low-rank LMMSE and subspace tracking. It does not identify which object is recursively updated or which rank reduction is used. P5 original publisher abstract is absent; its available abstract transcription is secondary-hosted and cannot ground equations.','',
 '| Field | P4 | P5 |','|---|---|---|']
 for f in OZ_MATRIX['P4']:lines.append(f"| {f} | {OZ_MATRIX['P4'][f]['status']} | {OZ_MATRIX['P5'][f]['status']} |")
 lines+=['','All UNKNOWN cells refer to unavailable full text; the abstract-level YES cells do not imply known update equations. Generic algebra can write an estimator as Ws plus its difference, but this does not prove fixed prepared corrections, causal cross-pilot coordinates or correction-only compression.','',
 'Access log: DOI resolved to IEEE but returned empty 202 content; Crossref PDF URLs returned original publisher metadata HTML rather than PDF; observed actual PDF endpoints returned 418. OpenAlex and Semantic Scholar returned closed/no public PDF. Author/institution exact-title routes yielded bibliographic records only; Medipol author index direct request 403. Syracuse dissertation page provides description only and directs full text to ProQuest subscribers. No institution credential or author-contact permission was available or used. No paywall/captcha bypass attempted.','',
 'Conservative disclosure: Adaptive low-rank MIMO-OFDM LMMSE and subspace tracking predate this work. The exact adaptation variables and rank construction of the two Özdemir works could not be audited from the available public full text. We therefore make no priority claim over that literature.','',
 'Severity labels are observed component overlap only, not exoneration of the missing full methods. R2 stays OPEN_SOURCE_LIMIT. Under the mentor policy this is compatible with READY_WITH_DISCLOSURES, not an automatic NOT_READY or a fictional CLOSED_BY_POSITIONING.']
 write_doc('sp_u9a_r1_ozdemir_audit.md','\n'.join(lines))

 lines=['# SP-U9A-R1 novelty ledger','',SAFE_MISSING,'',
 'This is a bounded audit statement, not proof of global absence or a priority claim. Özdemir unknowns and Gui published-version limits remain disclosed.','']
 for i,n in enumerate(NOVELTY,1):
  lines+=['## Item '+str(i),'']+[f'- {k}: {v}' for k,v in n.items()]+['']
 lines+=['Forbidden claims: first; novel mixing; unprecedented; universal; affine combination itself is new; pilot-dependent conditional LMMSE mixing itself is new; windowed LS itself is new; heldout noisy-pilot prediction itself is new; our receiver outperforms MFA/A-MMSE; MFA failed; A-MMSE failed.','',
 'T20B remains reporting-only and is not used to choose theory, thresholds or method. U2/U4/U5 diagnostics are reused only through frozen evidence roles, not reanalyzed to tune claims. No new scientific data were generated or read to select a method.']
 write_doc('sp_u9a_r1_novelty_ledger.md','\n'.join(lines))

 lines=['# SP-U9A-R1 related-work blueprint','', 'Six paragraphs; no manuscript or bibliography edited. Source IDs resolve to the ledger.','']
 for i,b in enumerate(BLUEPRINT,1):
  lines += [f'## Paragraph {i}','', 'Citations: '+', '.join(b['citations']), 'Main point: '+b['main'],'Transition: '+b['transition'],'']
 lines += ['Modern comparison disclosure: '+MODERN_DISCLOSURE,'No third modern performance baseline is sought or implemented. No almost-zero-cost interface-matched comparator was established by this targeted audit.']
 write_doc('sp_u9a_r1_related_work_blueprint.md','\n'.join(lines))

 lines=['# SP-U9A-R1 contribution language','', '## Conservative','']
 lines.extend(f'{i}. {v}' for i,v in enumerate(CONSERVATIVE,1));lines+=['','## Moderate','']
 lines.extend(f'{i}. {v}' for i,v in enumerate(MODERATE,1))
 lines+=['','## Title decision','',
 'Replace the proposed title "Pilot-Driven Low-Rank Affine LMMSE Adaptation for MIMO-OFDM Channel Estimation" at a future mentor-authorized manuscript step. It combines broad pre-existing labels and may suggest a low-rank whole estimator. No title has been changed in the manuscript.','',
 'Recommended: '+TITLE,'',
 'Optional domain-explicit version: Source-Anchored LMMSE Estimation for MIMO-OFDM with Cross-Pilot Fitting and Low-Rank Corrections.','',
 '## Readiness and disclosure','',
 'READY_WITH_DISCLOSURES. R1=CLOSED_BY_POSITIONING; R2=OPEN_SOURCE_LIMIT; R3=DISCLOSE_AS_LIMITATION. No DIRECT_COLLISION found among audited methods; missing Özdemir methods and Gui exact published-version constraints remain unresolved.','',
 MODERN_DISCLOSURE,'',
 'Use Conservative by default. Moderate emphasizes an implemented interface rather than priority. Neither version asserts statistical mixture, affine LS, temporal adaptation, low rank or cross-prediction as a new isolated principle.','',
 'Decision delta: scientific verdicts and frozen data are unchanged. The gate moves from unresolved high risks to bounded positioning under the explicit source-limit and modern-baseline disclosure policy. This is readiness for mentor review, not automatic authorization to run U9B.','',
 'STOP: await mentor review; no paper/bib/figure edit, experiment, rescue, new profile, or Git commit.']
 write_doc('sp_u9a_r1_contribution_language.md','\n'.join(lines))
 write_json(OUT/'integrity_audit.json',audit)
 print(json.dumps(dict(task='SP-U9A-R1',sources=len(records),matrix_rows=len(rows),queries=ledger['queries']['total'],preserved=audit['checked_files'],verdict=decision['final_readiness']),ensure_ascii=False))
 return decision

if __name__=='__main__':main()
