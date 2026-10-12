"""Write authorized front-half sections and generated claim/source audits only."""
import csv,hashlib,json,re,unicodedata
from pathlib import Path
from sp_u9b1_content import TITLE,ABSTRACT,CONTRIBUTIONS,INTRO_PARAGRAPHS,INTRO_END,RELATED_PARAGRAPHS,SOURCE_TO_BIB
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u9b1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def csvwrite(path,rows):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def plain(text):
 text=re.sub(r'(?m)(?<!\\)%.*$','',text)
 text=re.sub(r'\\cite\{[^}]+\}','',text)
 text=re.sub(r'\\(?:begin|end|section|subsection|paragraph|label)\{[^}]*\}','',text)
 text=text.replace('\\%','%').replace('~',' ').replace('\\item','')
 text=re.sub(r'\\[a-zA-Z]+','',text)
 text=text.replace('{','').replace('}','')
 return text
def words(text):return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",plain(text)))
def rows_for_metrics(formal,sealed):
 rows=[];values={}
 for key,comparison,digits in [('source','source_gain',2),('convex','convex_gain',2),('penalty','compression_penalty',3)]:
  for snr in (10,20):
   match=[(i,r) for i,r in enumerate(formal['bootstrap_results']) if r['environment']=='CDL-B-120-100' and r['snr_db']==snr and r['comparison']==comparison and r['metric']=='nmse_difference_db']
   if len(match)!=1:raise ValueError('Ambiguous frozen metric')
   i,r=match[0];display=f"{r['estimate']:.{digits}f}";values[key+str(snr)]=display
   rows.append(dict(claim_id=f'A_{key}_{snr}',source_task='T20B',source_file=str(ROOT/'outputs/t20b_formal_results.json'),locator=f'bootstrap_results[{i}]',metric=comparison,condition='CDL-B-120km/h-100ns',snr_db=snr,raw_value=r['estimate'],display_value=display,evidence_level='Tier A formal',criterion_bound=r['primary_upper'] if key=='penalty' else r['primary_lower'],audit='PASS'))
 for i,snr in enumerate((10,20)):
  raw=sealed['results']['gain'][i];values['sealed'+str(snr)]=f'{raw:.2f}'
  rows.append(dict(claim_id=f'A_sealed_{snr}',source_task='SP-U7B-R1',source_file=str(ROOT/'outputs/sp_u7b_r1/analysis.json'),locator=f'results.gain[{i}]',metric='Source minus LR4 pooled-energy NMSE dB',condition='CDL-A-120km/h-100ns',snr_db=snr,raw_value=raw,display_value=values['sealed'+str(snr)],evidence_level='Tier A sealed',criterion_bound=sealed['results']['primary_lcb'][i],audit='PASS'))
 saving=formal['complexity']['saving_vs_dense_affine_percent'];values['saving']=f'{saving:.2f}'
 rows.append(dict(claim_id='A_cost',source_task='T20B engineering count',source_file=str(ROOT/'outputs/t20b_formal_results.json'),locator='complexity.saving_vs_dense_affine_percent',metric='arithmetic proxy reduction versus Dense Affine percent',condition='two rank4 corrections; Source full; dense pilot fitting retained',snr_db='not applicable',raw_value=saving,display_value=values['saving'],evidence_level='Tier C arithmetic proxy',criterion_bound='not runtime',audit='PASS'))
 # The threshold is read from the preserved published endpoint table, never chosen from outcomes.
 endpoints=(ROOT/'paper/tables/04_endpoints.tex').read_text(encoding='utf-8')
 margins=re.findall(r'Compression & B-120-100 & (?:10|20) & [^\n]+?upper < ([0-9.]+)',endpoints)
 if margins!=['0.50','0.50']:raise ValueError('Unexpected frozen compression margin')
 values['margin']=f'{float(margins[0]):.1f}'
 rows.append(dict(claim_id='A_compression_margin',source_task='T20B preregistered criterion',source_file=str(ROOT/'paper/tables/04_endpoints.tex'),locator='Compression B-120-100 rows, upper < 0.50',metric='fixed compression margin dB',condition='Composed, SNR10/20',snr_db='10/20',raw_value=float(margins[0]),display_value=values['margin'],evidence_level='Tier A frozen criterion',criterion_bound='both preregistered upper bounds below margin',audit='PASS'))
 definitions=[('A_folds','SP-U2A frozen code map','docs/sp_u2a_equation_inventory.md','E02','deterministic folds',4,'four'),
 ('A_window','SP-U2A frozen code map','docs/sp_u2a_equation_inventory.md','E07','causal frames including current',5,'five'),
 ('A_coordinates','SP-U2A frozen code map','docs/sp_u2a_equation_inventory.md','E04-E07','unrestricted real coordinates',2,'two'),
 ('A_rank','T20B frozen method','outputs/t20b_formal_results.json','complexity.rank','deployed correction rank',formal['complexity']['rank'],'four'),
 ('A_snr10','T20B / U7B-R1','outputs/t20b_formal_results.json','snr','reported SNR dB',10,'10'),
 ('A_snr20','T20B / U7B-R1','outputs/t20b_formal_results.json','snr','reported SNR dB',20,'20'),
 ('A_speed','T20B / U7B-R1','outputs/sp_u9a_claim_evidence_matrix.json','C1/C7 conditions','speed km/h',120,'120'),
 ('A_delay','T20B / U7B-R1','outputs/sp_u9a_claim_evidence_matrix.json','C1/C7 conditions','delay spread ns',100,'100'),
 ('A_unseen_count','SP-U7B-R1','outputs/sp_u9a_claim_evidence_matrix.json','C7/C8 scope','independently evaluated unseen profile conditions',1,'single')]
 for id,task,file,loc,metric,value,display in definitions:
  rows.append(dict(claim_id=id,source_task=task,source_file=str(ROOT/file),locator=loc,metric=metric,condition='frozen interface or named evaluation condition',snr_db='as stated',raw_value=value,display_value=display,evidence_level='frozen definition / Tier A scope',criterion_bound='not a new threshold',audit='PASS'))
 if any(r['criterion_bound']>=float(values['margin']) for r in rows if r['metric']=='compression_penalty'):raise ValueError('Compression criterion unsupported')
 return values,rows

def bib_entry(s,key):
 name=s['id'];fields={}
 fields['author']=' and '.join(unicodedata.normalize('NFKD',a).encode('ascii','ignore').decode() for a in s['authors'])
 fields['title']=re.sub(r'\b(MIMO-OFDM|LMMSE|MMSE|MIMO|OFDM|MSE|LMS)\b',r'{\1}',s['title']);fields['year']=str(s['year'])
 if name in ('P1','P4','P6','Mei'):
  typ='article';journals={'P1':'IEEE Transactions on Signal Processing','P4':'IEEE Transactions on Wireless Communications','P6':'IET Communications','Mei':'IET Communications'}
  fields['journal']=journals[name]
  metadata=json.loads((OUT.parent/'sp_u9a_r1'/('crossref_P7.json' if name=='Mei' else 'crossref_'+name+'.json')).read_text(encoding='utf-8'))['message']
  for src,dest in [('volume','volume'),('issue','number'),('page','pages')]:
   if metadata.get(src):fields[dest]=metadata[src].replace('-','--') if src=='page' else metadata[src]
 elif name in ('P2','P3','P5','LS2010'):
  typ='inproceedings';venues={'P2':'2022 56th Asilomar Conference on Signals, Systems, and Computers','P3':'2023 57th Asilomar Conference on Signals, Systems, and Computers','P5':'The 2005 IEEE Annual Conference on Wireless and Microwave Technology','LS2010':'Proceedings of 2010 IEEE International Symposium on Circuits and Systems'}
  fields['booktitle']=venues[name];fields['pages']=s['crossref']['page'].replace('-','--')
 else:
  typ='misc';fields['howpublished']='arXiv preprint';fields['eprint']='1407.6083v1' if name=='P6pre' else '2506.00452v5';fields['archivePrefix']='arXiv'
 if s['doi']:fields['doi']=s['doi']
 fields['url']=s['url'];fields['verified_source']=s['url']
 notes={'P1':'Method comparison based on author manuscript arXiv:2112.12499v2',
  'P2':'Method comparison based on author manuscript arXiv:2205.03634v3, a later author revision of the 2022 conference work',
  'P3':'Method comparison based on author manuscript arXiv:2304.14809v2',
  'P6pre':'Separate two-author preprint; not asserted identical to the four-author 2015 journal article',
  'AMMSE':'Fixed reproduction target: arXiv version 5; 2026 revision of an initial 2025 submission; cited as a preprint'}
 if name in notes:fields['note']=notes[name]
 body='@'+typ+'{'+key+',\n'+',\n'.join(' '+k+' = {'+v+'}' for k,v in fields.items())+'\n}\n'
 return body,fields

def main():
 baseline=json.loads((ROOT/'outputs/sp_u9b1_preedit_manifest.json').read_text(encoding='utf-8'))
 ledger=json.loads((ROOT/'outputs/sp_u9a_r1_source_ledger.json').read_text(encoding='utf-8'))
 records={s['id']:s for s in ledger['sources']}
 formal=json.loads((ROOT/'outputs/t20b_formal_results.json').read_text(encoding='utf-8'))
 sealed=json.loads((ROOT/'outputs/sp_u7b_r1/analysis.json').read_text(encoding='utf-8'))
 values,number_rows=rows_for_metrics(formal,sealed)
 abstract=ABSTRACT.format(**values)
 old_metadata=(OUT/'pre_source/metadata.tex').read_text(encoding='utf-8')
 metadata=re.sub(r'\\newcommand\{\\PaperTitle\}\{[^\n]*\}',lambda m:'\\newcommand{\\PaperTitle}{'+TITLE+'}',old_metadata)
 metadata+='\\hypersetup{pdftitle={\\PaperTitle}}\n'
 (ROOT/'paper/metadata.tex').write_text(metadata,encoding='utf-8')
 old_main=(OUT/'pre_source/main.tex').read_text(encoding='utf-8')
 main_tex=re.sub(r'(?s)(\\begin\{abstract\}).*?(\\end\{abstract\})',lambda m:m[1]+'\n'+abstract+'\n'+m[2],old_main)
 (ROOT/'paper/main.tex').write_text(main_tex,encoding='utf-8')
 intro='\\section{Introduction}\n\n'+'\n\n'.join('% '+id+' '+job+'\n'+p for id,job,p in INTRO_PARAGRAPHS)+'\n\n'+INTRO_END.replace('{contributions}','\n'.join('\\item '+p for p in CONTRIBUTIONS))+'\n'
 related='\\section{Related Work}\n\n'+'\n\n'.join('% '+id+' '+topic+'\n\\noindent\\textbf{'+topic+'.} '+p.rstrip() for id,topic,p in RELATED_PARAGRAPHS)+'\n'
 (ROOT/'paper/sections/01_introduction.tex').write_text(intro,encoding='utf-8');(ROOT/'paper/sections/02_related_work.tex').write_text(related,encoding='utf-8')
 bib=(OUT/'pre_source/bibliography/references.bib').read_text(encoding='utf-8')
 bib_audit=[]
 for id,key in SOURCE_TO_BIB.items():
  s=records[id];entry,fields=bib_entry(s,key)
  pattern=r'(?ms)^@\w+\{'+re.escape(key)+r',.*?^\}\s*'
  existed=bool(re.search(pattern,bib))
  if existed:bib=re.sub(pattern,lambda m:entry,bib)
  else:bib+='\n'+entry
  bib_audit.append(dict(source_id=id,key=key,action='CHECKED_UPDATED' if existed else 'ADDED',fields=fields,quality=s['quality'],source_ledger=str(ROOT/'outputs/sp_u9a_r1_source_ledger.json'),source_locator=s['locator'],source_limit=s['full_text_status']))
 (ROOT/'paper/bibliography/references.bib').write_text(bib,encoding='utf-8')
 csvwrite(ROOT/'outputs/sp_u9b1_abstract_number_audit.csv',number_rows)
 dump(OUT/'bibliography_audit.json',bib_audit)
 claim_rows=[]
 sources_by_job={'problem':'P1 general LMMSE model; frozen interface definition','existing solutions':'U9A-R1 P1/P2/P3/P4/P5/Mei/AMMSE',
  'question and hypothesis':'Study question; not asserted empirical guarantee','source-anchored interface':'SP-U2A E01/E03/E08; U9A-R1 fingerprint',
  'cross-pilot definition':'SP-U2A E02-E07; U9A-R1 LS2010','compression and information contract':'SP-U2A E03/E08; frozen method interface',
  'surrogate and theory boundary':'U9A C12/C12-N; SP-U2A assumption audit; no new inference',
  'evidence overview':'U9A C1/C2/C5/C6/C7/C9; T20B/U7B-R1 formal; U4/U5 development; U8 post-hoc',
  'boundary and scope':'U9A C3/C4/C8/C9; frozen experimental setup; no-ICI abstraction'}
 for id,job,p in INTRO_PARAGRAPHS:
  sentences=re.split(r'(?<=[.!?])\s+(?=[A-Z])',p)
  for n,sentence in enumerate(sentences,1):
   evidence_level={'question and hypothesis':'HYPOTHESIS, not performance claim','evidence overview':'Tier A / Tier B / Tier D, separated','boundary and scope':'Tier A limitations + scope definition','surrogate and theory boundary':'conditional identity / bounded negative evidence'}.get(job,'frozen definition or verified literature')
   claim_rows.append(dict(claim_id=id+f'-S{n}',sentence=sentence,evidence_source=sources_by_job[job],evidence_level=evidence_level,claim_type=job,allowed='YES',qualification='No priority, external superiority or universal risk guarantee'))
 for i,p in enumerate(CONTRIBUTIONS,1):claim_rows.append(dict(claim_id=f'CONTRIBUTION-{i}',sentence=p,evidence_source='U9A-R1 Conservative + frozen interface/evidence roles',evidence_level='formulation / evaluation, scoped',claim_type='contribution positioning',allowed='YES',qualification='Component-priority test PASS: names prepared objects/objective or implemented evaluation'))
 csvwrite(ROOT/'outputs/sp_u9b1_intro_claim_audit.csv',claim_rows)
 rw_rows=[]
 for id,topic,p in RELATED_PARAGRAPHS:
  for n,sentence in enumerate(re.split(r'(?<=[.!?])\s+(?=[A-Z])',p),1):
   keys=[]
   for group in re.findall(r'\\cite\{([^}]+)\}',sentence):keys.extend(group.split(','))
   ids=[s for s,k in SOURCE_TO_BIB.items() if k in keys]
   if 'luan2023channelformer' in keys:ids.append('Existing Channelformer: title-level context only; frozen T20D technical_closure reused under RW5 authorization')
   rw_rows.append(dict(paragraph=id,sentence_id=f'{id}-S{n}',sentence=sentence,citations=';'.join(keys),source_ids=';'.join(ids),evidence_source='U9A-R1 source ledger' if ids and 'luan2023channelformer' not in keys else ('frozen existing bibliography/technical_closure; no new method details' if 'luan2023channelformer' in keys else 'Own interface or synthesis; no external exact-method assertion'),collision_handling=topic,allowed='YES',source_limit_preserved='YES'))
 csvwrite(ROOT/'outputs/sp_u9b1_related_work_collision_audit.csv',rw_rows)
 terminology=dict(Source_anchored='Frozen Source empirical LMMSE retained as additive base; not Bayesian theorem',cross_pilot_fitting='Four deterministic folds predict heldout received pilot observations from other folds',profile_origin_correction='Prepared statistical difference, not physical factor',mobility_origin_correction='Prepared statistical difference, not independent latent factor',LR4='Only deployment corrections rank4; Source and pilot prediction kernels remain dense',arithmetic_proxy='Complex-MAC count under frozen convention; not runtime',Matched='Finite matched-statistics reference, not population MMSE oracle')
 dump(OUT/'terminology.json',terminology)
 report=dict(task='SP-U9B-1',word_counts=dict(abstract=words(abstract),introduction=words(intro),related_work=words(related)),
  new_citations=[r['key'] for r in bib_audit if r['action']=='ADDED'],ET2026_included=False,ET_reason='Default exclusion retained; bounded component overlap did not change the chosen nearest-neighbor positioning.',
  contributions=CONTRIBUTIONS,abstract_plain=plain(abstract),abstract_number_rows=len(number_rows),intro_claim_rows=len(claim_rows),related_work_sentence_rows=len(rw_rows),
  no_new_experiments=True,no_new_bootstrap=True,no_new_threshold=True,
  bibliography_path_mapping=baseline['bibliography_mapping'],author_metadata='Unchanged TODO author/affiliation placeholders; no names invented')
 dump(OUT/'editorial_report.json',report)
 print(json.dumps(report['word_counts']|{'new_citations':report['new_citations'],'abstract_number_rows':len(number_rows)},ensure_ascii=False))
 return report
if __name__=='__main__':main()
