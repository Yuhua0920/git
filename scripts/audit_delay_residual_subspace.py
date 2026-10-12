"""T19D frozen artifacts, exact reconstruction and final historical regression."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src import delay_residual_subspace as p
from src import frozen_affine_validation as old
from src.cdl_benchmark import array_hash
from scripts.run_delay_residual_subspace import OUT,RESULT,SOURCES,TABLES,guard,accepted,inputs,regression,compute,verify_protected
from scripts.run_cdl_mismatch import load,save,sha,environments
from scripts.run_pilot_cv import readcsv


def verify(r):
    verify_protected(r)
    for group in ('numeric_artifact_sha256','claim_document_sha256','figure_artifact_sha256'):
        for path,h in r.get(group,{}).items():
            if sha(ROOT/path)!=h:raise ValueError('artifact byte drift:'+path)
    if any(r[k] for k in ('new_channel_generated','new_module_added','new_model_trained','rank_compression_performed','gamma_pilot_fitting','deployed_fitting','validation_NMSE','BER_evaluations','formal_test_used','target_weighting_used_for_basis')):
        raise ValueError('scientific scope violation')
    if r['generator_calls']!=0 or r['statistics_refits']!=0 or not all(r[k] for k in ('method_frozen','formal_test_sealed','source_common_weighting','development_heldout_separation','t19a_rank4_out_of_factor_warning')):
        raise ValueError('frozen/no-generation/formal gate failed')
    if r['development_conditions']!=['C30/50','C30/200'] or r['heldout_conditions']!=['C90/50','C90/200','B30/50','B30/200','B90/50','B90/200'] or r['module_space_dimensions']!=[1,2]:
        raise ValueError('development/heldout/dimension design changed')
    if (r['c6_claim_status'],r['c7_claim_status'],r['c8_claim_status'],r['c9_claim_status'],r['c10_claim_status'],r['c11_claim_status'],r['t18a_materiality_status'])!=(old.C6,old.C7,'SUPPORTED_IN_FIXED_ENVIRONMENTS','INCONCLUSIVE','INCONCLUSIVE','SUPPORTED_IN_C30_200_10_STATISTICS_REPLICATION','PARTIAL'):
        raise ValueError('historical claim overwritten')
    mapping={'residual_pca':'pca_explained_energy','loro_stability':'loro_stability','heldout_transfer':'heldout_transfer',
             'direction_similarity':'direction_similarity','direction_comparison':'direction_comparison','development_reconstruction':'development_reconstruction'}
    for name,key in mapping.items():
        rows=readcsv(OUT/f't19d_{name}.csv')
        if len(rows)!=len(r[key]):raise ValueError('CSV/JSON row count mismatch')
        for csvrow,value in zip(rows,r[key]):
            if any(csvrow[k]!=str(v) for k,v in value.items()):raise ValueError('CSV/JSON field mismatch:'+name)
    with guard():
        accepted();arrays,source,dev,held,proof=inputs()
        if proof!=r['method_provenance'] or regression(arrays,source,dev,held)!=r['historical_regression']:
            raise ValueError('frozen method/historical operator replay drift')
        tables,residuals,basis,metadata,similarity=compute(arrays,source,dev,held)
    for name,key in mapping.items():
        if tables[name]!=r[key]:raise ValueError('identical computation numeric replay not exact:'+name)
    if metadata!=r['basis_metadata'] or similarity!=r['cross_delay_cosine_summary']:
        raise ValueError('basis sign/ordering/similarity metadata drift')
    for name,computed in (('pure_delay_residuals',residuals),('delay_basis',basis)):
        with np.load(OUT/f't19d_{name}.npz',allow_pickle=False) as f:
            if set(f.files)!=set(computed) or set(f.files)!=set(r['array_metadata'][name]):raise ValueError('persisted arrays incomplete')
            for key,value in computed.items():
                info=r['array_metadata'][name][key];stored=f[key]
                if str(stored.dtype)!=info['dtype'] or list(stored.shape)!=info['shape'] or array_hash(stored)!=info['sha256']:
                    raise ValueError('persisted shape/dtype/array hash mismatch:'+key)
                if not np.array_equal(stored,value):raise ValueError('same-environment residual/basis array reconstruction not exact:'+key)
    if len(r['loro_stability'])!=40 or len(r['direction_similarity'])!=90 or len(r['development_reconstruction'])!=20:
        raise ValueError('stability/similarity/development population incomplete')
    for row in r['loro_stability']:
        k=row['heldout_pair'];expected=[[n,i] for n in p.DEVELOPMENT for i in p.REPLICATES if i!=k]
        if row['construction_keys']!=expected or row['replicate']!=k or row['used_for_transfer_gate']:
            raise ValueError('LORO exclusion leak')
    for s in p.SNRS:
        md=r['basis_metadata'][str(s)]
        if [list(v) for v in md['construction_keys']]!=[[n,i] for n in p.DEVELOPMENT for i in p.REPLICATES] or md['centered']:
            raise ValueError('final normalized uncentered construction leak')
    for key in ('capture_q0','capture_q1','capture_q2','increment_q1','increment_q2','marginal_q2'):
        if r[key]!=[v[key] for v in tables['heldout_transfer']]:raise ValueError('endpoint field drift')
    if any(r[k]!=v for k,v in p.transfer_gate(r['heldout_transfer']).items()):raise ValueError('pre-registered scientific gate drift')
    return True


def claim():
    r=load(RESULT);verify(r)
    if r.get('claim_document_sha256'):raise ValueError('existing claim documents protected')
    text='# T19D Claim–Evidence Update\n\nResearch Question: Pure-delay residual operator-module-space dimension and held-out factor transfer.\n\n'
    text+='C12: '+p.C12+'\n\nStatus: '+r['c12_claim_status']+'.\n\n'
    text+='Historical C6='+old.C6+', C7='+old.C7+', C8 SUPPORTED_IN_FIXED_ENVIRONMENTS, C9/C10 INCONCLUSIVE, C11 SUPPORTED_IN_C30_200_10_STATISTICS_REPLICATION; T18A PARTIAL and rank4 warning unchanged.\n\n'
    text+='| Held-out context | SNR | q0 | q1 | q2 | Increment1 | Increment2 | Marginal2 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for v in r['heldout_transfer']:
        text+='| '+v['environment']+' | '+str(v['snr_db'])+' | '+' | '.join(str(v[k]) for k in ('capture_q0','capture_q1','capture_q2','increment_q1','increment_q2','marginal_q2'))+' |\n'
    text+='\n'+''.join(f'- {k}: {r[k]}\n' for k in ('q1_promising_count','q2_promising_count','q2_marginal_at_least_005_count','q2_increment_below_005_count',
        'single_delay_direction_transfer_supported','two_direction_delay_subspace_transfer_supported','delay_residual_subspace_transfer_weak','delay_residual_subspace_transfer_partial'))
    text+='\nDecision dependency: '+r['next_step_dependency']+' No next-stage work implemented.\n\n'
    for s in p.SNRS:
        spectrum=[v for v in r['pca_explained_energy'] if v['snr_db']==s]
        text+=f"{s}dB normalized development energy: q1={spectrum[0]['cumulative_energy']}, q2={spectrum[1]['cumulative_energy']}. Development/LORO diagnostics do not determine C12.\n\n"
    text+='Five C30/200/10 target-statistics replicates support a condition-specific deficiency, but do not serve as held-out transfer evidence. Dimensions q1/q2 count operator matrices, not matrix rank. All extensions here are joint real oracle LS only.\n'
    review='# T19D Reviewer Red Team\n\n'
    review+='- Development uses exactly ten pure-delay C30 statistics per SNR. Neither held-out C90/B30/B90, original T19A pure-delay statistics nor pooled2500 constructs the basis. Five LORO folds exclude both delay samples of the same replicate pair.\n'
    review+='- All residuals are projected using one source Cz per SNR, equal-direction normalization and uncentered real-augmented SVD. Complex phases cannot be absorbed into a real scalar. SNR-specific bases do not establish an SNR-invariant direction.\n'
    review+='- q1/q2 mean additional operator-module-space dimension, not matrix rank. Source-weighted basis recovery uses a transpose right triangular solve; sign ambiguity is only plus/minus with a frozen pivot rule.\n'
    review+='- High development PCA energy or LORO reconstruction does not demonstrate factor transfer. C12 uses all twelve completely held-out contexts under frozen pre-registered capture/increment counts.\n'
    review+='- A high q0 limits attainable incremental capture: increment cannot exceed1-q0 in exact oracle arithmetic. Keep the fixed0.10/0.05 gates and report ceiling effects without redefining thresholds.\n'
    review+='- The twelve context/SNR values share frozen source components and are not twelve independent statistical replicates. No bootstrap, p-value, universal population claim or deployed-NMSE claim is made.\n'
    review+='- Dense q1/q2 fitting is joint, never sequential. Target-weighted oracle coordinates and source-weighted basis construction are distinct operations. No deployed fitting or third module is introduced.\n'
    review+='- T19B D200 is replayed exactly over the same twelve contexts and compared descriptively. PCA uses a broader independent-statistics development set; the comparison is not an independent deployment replication.\n'
    review+='- Cross-delay negative cosines may describe a signed axis but cannot establish a physical linear delay law. Compression/rank4,Pilot-CV gamma,channel realization and BER are outside this study.\n\n'
    review+='Actual C12: '+r['c12_claim_status']+'; '+r['next_step_dependency']+'\n'
    hashes={}
    for name,value in (('t19d_claim_evidence_update.md',text),('t19d_reviewer_update.md',review)):
        path=ROOT/'docs'/name
        if path.exists():raise ValueError('existing document protected')
        path.write_text(value,encoding='utf8');hashes['docs/'+name]=sha(path)
    r['claim_document_sha256']=hashes;save(RESULT,r);print('T19D bounded Claim–Evidence and Reviewer update generated',flush=True)


def visual_review():
    r=load(OUT/'t19d_figure_audit.json')
    if r['source_returncode'] or not all(r['CSV_only_rebuild'].values()) or any(v['text_returncode'] or v['collision_returncode'] for v in r['rendered_checks'].values()):
        raise ValueError('automatic figure QA failed')
    r.update(visual_review='Actual three PNGs inspected: two ten-component normalized energy curves; all12 held-out paired q0/q1/q2 and all36 incremental/marginal marks. Labels/legends distinguish development and transfer, q is module-space dimension, no clipping/collision/CI.',all_checks_passed=True)
    save(OUT/'t19d_figure_audit.json',r)


def finalize():
    tick=time.perf_counter();r=load(RESULT);verify(r)
    if not r.get('claim_document_sha256') or not load(OUT/'t19d_figure_audit.json')['all_checks_passed']:
        raise ValueError('claims/actual figure review incomplete')
    env=dict(os.environ);env['TEMP']=env['TMP']=str(ROOT/'.pytest_cache/pytest_workspace_temp');tests={}
    groups=[('related',['tests/test_delay_residual_subspace.py']),('provenance',['tests/test_target_statistics_replication.py','tests/test_operator_space_audit.py','tests/test_historical_provenance.py']),('full',[])]
    for label,args in groups:
        path=OUT/f't19d_pytest_{label}.txt'
        if path.exists():raise ValueError('existing regression log protected')
        start=time.perf_counter();print('Actual '+label+' regression',flush=True)
        with path.open('w',encoding='utf8') as f:
            proc=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-m','pytest',*args,'-q'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        summary=path.read_text(encoding='utf8').strip().splitlines()[-1];print(summary,flush=True)
        tests[label]=dict(command=[str(ROOT/'.venv/Scripts/python.exe'),'-m','pytest',*args,'-q'],summary=summary,exit_code=proc.returncode,seconds=time.perf_counter()-start,sha256=sha(path))
        if proc.returncode or 'skipped' in summary:raise ValueError('real regression failure/skip:STOP')
    verify(r);after=environments()
    if after!=r['environment_before']:raise ValueError('frozen environment payload changed')
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode:raise ValueError('git diff --check failed')
    equivalent={}
    for path in (*SOURCES,*r['claim_document_sha256']):
        lines=(ROOT/path).read_text(encoding='utf8').splitlines();bad=[i for i,l in enumerate(lines,1) if l.rstrip()!=l]
        if bad:raise ValueError('new source trailing whitespace:'+path)
        equivalent[path]=dict(lines=len(lines),sha256=sha(ROOT/path),trailing_whitespace_lines=bad)
    from scripts.plot_delay_residual_subspace import STEMS
    r.update(pytest=tests,environment_after=after,git_diff_check=dict(exit_code=diff.returncode,output=diff.stdout+diff.stderr),
        final_hash_audit=dict(passed=True,historical_files=len(r['frozen_history_sha256']),unexpected_content_mismatch=0),
        new_file_diff_equivalent=equivalent,figure_audit_sha256=sha(OUT/'t19d_figure_audit.json'),
        figure_artifact_sha256={f'outputs/{stem}.{ext}':sha(OUT/f'{stem}.{ext}') for stem in STEMS for ext in ('png','pdf','svg','alignment.json','collision-audit.json')},
        all_checks_passed=True,completion_state='completed')
    r['runtime']['final_audit_and_regression_seconds']=time.perf_counter()-tick
    save(RESULT,r);print('T19D engineering PASS; C12='+r['c12_claim_status']+'; no deployed module; formal sealed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['claim','visual-review','finalize']);args=parser.parse_args()
    {'claim':claim,'visual-review':visual_review,'finalize':finalize}[args.mode]()
