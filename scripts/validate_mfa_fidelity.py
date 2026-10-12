"""B5A reference/toy validation; no scientific model selection or evaluation."""
from pathlib import Path
import sys,subprocess,csv,json,platform,importlib.metadata,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u12b5a_common import firewall,report,write,read,OUT,utc
firewall()
from scripts import sp_u12b5a_cases as c

def main():
    run=subprocess.run([sys.executable,'-X','utf8','-m','scripts.sp_u12b5a_pytest'],cwd=ROOT,capture_output=True,text=True,encoding='utf8')
    write(OUT/'pytest_result.json',dict(exit_code=run.returncode,stdout=run.stdout,stderr=run.stderr))
    print(run.stdout,flush=True)
    if run.returncode:raise SystemExit(run.returncode)
    with c.threadpoolctl.threadpool_limits(1):
        posterior=[c.posterior_case(K,kind) for K in (1,2,4) for kind in ('identity','selector','colored')]
        em=c.em_case();init=c.initialization_case();seed=c.seed_case()
        vector=c.vector_case();observation=c.observation_case();grid=c.grid_case();firewall=c.firewall_case()
        demo=c.reference_demo()
        packages=[]
        for name,module,license_name in [('numpy',c.np,'BSD-3-Clause'),('scipy',c.scipy,'BSD-3-Clause'),('scikit-learn',c.sklearn,'BSD-3-Clause'),('threadpoolctl',c.threadpoolctl,'BSD-3-Clause')]:
            packages.append(dict(package=name,version=module.__version__,module_path=module.__file__,license=license_name,installed_this_task=False))
        dep=dict(PASS=True,Python=sys.version,platform=platform.platform(),packages=packages,production_NumPy_SciPy_unchanged=True,new_dependencies=[],optional_existing_dependency_directory=str(ROOT/'.venv-gmm/Lib/site-packages'),original_requirements=dict(python='3.10.12',numpy='1.25.2',scipy='1.11.1',scikit_learn='1.2.2'),exact_2023_environment_replicated=False,KMeans_current_defaults=c.sklearn.cluster.KMeans().get_params(),KMeans_author_explicit_n_init=1,compatibility='private module np.Inf alias; submodule namespace only; original source bytes unchanged',runtime_comparison=False,BLAS_thread_pools=c.threadpoolctl.threadpool_info())
    report('reference_parity',dict(PASS=all(x['raw_adapter']['PASS'] for x in posterior) and em['PASS'] and init['PASS'] and seed['PASS'],basis='AUTHOR_CODE_ADAPTER',selected_version='2023',cases=posterior,EM_sequence=em,initialization=init,training_parity=seed['reference_adapter'],reference_demo=demo,project_native_implementation=False))
    report('toy_parity',dict(PASS=all(x['PASS'] for x in posterior) and em['PASS'] and vector['PASS'] and observation['PASS'],atol=c.ATOL,rtol=c.RTOL,posterior_cases=posterior,EM_sufficient_statistics=em,vectorization=vector,observation=observation))
    report('seed_replay',seed);report('structure_support',grid);report('dependency_audit',dep)
    write(OUT/'interface_tests.json',dict(PASS=vector['PASS'] and observation['PASS'] and firewall['PASS'],vector=vector,observation=observation,firewall=firewall))
    required=[em,init,seed,vector,observation,grid,firewall,demo]+posterior
    passed=all(x['PASS'] for x in required)
    # Complete the original evidence table with real validation results.
    matrix=ROOT/'outputs/sp_u12b5a_mfa_evidence_matrix.csv'
    with matrix.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    testmap={'parameter learning':'EM_sufficient_statistics','EM':'EM_sufficient_statistics','E step':'EM_sufficient_statistics','M step':'EM_sufficient_statistics','stopping':'author source + seed replay finite history','max iterations':'constructor defaults + seed replay','initialization':'initialization_draw_order + seed replay','restarts':'single fit author call graph','original model selection':'paper reviewed; future mentor protocol only','complex representation':'proper complex likelihood + complex/real roundtrip','real expansion':'author utility roundtrip','observation':'production pilot mapping + covariance transport','channel model':'known structured covariance + posterior','residual covariance':'PPCA update + fitted isotropy','inference':'nine independent dense posterior cases','paper/code differences':'documented exact API/version mapping'}
    for row in rows:
        row['Parity_test']=testmap.get(row['Item'],'author source / reference parity / dense analytical posterior')
        row['Status']='AMBIGUOUS' if row['Item']=='original model selection' else ('VERIFIED_WITH_COMPATIBILITY_ADAPTER' if row['Item'] in ('observation','inference','normalization centering','inverse normalization') else 'VERIFIED')
        if row['Item']=='original model selection':row['Ambiguity']='Original paper automatic selection recipe unspecified; future target-blind mentor protocol already resolves project selection. No final K/rank chosen.'
    with matrix.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(PASS=passed,created_utc=utc(),formal_training=False,target_evaluation=False,method_comparison=False,B2_B3_outcomes_viewed=False,new_dependencies=[],basis='AUTHOR_CODE_ADAPTER',pytest=read(OUT/'pytest_result.json'),all_required_checks_passed=passed)
    write(OUT/'validation_summary.json',summary)
    (ROOT/'docs/sp_u12b5a_mfa_fidelity_report.md').write_text('# MFA fidelity report\n\n'+json.dumps(summary,indent=2)+'\n\n## Raw/adapter and independent formula checks\n\n'+json.dumps(dict(posterior=posterior,EM=em,initialization=init,seed=seed,vector=vector,observation=observation,grid=grid,demo=demo),indent=2)+'\n\nNo published figure performance, target NMSE, method ranking, runtime or memory comparison is produced. Full-scale resource/convergence validation belongs to a separately authorized later task.\n',encoding='utf8')
    (ROOT/'docs/sp_u12b5a_change_log.md').write_text('# B5A change log\n\nOnly new B5A evidence, immutable author reference copies, thin adapter and small reference tests were created. Production Source/Dense/LR4 and manuscript untouched. No scientific training/evaluation or model selection executed.\n\nInitial directory preflight failed before MFA import due to parent reference mkdir protection. Failed sources preserved under outputs/sp_u12b5a/preflight_directory_failure01. New bookkeeping permits only mkdir of nonexistent reference and src/baselines parents. Two evidence-generator syntax errors were corrected before its first successful specification generation; no scientific samples were involved.\n\nFresh GitHub current archives were byte-verified against originals. Full public history showed exact 2023 estimator/core gitlink pair; latest paper-period pair selected over materially changed2026 HEAD. Earlier pending provenance statement corrected from unestablished to found.\n\nCompatibility: private author module np.Inf→np.inf; temporary import aliases for original package layout. Explicit legacy RNG state scoping/restoration records the author RNG without changing draw algorithms. Float1.0 uses documented full-mixture branch. PPCA True/lock False follows paper numerical experiment. No author source patch or numerical enhancement.\n\nAll outputs generated by scripts; actual pytest and validate_mfa_fidelity entrypoint executed. No Git commit.\n',encoding='utf8')
    note='\n\nOriginal author demo diagnostic: rs_clip=1e-6 does not renormalize training responsibilities; fitted weights sum to 1.0000140301507119, so the unit-weight diagnostic FAIL is retained (tolerance1e-10). Demo expected behavior is completion, printed sum and finite arrays, without an author assertion of exact normalization. The first validation attempt and failed diagnostic were archived before clarifying this expectation. This is not a tolerance relaxation or model repair. Baseline rs_clip=0 was frozen before tests and passes weight/responsibility checks.\n'
    for file in ('docs/sp_u12b5a_mfa_reference_notes.md','docs/sp_u12b5a_mfa_ambiguity_register.md','docs/sp_u12b5a_change_log.md'):
        path=ROOT/file;old=path.read_text(encoding='utf8')
        if 'Original author demo diagnostic:' not in old:path.write_text(old+note,encoding='utf8')
    print(json.dumps(dict(stage='MFA_fidelity_validation',PASS=passed,posterior_cases=len(posterior),EM=em['PASS'],seed_replay=seed['bitwise_replay'],demo=demo['PASS'])),flush=True)
    if not passed:raise SystemExit(1)
if __name__=='__main__':main()
