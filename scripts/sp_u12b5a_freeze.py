"""Close B5A provenance, future RNG registration and fidelity manifest only."""
import hashlib,json,csv,subprocess
from scripts.sp_u12b5a_common import ROOT,OUT,REF,read,write,sha,utc,firewall

def digest_tree(files):
    return hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main():
    firewall()
    validation=read(OUT/'validation_summary.json')
    preservation=read(ROOT/'outputs/sp_u12b5a_mfa_preservation.json')
    seals=read(ROOT/'outputs/sp_u12b5a_mfa_b2_b3_seal_audit.json')
    integrity=read(OUT/'reference_integrity.json')
    assert validation['PASS'] and preservation['PASS'] and seals['PASS'] and integrity['PASS']
    metadata=read(ROOT/'outputs/sp_u12b5a_mfa_reference_metadata.json')
    historical=read(OUT/'historical_reference.json')
    selected={}
    for repo,h in historical.items():
        commits=read(OUT/f'{repo}_verified_commits.json')
        entry=next(c for c in commits if c['sha']==h['commit'])
        selected[repo]=dict(h,repository_url=f'https://github.com/benediktfesl/{repo}',owner='benediktfesl',branch='main history; immutable 2023 commit snapshot',commit_UTC=entry['commit']['committer']['date'],accessed_UTC=utc(),README_version='2023 README; no semantic version/tag',README_sha256=h['files']['README.md'],license='BSD-3-Clause; core additionally retains original MIT notice' if repo=='cplx-mfa' else 'BSD-3-Clause',requirements=dict(Python='3.10.12',NumPy='1.25.2',SciPy='1.11.1',scikit_learn='1.2.2'))
    source_hashes={f'{repo}/{p}':v for repo,h in historical.items() for p,v in h['files'].items()}
    metadata['selected_reference']['repositories']=selected
    metadata['selected_reference']['combined_tree_sha256']=digest_tree(source_hashes)
    metadata['selected_reference']['combined_tree_hash_definition']='SHA256 UTF8 canonical JSON of sorted relative-file-path:SHA256 mapping across the two complete 2023 source trees'
    metadata['reference_immutable_copy']='REFERENCE_MFA_IMMUTABLE_COPY'
    write(ROOT/'outputs/sp_u12b5a_mfa_reference_metadata.json',metadata)

    # The seed rule is fixed without any preparation/validation performance.
    namespace='SP_U12B5A_MFA_AUTHOR2023_V1'
    def seed(role,K,L):
        return int.from_bytes(hashlib.sha256(f'{namespace}|{role}|K={K}|L={L}|fit=0'.encode()).digest()[:4],'big')
    registrations=[dict(K=K,loading_rank=L,candidate_training_initialization_seed=seed('candidate',K,L),full_retraining_initialization_seed=seed('full_retraining',K,L)) for K in (1,2,4) for L in (4,8,16)]
    allocated=[r[key] for r in registrations for key in ('candidate_training_initialization_seed','full_retraining_initialization_seed')]
    assert len(set(allocated))==18 and not ({4190123500,4190123501,0}&set(allocated))
    initpath=ROOT/'outputs/sp_u12b5a_mfa_initialization_audit.json'
    policy=read(initpath)
    policy.update(future_formal_seeds=dict(namespace=namespace,derivation='uint32 big endian first four bytes of SHA256(namespace|role|K=K|L=L|fit=0)',roles=['candidate','full_retraining'],registered=registrations,generated_without_training=True),training_seed='Per-role registered initialization seed; no independent stochastic EM draws',restart_seed='NOT_APPLICABLE: one fit, zero additional restarts',future_structure_selected=False)
    write(initpath,policy)
    specpath=ROOT/'outputs/sp_u12b5a_mfa_method_spec.json';spec=read(specpath);spec['policy']=policy;spec['future_rng_registered_at_freeze']=True;write(specpath,spec)
    specmd=ROOT/'docs/sp_u12b5a_mfa_method_specification.md'
    with specmd.open('a',encoding='utf8') as f:
        f.write('\n## Future RNG registration (design only)\n\n'+json.dumps(policy['future_formal_seeds'],indent=2)+'\n\nThese seed registrations do not select K or loading rank and do not execute training. Full retraining uses the selected configuration\'s preassigned independent role seed.\n')

    toy=read(ROOT/'outputs/sp_u12b5a_mfa_toy_parity.json');ref=read(ROOT/'outputs/sp_u12b5a_mfa_reference_parity.json');replay=read(ROOT/'outputs/sp_u12b5a_mfa_seed_replay.json')
    stats=dict(posterior_max_abs=max(x['final']['max_absolute_difference'] for x in toy['posterior_cases']),posterior_max_relative=max(x['final']['max_relative_difference'] for x in toy['posterior_cases']),responsibility_max_abs=max(x['responsibilities']['max_absolute_difference'] for x in toy['posterior_cases']),component_mean_max_abs=max(x['component_means']['max_absolute_difference'] for x in toy['posterior_cases']),EM_max_abs=max(x['max_absolute_difference'] for x in toy['EM_sufficient_statistics']['checks'].values()),author_adapter_max_abs=max(x['raw_adapter']['max_absolute_difference'] for x in ref['cases']),author_adapter_max_relative=max(x['raw_adapter']['max_relative_difference'] for x in ref['cases']),training_adapter_max_abs=max(x['max_absolute_difference'] for x in ref['training_parity'].values()),bitwise_seed_replay=replay['bitwise_replay'])
    reportpath=ROOT/'docs/sp_u12b5a_mfa_fidelity_report.md'
    with reportpath.open('a',encoding='utf8') as f:
        f.write('\n## Closure\n\nMFA_FIDELITY_READY: all core requirements pass for the pinned 2023 default-zero-clipping model. The original off-preset clipped-demo weight diagnostic FAIL remains visible; demo completion is a separate source-behavior PASS. No algorithm enhancement or relaxed numerical tolerance.\n\n'+json.dumps(dict(stats=stats,preservation=preservation,seals=seals),indent=2)+'\n\nFull-scale cost/convergence and target-blind model selection have not been executed. Stop for mentor review.\n')
    with (ROOT/'docs/sp_u12b5a_change_log.md').open('a',encoding='utf8') as f:
        f.write('\nReusable audit lessons: inspect full author commit history and exact submodule gitlinks before treating current HEAD as the only reference; distinguish documented demo behavior from mathematical properties that an optional clipping preset may violate. Do not repair author numerics to satisfy a mistaken expectation. Task scope limits lesson recording to B5A project reports.\n')
    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert git_head==preservation['Git_HEAD']
    status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True,encoding='utf8')
    (OUT/'git_status_final.txt').write_text(status,encoding='utf8')
    gitcheck=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True,encoding='utf8')
    write(OUT/'git_diff_check_final.json',dict(exit_code=gitcheck.returncode,stdout=gitcheck.stdout,stderr=gitcheck.stderr,HEAD=git_head))
    assert gitcheck.returncode==0

    required=['reference_metadata.json','evidence_matrix.csv','method_spec.json','symbol_mapping.csv','interface_contract.json','initialization_audit.json','dependency_audit.json','reference_parity.json','toy_parity.json','seed_replay.json','structure_support.json','b2_b3_seal_audit.json','preservation.json']
    required=[ROOT/f'outputs/sp_u12b5a_mfa_{n}' for n in required]
    required+=[ROOT/f'docs/sp_u12b5a_{n}.md' for n in ['mfa_reference_notes','mfa_method_specification','mfa_fidelity_report','mfa_ambiguity_register','change_log']]
    assert all(p.exists() for p in required)
    assert validation['pytest']['exit_code']==0 and '33 passed' in validation['pytest']['stdout']
    manifestpath=ROOT/'outputs/sp_u12b5a_mfa_fidelity_manifest.json'
    sidecar=manifestpath.with_suffix('.sha256')
    files=list(OUT.rglob('*'))+list(REF.rglob('*'))
    for d in ('docs','outputs','scripts'):
        files.extend((ROOT/d).glob('sp_u12b5a*'))
    files.extend([ROOT/'src/baselines/mfa_reference_adapter.py',ROOT/'tests/test_mfa_reference.py',ROOT/'scripts/validate_mfa_fidelity.py'])
    files={p for p in files if p.is_file() and p not in (manifestpath,sidecar) and '__pycache__' not in p.parts}
    artifacts={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    manifest=dict(task='SP-U12B5A',verdict='MFA_FIDELITY_READY',frozen_UTC=utc(),implementation_basis='AUTHOR_CODE_ADAPTER',reference=metadata['selected_reference'],frozen_policy=policy,parity_summary=stats,pytest=dict(passed=33,failed=0,skipped=0,entrypoint='scripts/validate_mfa_fidelity.py -> scripts.sp_u12b5a_pytest'),compatibility_patches=[dict(original='np.Inf in original MFA_cplx.py clipping paths',adapter='np.Inf attribute aliases np.inf only in private module globals',reason='NumPy2 removed alias',semantic_effect='NONE'),dict(original='from MFA_cplx import MFA_cplx / import utils',adapter='temporary import namespaces reconstruct original git-submodule module layout',reason='reference tree relocated under immutable task directory',semantic_effect='NONE')],author_source_bytes_modified=0,numerical_enhancements=[],algorithmic_ambiguities_unresolved=[],known_reference_deviations=['2023 rs_clip>0 demo does not renormalize responsibility/mixture weights; frozen default rs_clip=0 passes normalization','K=1 squeeze miscenters private noisy log likelihood; responsibilities1 and final posterior mean verified; private noisy likelihood not used for model selection','2023 exact dependency versions not replicated; unchanged original source passes strict well-conditioned formula and replay tests in recorded existing environment'],non_algorithmic_unknowns=['Complete numeric paper-figure replication recipe/data unavailable','Full4032 resource/convergence gate not run'],native_implementation=False,formal_candidate_models_trained=False,target_evaluation_executed=False,LR4_MFA_comparison_generated=False,final_K=None,final_loading_rank=None,B2_B3_outcomes_viewed=False,target_information_leakage=0,seals_intact=seals['PASS'],preservation=preservation,artifacts=artifacts,STOP=True,next_action='Mentor review only; B5B requires separate authorization')
    write(manifestpath,manifest);value=sha(manifestpath);sidecar.write_text(value+'\n',encoding='ascii')
    print(json.dumps(dict(verdict=manifest['verdict'],manifest=str(manifestpath),SHA256=value,reference_combined_tree_sha256=metadata['selected_reference']['combined_tree_sha256'],stats=stats,files_frozen=len(artifacts))))
if __name__=='__main__':main()
