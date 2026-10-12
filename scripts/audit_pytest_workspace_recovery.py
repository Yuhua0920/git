"""Permission-only integrity/capability/preflight evidence; no mechanism scan."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'outputs/t17a_permission_recovery'
sys.path.insert(0,str(ROOT))


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name,value):
    path=AUDIT/name
    if path.exists():raise ValueError('existing permission record protected:'+name)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def integrity(phase):
    before=load(AUDIT/'before.json')
    groups={key:before[key] for key in ('cache_sha256','outputs_sha256','preparation_sha256')}
    g=load(ROOT/'outputs/t16g_results.json')
    groups['t16g_frozen_artifacts']={**g['source_sha256'],**g['frozen_history_sha256'],
        **g['numeric_artifact_sha256'],**g['preparation_artifact_sha256'],
        **g['preflight_schema_recovery']['artifact_sha256']}
    runtime={'.pytest_cache/v/cache/nodeids','.pytest_cache/v/cache/lastfailed'}
    mismatches={};runtime_changes={}
    for group,mapping in groups.items():
        changed={p:dict(before=h,after=sha(ROOT/p)) for p,h in mapping.items() if sha(ROOT/p)!=h}
        if group=='cache_sha256' and phase=='after_tests':
            runtime_changes={p:v for p,v in changed.items() if p in runtime}
            changed={p:v for p,v in changed.items() if p not in runtime}
        mismatches[group]=changed
    a=load(AUDIT/'acl_after.json')
    old={v['path']:v for v in before['cache_acl']};new={v['path']:v for v in a['cache_acl']}
    owner_drift=[p for p,v in old.items() if v['owner']!=new[p]['owner']]
    protection_drift=[p for p,v in old.items() if v['inheritance_protected']!=new[p]['inheritance_protected']]
    result=dict(phase=phase,group_file_counts={k:len(v) for k,v in groups.items()},
        current_cache_file_count=sum(p.is_file() for p in (ROOT/'.pytest_cache').rglob('*')),
        current_outputs_file_count=sum(p.is_file() for p in (ROOT/'outputs').rglob('*')),
        content_mismatches=mismatches,runtime_cache_metadata_changes=runtime_changes,
        cache_owner_mismatches=owner_drift,cache_inheritance_protection_mismatches=protection_drift,
        runtime_cache_metadata_classification='pytest-written nodeids/lastfailed; not scientific artifacts',
        new_temp_files_classification='new pytest runtime fixtures; original cache files retained',
        all_checks_passed=not any(mismatches.values()) and not owner_drift and not protection_drift)
    save('integrity_'+phase+'.json',result)
    print(json.dumps({k:result[k] for k in ('phase','group_file_counts','all_checks_passed')},indent=2))
    if not result['all_checks_passed']:raise ValueError('permission integrity gate failed')


def cache_writer_probe():
    """Use pytest's own writer with unchanged values, never invent failure data."""
    import pytest
    config=pytest.Config.fromdictargs({},['-q'])
    config._do_configure()
    try:
        proof=[]
        for key in ('nodeids','lastfailed'):
            path=ROOT/'.pytest_cache/v/cache'/key
            value=config.cache.get('cache/'+key,None)
            if value is None:raise ValueError('existing pytest metadata required')
            original=load(path);original_hash=sha(path)
            if value!=original:raise ValueError('pytest cache read mismatch')
            config.cache.set('cache/'+key,value)
            if load(path)!=original:raise ValueError('cache semantic values changed')
            proof.append(dict(key=key,writer='pytest Cache.set',values_unchanged=True,
                before_sha256=original_hash,after_sha256=sha(path),write_succeeded=True))
        save('minimal_cache_writer.json',dict(passed=True,metadata=proof,
            manual_cache_content_edit=False,failed_test_history_invented=False,
            description='pytest API rewrites existing values only; write-capability probe, not regression evidence'))
        print('PASS: pytest Cache.set writes nodeids and lastfailed; existing values retained')
    finally:
        config._ensure_unconfigure()


def preflight():
    # Only the locations of fresh transcripts and the new audit output change.
    # Existing scientific/provenance functions and their acceptance criteria stay intact.
    from scripts import check_wiener_preflight as check
    check.OUT=AUDIT
    code=check.check()
    if code:raise ValueError('current preflight failed')
    raw=load(AUDIT/'t17a_execution_preflight.json')
    full=load(AUDIT/'full_execution.json');math=load(AUDIT/'math_execution.json')
    if full['exit_code'] or math['exit_code']:raise ValueError('actual successful transcripts required')
    if '25 passed' not in math['summary']:raise ValueError('25 mathematics tests required')
    c6=(ROOT/'docs/t17a_claim_evidence_matrix.md').read_text(encoding='utf8')
    if 'UNCONFIRMED' not in next(x for x in c6.splitlines() if x.startswith('| C6 ')):
        raise ValueError('C6 prematurely upgraded')
    if (ROOT/'outputs/t17a_results.json').exists():raise ValueError('scientific run prohibited during permission repair')
    report=dict(stage='permission recovery preflight only',all_checks_passed=True,
        raw_provenance_record='outputs/t17a_permission_recovery/t17a_execution_preflight.json',
        raw_provenance_sha256=sha(AUDIT/'t17a_execution_preflight.json'),
        generator_calls=raw['generator_calls'],new_statistics_fit=raw['new_statistics_fit'],
        new_channel_generated=raw['new_channel_generated'],formal_test_used=raw['formal_test_used'],
        formal_test_sealed=raw['formal_test_sealed'],C6='UNCONFIRMED',mechanism_run_performed=False,
        full_pytest_summary=full['summary'],math_pytest_summary=math['summary'],
        historical_warnings_in_raw_recorder='raw recorder retains prior-attempt warning strings; current math log has no cache warning')
    save('permission_preflight.json',report)
    print('PASS: frozen provenance / no generator / no fitting / sealed formal test / C6 UNCONFIRMED')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['after_acl','after_tests','cache_writer','preflight'])
    mode=parser.parse_args().mode
    if mode in ('after_acl','after_tests'):integrity(mode)
    elif mode=='cache_writer':cache_writer_probe()
    else:preflight()
