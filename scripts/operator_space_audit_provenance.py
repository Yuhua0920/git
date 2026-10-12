"""Preserve T19B execution provenance while checking one post-run auditor fix.

No numerical/source execution hash is rewritten. Only the verifier expression
uses the executor's identical floating-point association. Exact reverse-patch
comparison binds the current auditor to its archived pre-execution version.
"""
from scripts.run_operator_space_audit import ROOT,verify_hashes
from scripts.run_cdl_mismatch import sha

AUDITOR='scripts/audit_operator_space_audit.py'
HELPER='scripts/operator_space_audit_provenance.py'
REVERSE_EDITS=(
    ('from scripts.operator_space_audit_provenance import verify_sources\n',''),
    ('    verify_sources(r)\n','    verify_hashes(r)\n'),
    ('                residual=wt-(ws+sum(c*m for c,m in zip(co,mods)))\n',
     '                residual=wt-ws-sum(c*m for c,m in zip(co,mods))\n'),
)


def verify_sources(r):
    execution=ROOT/r['source_snapshot_root']/AUDITOR
    if sha(execution)!=r['source_sha256'][AUDITOR]:
        raise ValueError('original executed auditor snapshot changed')
    current=(ROOT/AUDITOR).read_text(encoding='utf8')
    restored=current
    for new,old in REVERSE_EDITS:
        if restored.count(new)!=1:
            raise ValueError('auditor differs beyond the single documented association fix')
        restored=restored.replace(new,old,1)
    if restored!=execution.read_text(encoding='utf8'):
        raise ValueError('undocumented auditor change from executed snapshot')
    checked=dict(r)
    checked['source_sha256']={path:h for path,h in r['source_sha256'].items() if path!=AUDITOR}
    verify_hashes(checked)
    proof=dict(reason='post-run verifier used reassociated subtraction; executor residual and numerical artifacts are unchanged',
               original_auditor_path=str(execution.relative_to(ROOT)).replace('\\','/'),
               original_auditor_sha256=sha(execution),current_auditor_sha256=sha(ROOT/AUDITOR),
               reverse_patch_to_original_exact=True,helper_path=HELPER,helper_sha256=sha(ROOT/HELPER),
               execution_source_hashes_rewritten=False,scientific_scan_repeated=False)
    if r.get('post_run_auditor_revision') and r['post_run_auditor_revision']!=proof:
        raise ValueError('final verifier provenance drift')
    for path,h in r.get('post_run_source_sha256',{}).items():
        if sha(ROOT/path)!=h:raise ValueError('post-run audit source changed:'+path)
    return proof
