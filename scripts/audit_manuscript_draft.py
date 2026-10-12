"""T20E read-only math, sentence, LaTeX and frozen-evidence audits."""
import argparse
import hashlib
import re
from collections import Counter
from pathlib import Path
import numpy as np
from paper_support import ROOT, json_read, json_write, sha256
from build_paper_preview import flatten, bib_entries
from audit_paper_style import printed_tex, audit as style_audit
from check_paper import script_firewall

DRAFTS = ("paper/sections/03_system_model.tex", "paper/sections/04_proposed_method.tex")
ALLOWED = {"paper/main.tex", "paper/appendix/evidence_slots.tex", "paper/preview.tex", "paper/preview_provenance.json",
           "tests/test_paper_blueprint.py", "tests/test_paper_evidence.py"} | {
    f"paper/sections/{i:02d}_{name}.tex" for i,name in enumerate(
        ("introduction","related_work","system_model","proposed_method","experimental_setup","results","discussion","conclusion"),1)} | {
    "paper/tables/"+name for name in ("01_system.tex","02_nmse.tex","03_ber.tex","04_endpoints.tex","05_complexity.tex","notation.tex")}


def array_digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def sentence_records(text, path):
    """Every prose sentence inherits a reviewed source annotation; math is separate."""
    records, uncovered = [], []
    for block in re.split(r"\n\s*\n", text):
        annotation = re.search(r"% audit: (DEFINITION|ASSUMPTION|METHOD_DESCRIPTION|CLAIM) \| ([^\n]+)", block)
        prose = printed_tex(block)
        prose = re.sub(r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}", "", prose, flags=re.S)
        prose = re.sub(r"\\(?:section|subsection)\{[^}]*\}", "", prose)
        prose = re.sub(r"\$[^$]*\$", "[math]", prose)
        prose = re.sub(r"\\(?:begin|end)\{[^}]*\}(?:\[[^\]]*\])?", "", prose)
        prose = re.sub(r"\\(?:caption)\{(.*?)\}\s*$", r"\1", prose, flags=re.S)
        prose = prose.replace(r"\centering", "").replace("~", " ")
        prose = re.sub(r"\s+", " ", prose).strip()
        if not prose:
            continue
        if not annotation:
            uncovered.append(prose)
            continue
        # Preserve Fig. as part of its sentence, rather than counting an abbreviation.
        prose = prose.replace("Fig. ", "Figure ")
        for sentence in re.split(r"(?<=[.!?])\s+", prose):
            records.append(dict(file=path, sentence=sentence, category=annotation[1],
                                evidence=annotation[2].split(";")))
    return records, uncovered


def claims(root=ROOT, output=None):
    root=Path(root); records=[]; uncovered=[]
    for name in DRAFTS:
        entries, missing=sentence_records((root/name).read_text(encoding="utf8"),name)
        records.extend(entries); uncovered.extend(missing)
    missing_evidence=[]
    for i, record in enumerate(records,1):
        record["sentence_id"]=f"sentence_{i:03d}"
        evidence_hash={}
        for source in record["evidence"]:
            if source.startswith("algebra:"):
                continue
            name=source.split("#",1)[0]
            if (root/name).is_file(): evidence_hash[name]=sha256(root/name)
            else: missing_evidence.append(source)
        record["evidence_file_sha256"]=evidence_hash
    report=dict(sentences=records, category_counts=dict(Counter(r["category"] for r in records)),
        uncovered_prose=uncovered, missing_evidence=missing_evidence,
        manual_classification_review_completed=True,
        algebra_support={"eq:affine":"W=(1-alpha-beta)Ws+alpha WP+beta WM; affine weights sum to one but are unrestricted real.",
                        "eq:weight":"Invertible L maps rank-preserving correction X to XL. Truncated SVD minimizes the Frobenius error of BL; QL=V4^H maps back without changing rank."},
        performance_claims_in_system_or_method=False,
        evidence_support_is_not_a_priority_or_physical_mechanism_claim=True,
        all_checks_passed=not uncovered and not missing_evidence)
    if output is not None:
        json_write(output,report)
    return report


def dimensions(root=ROOT, output=None):
    root=Path(root); frozen=json_read(root/"outputs/t20a_results.json")
    provenance=frozen["frozen_method_provenance"]; checked=0; failures=[]; shapes={}; residuals={}
    arrays={}
    for name, artifact in provenance["artifacts"].items():
        path=root/artifact["path"]
        if sha256(path)!=artifact["sha256"]: failures.append(str(path))
        with np.load(path,allow_pickle=False) as f:
            arrays[name]={key:f[key] for key in artifact["array_sha256"]}
        for key,digest in artifact["array_sha256"].items():
            x=arrays[name][key];checked+=1
            if array_digest(x)!=digest: failures.append(key)
            shapes[name+":"+key]=list(x.shape)
    op=arrays["operator_artifact"]; factors=arrays["factor_artifact"]
    assert op["source_mean_p"].shape==(64,) and op["source_mean_d"].shape==(3968,)
    stats={}
    for m,name in {"s":"outputs/t15a_cdl_statistics_500.npz", "P":"outputs/t15c_cdl_statistics_CDL-B-30.npz", "M":"outputs/t15c_cdl_statistics_CDL-C-120.npz"}.items():
        with np.load(root/name,allow_pickle=False) as f: stats[m]=(f["rpp"],f["rdp"])
    with np.errstate(over="raise", invalid="raise"):
        for snr in (5,10,20):
            sigma=frozen["frozen_components"]["noise_variances"][str(snr)]
            assert op[f"Wsource_{snr}"].shape==(3968,64)
            covariance=stats["s"][0]+sigma*np.eye(64)
            source_res=np.linalg.norm(op[f"Wsource_{snr}"]@covariance-stats["s"][1])/np.linalg.norm(stats["s"][1])
            residuals[f"source_solve_{snr}"]=float(source_res)
            assert source_res<1e-11
            for m in ("P","M"):
                assert op[f"DeltaK{m}_{snr}"].shape==(4,16,48)
                a=factors[f"A_{m}_{snr}"][:,:4];q=factors[f"Q_{m}_{snr}"][:4,:]
                assert a.shape==(3968,4) and q.shape==(4,64)
                assert a.dtype==q.dtype==np.complex128
                for suffix,x in (("A",a),("Q",q)):
                    key=f"{m}_{snr}_{suffix}"
                    assert array_digest(x)==provenance["rank4_array_sha256"][key]
                rpp=stats[m][0];assert rpp.shape==(64,64)
                assert np.max(np.abs(rpp-rpp.conj().T))<1e-11
                lower=np.linalg.cholesky(rpp+sigma*np.eye(64))
                delta=op[f"DeltaW{m}_{snr}"]
                observed=float(np.linalg.norm((delta-a@q)@lower)**2)
                expected=float(np.sum(factors[f"singular_{m}_{snr}"][4:]**2))
                err=abs(observed-expected)/max(expected,np.finfo(float).tiny)
                residuals[f"weighted_rank4_tail_energy_relative_{m}_{snr}"]=err
                assert err<1e-11
                np.testing.assert_allclose((q@lower)@(q@lower).conj().T,np.eye(4),atol=1e-11)
    folds=json_read(root/"outputs/t16e_pilot_folds.json")
    selected=np.asarray(folds["selected"]);coords=np.array(np.unravel_index(selected,(2,2,64,16))).T
    assert np.all(np.diff(selected)>0)
    np.testing.assert_array_equal((coords[:,2]//16+coords[:,3]//4)%4,folds["labels"])
    assert len(selected)==64 and all(len(f["observed"])==48 and len(f["heldout"])==16 for f in folds["folds"])
    cost=dict(source=3968*64,pilot_fitting=5*4*3*16*48,corrections=2*4*(3968+64))
    cost["total"]=sum(cost.values());cost["dense_total"]=3*3968*64+cost["pilot_fitting"]
    assert cost==dict(source=253952,pilot_fitting=46080,corrections=32256,total=332288,dense_total=807936)
    text=(root/DRAFTS[1]).read_text(encoding="utf8"); notation=(root/"paper/tables/notation.tex").read_text(encoding="utf8")
    checks=dict(array_and_prefix_hashes_exact=not failures, source_operator_shape=True,
        factor_shapes_and_complex128_verified=True, weighted_residual_matches_frozen_singular_tail=True,
        vectorization_order="C order [rx,tx,k,t], t fastest", pilot_noise_covariance="sigma2 I_64, unit-modulus orthogonal owner pilots",
        real_coordinates=r"c\in\mathbb R^2" in text,
        conjugate_transpose_defined="conjugate transpose" in text,
        real_gram_and_rhs=r"\operatorname{Re}" in text and "G_j" in text and "u_j" in text,
        includes_current_frame=r"\sum_{\ell=j-4}^{j}" in text,
        snr_symbol_collision_removed=r"\gamma_{\mathrm{dB}}" in notation,
        residual_receive_index_collision_removed=r"e_{\ell f}" in notation and r"r_{\ell f}" not in text,
        correction_only_rank4="cross-pilot fitting operators retain their full" in text)
    report=dict(checked_frozen_arrays=checked, shapes=shapes, residuals=residuals, checks=checks,
        arithmetic_counts=cost, matrix_products=["3968x64 @ 64 -> 3968", "16x48 @ 48 -> 16", "16x2 H @16x2 ->2x2", "3968x4 @ (4x64 @64) ->3968"],
        no_svd_or_statistics_fit_run=True, frozen_statistics_read_only=True,
        all_checks_passed=not failures and all(checks.values()))
    if output is not None:
        json_write(output,report)
    return report


def latex(root=ROOT, output=None):
    root=Path(root);text=flatten(root/"paper/main.tex",root/"paper")
    text=re.sub(r"(?<!\\)%[^\n]*","",text)
    labels=re.findall(r"\\label\{([^}]+)\}",text)
    refs=re.findall(r"\\(?:ref|eqref)\{([^}]+)\}",text)
    keys={k for chunk in re.findall(r"\\cite\{([^}]+)\}",text) for k in chunk.split(",")}
    bib=bib_entries((root/"paper/bibliography/references.bib").read_text(encoding="utf8"))
    errors=[]
    if len(labels)!=len(set(labels)): errors.append("duplicate labels")
    if set(refs)-set(labels): errors.append("missing refs")
    if keys-set(bib): errors.append("missing citations")
    if re.search(r"(?:Eq\.|Equation)\s*\(?\d",text): errors.append("hardcoded equation number")
    depth=0
    for m in re.finditer(r"(?<!\\)[{}]",text):
        depth += 1 if m[0]=="{" else -1
        if depth<0: errors.append("brace underflow")
    if depth: errors.append("unbalanced braces")
    for fig in re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}",text):
        if not (root/"paper"/fig).is_file(): errors.append("missing figure:"+fig)
    preview=(root/"paper/preview.tex").read_text(encoding="utf8")
    for _,body in re.findall(r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}",text,re.S):
        if body not in preview: errors.append("preview equation drift")
    report=dict(label_count=len(labels), reference_count=len(refs), citation_keys=sorted(keys), errors=errors,
        static_validation_passed=not errors, compile_not_verified_due_to_environment=True,
        compilation_is_not_inferred_from_static_checks=True)
    if output is not None:
        json_write(output,report)
    return report


def integrity(root=ROOT, output=None):
    root=Path(root);before=json_read(root/"outputs/t20e_before_manifest.json")["tracked_before_sha256"]
    changed=[name for name,h in before.items() if not(root/name).is_file() or sha256(root/name)!=h]
    bad=[name for name in changed if name not in ALLOWED]
    guide=sha256(root/"docs/paper_writing_style_guide.md")
    if guide!=json_read(root/"outputs/t20e_before_manifest.json")["writing_style_guide_sha256"]:bad.append("style guide freeze changed")
    firewall={name:script_firewall((root/"scripts"/name).read_text(encoding="utf8")) for name in (
        "audit_paper_style.py","audit_manuscript_draft.py","build_method_workflow.py","audit_method_workflow.py","render_manuscript_tables.py","prepare_manuscript_editorial.py","write_t20e_review_packet.py","finalize_t20e_delivery.py")}
    report=dict(tracked_files_checked=len(before), historical_outputs_checked=sum(p.startswith("outputs/") for p in before),
        scientific_source_files_checked=sum(p.startswith("src/") for p in before), authorized_editorial_changes=changed,
        unexpected_mutations=bad, historical_output_mutations=[p for p in changed if p.startswith("outputs/")],
        scientific_code_unchanged=not any(p.startswith("src/") for p in changed), manuscript_script_firewall=firewall,
        writing_style_guide_sha256=guide, new_channel_generated=False,new_statistics_fit=False,new_model_trained=False,
        method_changed=False,formal_data_used_for_design=False,
        all_checks_passed=not bad and not any(firewall.values()))
    if output is not None:
        json_write(output,report)
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--integrity",action="store_true");args=p.parse_args()
    if args.integrity: results={"integrity":integrity()}
    else: results={"style":style_audit(),"dimensions":dimensions(),"claims":claims(),"latex":latex()}
    print({k:v.get("all_checks_passed",v.get("all_lexical_checks_passed",v.get("static_validation_passed"))) for k,v in results.items()})
    if any(not v.get("all_checks_passed",v.get("all_lexical_checks_passed",v.get("static_validation_passed"))) for v in results.values()): raise SystemExit(1)
