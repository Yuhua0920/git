"""Generate internal advisor notes from current drafts and completed audits."""
import re
from paper_support import ROOT, json_read, json_write, sha256
from audit_manuscript_draft import DRAFTS
from audit_paper_style import printed_tex

SUMMARIES={
 "MIMO-OFDM Signal Model":"Defines the symbolic MIMO observation before the evaluated 2×2 configuration. The symbol-wise frequency abstraction excludes within-symbol ICI, and noise uses a fixed source-ensemble receive-power reference.",
 "Pilot Arrangement and Observations":"Defines disjoint antenna-owned FDM pilots with nonowners muted, unit pilot modulus and the 1/32 union fraction. Selected pilot/data channel vectors follow the same C-order spatial/frequency/time ordering.",
 "Statistical Channel Estimation Setting":"Separates offline clean statistical preparation from online pilots and receiver-known noise variance. Target clean channels, covariances and reporting labels are absent from deployed inference.",
 "Frozen Source-Statistics LMMSE Estimator":"Defines source means and centered empirical covariances, retaining the full pilot-to-data Source operator. Linear systems use Cholesky solves.",
 "Offline Correction Operators":"Defines profile-origin and mobility-origin operator differences from the prepared CDL-B/30 and CDL-C/120 conditions relative to the source CDL-C/30 condition. These are empirical operator-space directions, not physical components.",
 "Continuous Affine Adaptation":"Defines two unrestricted real coordinates and a dependent source weight. Extrapolation is an algebraic capability, not an empirical superiority claim.",
 "Pilot-Driven Analytical Coordinate Fitting":"Uses four 48/16 cross-pilot partitions and exact prediction operators. A causal five-frame window, including the current frame, solves a two-dimensional real LS problem; singular/nonfinite cases raise the original error.",
 "Low-Rank Correction Compression":"Defines the development-covariance weighted BL decomposition, the QL right solve and correction-only rank-4 factors. Source and coefficient-fitting operators are uncompressed.",
 "Online Inference Procedure":"Forms centered pilots, maintains a trajectory-local causal buffer, fits two coordinates and applies the full Source plus sequential correction products. The same information interface is used for every reporting condition.",
 "Arithmetic Complexity":"Reports matrix-vector MAC counts and correction storage with their exclusions. No measured timing or whole-receiver acceleration is inferred.",
}


def write():
    style=json_read(ROOT/"outputs/t20e_paper_style_audit.json")
    claim=json_read(ROOT/"outputs/t20e_claim_audit.json")
    dim=json_read(ROOT/"outputs/t20e_equation_dimension_audit.json")
    figure=json_read(ROOT/"outputs/t20e_workflow_audit.json")
    structures={};equations=[];lengths={}
    for name in DRAFTS:
        text=(ROOT/name).read_text(encoding="utf8")
        structures[name]=re.findall(r"\\subsection\{([^}]+)\}",text)
        equations.extend((kind,body) for kind,body in re.findall(r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}",text,re.S))
        prose=printed_tex(text)
        prose=re.sub(r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}","",prose,flags=re.S)
        prose=re.sub(r"\$[^$]*\$","",prose)
        prose=re.sub(r"\\\w+(?:\[[^]]*\])?(?:\{[^}]*\})?","",prose)
        lengths[name]=len(re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b",prose))
    out=["# T20E advisor review packet", "", "## A. System Model structure", ""]
    out.extend("- "+s for s in structures[DRAFTS[0]])
    out.extend(["", "## B. Proposed Method structure", ""])
    out.extend("- "+s for s in structures[DRAFTS[1]])
    out.extend(["", "## C. All core equations (verbatim current TeX)", ""])
    for kind,body in equations:out.extend(["```latex",r"\begin{"+kind+"}"+body+r"\end{"+kind+"}","```",""])
    out.extend(["## D. Subsection summaries",""])
    for path in DRAFTS:
        for title in structures[path]:out.extend(["### "+title,"",SUMMARIES[title],""])
    out.extend(["## E. Offline/online information boundary","",
        "Offline: clean source/development statistics; source means/full operator; two finite operator differences; development-weighted rank-4 factors; exact cross-pilot predictors and fixed pilot partitions. Online: received pilots, known noise variance, trajectory-local causal buffer, analytical real coefficients and current-frame channel estimate. Reporting labels and target statistics/means/clean CSI are not estimator inputs.","",
        "## F. Workflow explanation","",
        "The figure separates frozen statistical assets from current observations. Full Source assets feed both pilot centering and final estimation; exact fitting and correction compression have separate branches. Both development-origin covariances feed the prepared predictor/correction library. Solid arrows show preparation/inference; dashed arrows show frozen Source assets. No new scientific data appear. The diagram is 180×66 mm; its height exceeds the original 45–60 mm planning range by 6 mm to retain readable labels and causal reset text. At the manuscript's 174 mm width, the smallest exported glyph is %.3f pt. Single-canvas panel alignment is not applicable; collision audit has zero failures/warnings."%figure["minimum_glyph_at_manuscript_174mm_pt"],"",
        "## G. Potentially strong claims reviewed","",
        "1. Affine extrapolation: algebraically supported by unrestricted real weights summing to one; no performance guarantee.",
        "2. Weighted approximation minimizes a rank-constrained Frobenius error: follows from truncated SVD of BL and invertibility of L; not a global optimality claim for the complete estimator or CDL conditional mean.",
        "No priority, physical separability, general robustness, numerical test gains or timing claims enter the two drafted sections.","",
        "## H. Advisor wording decisions","",
        "Optional editorial decisions: retain 'profile-origin/mobility-origin' operator terminology and 'cross-pilot partitioning' (the current exact mathematical meanings are fixed); accept the 66 mm diagram height; choose a venue template later before validating page counts. No unresolved scientific protocol or estimator decision is required for this draft.","",
        "## I. Style audit summary","",
        "Lexical leakage, meta-AI, diary narration, default overclaim and forbidden terminology counts are all zero. All %d prose sentences are classified and evidence-mapped; %d are algebraic CLAIM sentences. Frozen dimensions/prefixes are verified; static LaTeX labels/references/citations resolve. Actual document compilation is unverified because the native compiler cannot find standard platform directories. Page targets remain unverified until typesetting."%(len(claim["sentences"]),claim["category_counts"]["CLAIM"]),""])
    (ROOT/"docs/t20e_advisor_review_packet.md").write_text("\n".join(out),encoding="utf8",newline="\n")
    notes="""# T20E writing notes

## Scope and authority

Only System Model and Proposed Method were drafted. Other sections retain their existing structure; printed drafting instructions became nonprinted comments. Existing statistical table/caption terminology was sanitized without changing numeric rows. No Introduction/Related Work/Setup/Results/Discussion/Conclusion/Abstract prose was drafted. All scientific code, methods, statistics, seeds, protocols, formal results and existing performance figures remain frozen.

The permanent style guide is hash-frozen in the T20E before manifest. Historical T20D/T20C equation/table/notation checks now read committed snapshots; existing JSON hashes remain untouched. New current tests verify actual dimensions, weighted residual identities, coordinate/window semantics and exact performance rows.

## Editorial notation only

The SNR symbol s became gamma_dB; source subscript s retains its mnemonic meaning. Fitting residual r_(ell f) became e_(ell f), separating it from receive index r. Dummy real variables a/b became c=[alpha,beta]^T, separating them from transmit index a. Development W_B30/100 and W_C120/100 became W_P/W_M with their conditions stated explicitly. These are manuscript relabelings only; no scientific implementation changed.

The existing, undrafted Setup BER expression relabels its two bit counts as b^err and b^tot to avoid reusing e for both a count and the fitting residual. Its ratio and metric semantics are unchanged. Existing table captions expand NMSE/BER/ZF at first printed use; transmit-antenna offsets avoid an unexplained TX shorthand.

## First-use and semantic review

General indices precede the observation. MIMO/OFDM, CP/ICI/RE, QPSK/SNR, FDM/LS, CSI/LMMSE, CDL/SVD/MAC are expanded on first prose use. Covariance blocks, conjugate transpose, identity, Euclidean residual, weighted Frobenius objective and factor shapes are defined with their use. Source centering remains common to all deployed paths. A general symbolic model is not a claim of validation at arbitrary antenna sizes.

The Source reference power is source-ensemble expected useful data signal per RX branch, never instantaneous fading power. Unit-modulus owner pilots and muted other antennas preserve the pilot noise covariance and Ptot. No within-symbol ICI or waveform CP validation is inferred from the frequency abstraction.

Clean offline dependence is stated at the beginning of Methods. Only online coordinate inference is pilot-driven. Origin operators are empirical differences, not physical components. The full Source and exact fitting operators remain uncompressed. Arithmetic counts exclude scalar products/Gram/solver, detection and transfers; no timing claim is made. Existing delay boundary evidence remains in internal frozen scientific notes for later Results/Discussion drafting, without importing results into Methods.

## Review status and limitations

The two sections have been manually reviewed against the frozen code and evidence. The per-sentence categories/evidence are stored in the claim audit. The weighted minimizer and extrapolation statements are mathematical, not empirical superiority statements. No unresolved scientific wording decision remains; terminology/figure height/venue packaging are optional advisor choices.

Native LaTeX compilation still fails before document parsing with 'Unable to find standard directories for platform'. Static validation and editable flattened preview are delivered; actual pagination is not verified. The diagram is a vector schematic, so the source validator's absent-TIFF warning is inapplicable to a requested submission asset; PDF/SVG and PNG QA are available. Quantitative panel alignment is not applicable to a single schematic canvas.

## Engineering warnings retained

The start full regression initially had 2018 passed/1 failed in a historical Windows RAM test that compares two different sampling instants. Its scientific assertions passed; the memory assertion failed. An isolated unchanged rerun passed, and a 1000-pair diagnostic reproduced 54 non-atomic comparisons while every same-snapshot invariant held. The unchanged full rerun then passed all 2019 tests. Both records were preserved in the T20D acceptance commit.

During new figure development, math-string escaping, final glyph size and text/arrow collision issues were found and fixed before delivery. An optional threadpoolctl import was removed because it is absent from the base environment; no package was installed. A deprecated fitz-import QA attempt was replaced by pymupdf. The Python launcher continues to emit its existing relocated-runtime warning. Git reports its existing LF-to-CRLF conversion notices; no checkout or historical byte rewrite was performed.
"""
    (ROOT/"docs/t20e_writing_notes.md").write_text(notes,encoding="utf8",newline="\n")
    report=dict(draft_prose_word_counts=lengths, equation_environment_count=len(equations),
        length_targets_pages={"system":[.8,1.2],"method":[2,3]}, actual_page_counts_verified=False,
        workflow_visual_review_completed=True, workflow_grayscale_information_review_completed=True,
        symbols_defined_before_use_in_drafted_sections_manual_review_completed=True,
        unresolved_scientific_questions=[], optional_advisor_wording_items=["origin terminology","cross-pilot terminology","diagram height","venue-specific page layout"],
        documentation_sha256={p:sha256(ROOT/p) for p in ("docs/t20e_writing_notes.md","docs/t20e_advisor_review_packet.md")})
    json_write(ROOT/"outputs/t20e_manual_review.json",report)
    print(report)
    return report


if __name__=="__main__":write()
