"""Editorial research plans only: no experiment or manuscript dependencies."""
import hashlib
import json

REVIEW_HASHES={
    'docs/sp_u1_reviewer1_frozen.md':'40b5042ca3ef6e879446cd62f1b74d4c7a2a400f607e10b00db2571e142740b3',
    'outputs/sp_u1_reviewer1_frozen.json':'b27838bac15c975f0b04cbce25d74bd703a19508a12774d109f184693cb6e3e2',
    'docs/sp_u1_reviewer2_frozen.md':'a199097af4f3091919e249f4eeefe6c5002c74fa5e79d39f1eb8ce7e44f8f4c8',
    'outputs/sp_u1_reviewer2_frozen.json':'8ff2c4b59e8e9c187408000ad54ac73761f8863bcef260af03c9baa25deaea7e',
    'docs/sp_u1_reviewer3_frozen.md':'92cc0d10a0a8aea1b8b3ee25427579c64109506fa95239716c27ebf921b64c7d',
    'outputs/sp_u1_reviewer3_frozen.json':'a5711d5a6be1d4fe9094e5451fb0a6f9246d1f24f75915dbdfb704f51699d6ad'}

def write_plans(root,md,dump):
    for path,expected in REVIEW_HASHES.items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen reviewer changed: '+path)
    reports=[json.loads((root/f'outputs/sp_u1_reviewer{i}_frozen.json').read_text(encoding='utf-8')) for i in (1,2,3)]
    rows=['# Reviewer attack matrix','','Three isolated reviewers assessed the same frozen reading packet. Their original MD/JSON reports are immutable. Reviewer 1: statistical signal processing; Reviewer 2: wireless CE; Reviewer 3: adaptive/learning CE. These are independent analytical lenses, not journal decisions.','',
          'Only a concern independently raised by at least two reviewers is called consensus. Root audit findings and mandatory coverage questions are separate. Resolution tests in frozen reports are proposals, not completed experiments.','']
    for i,report in enumerate(reports,1):
        concerns=report['concerns']
        rows += [f'## Reviewer {i}: {len(concerns)} concerns','',
                 '| ID | Severity | Class | Attack / locator | Response feasibility |','|---|---|---|---|---|']
        for c in concerns:
            issue=c.get('title',c.get('concern',c.get('issue_key','UNKNOWN')))
            pointer=c.get('claim_pointer',c.get('evidence_pointer','see frozen JSON'))
            response=c.get('resolution_test','see frozen JSON')
            cells=[c['id'],c.get('severity','UNKNOWN'),c['classification'],issue+'; '+pointer,response]
            rows.append('| '+' | '.join(str(s).replace('|','/').replace('\n',' ') for s in cells)+' |')
    consensus=[
      ('Closest work and affine-LS collision',['R1-M01','R2-M01','R3-M1']),
      ('Pilot-loss versus data-risk alignment',['R1-M02','R2-M02','R3-M3']),
      ('Offline covariance-library budget/provenance',['R1-M07','R2-M04','R3-M8']),
      ('Modern comparable adaptation baseline',['R1-M09','R2-M05','R3-M2']),
      ('Unrestricted real coordinates and conditioning',['R1-M04','R2-M06','R3-M6']),
      ('Source centering and noncentral input moments',['R1-M06','R2-M07','R3-M5']),
      ('Rank truncation versus actual joint target risk',['R1-M05','R2-M08','R3-M4']),
      ('Mechanism/design provenance versus genuinely unused conditions',['R1-M10','R2-M09','R3-M9']),
      ('Frame-causal window/startup interpretation',['R1-m01','R2-m02','R3-m2']),
      ('Offline storage and SNR-indexed libraries',['R2-m01','R3-m3'])]
    # Reviewer 3 uses an uppercase M without the zero in its IDs; validate below.
    ids={c['id'] for report in reports for c in report['concerns']}
    # Preserve report identifiers verbatim, resolving only case/zero differences here.
    def resolve(name):
        if name in ids:return name
        norm=lambda s:s.replace('-M0','-M').replace('-m0','-m')
        hits=[s for s in ids if norm(s)==norm(name)]
        if len(hits)!=1:raise ValueError('Unresolved independent-review ID: '+name)
        return hits[0]
    consensus=[{'issue':s,'independent_ids':[resolve(x) for x in refs]} for s,refs in consensus]
    rows += ['','## Independent consensus','','| Issue | Independent report IDs |','|---|---|']
    rows += [f"| {c['issue']} | {', '.join(c['independent_ids'])} |" for c in consensus]
    rows += ['','## Required coverage questions (root synthesis, not extra reviewer votes)','',
             '| Attack | Class | Proposed answer / work |','|---|---|---|',
             '| Affine combination is old. What is new? | WRITING_FIX | ls2010 directly establishes difference-output LS; integrated information contract plus adaptation-specific analysis must carry contribution |',
             '| Low-rank approximation is standard. | THEORY_NEEDED | Generic SVD not a contribution; analyze coordinate-amplified correction error and origin/target weight mismatch |',
             '| Why only two directions? | WRITING_FIX | Prepared design assumption, with explicit representation failure boundary; do not add modules from old test |',
             '| Can sparse pilots identify coordinates? | THEORY_NEEDED | Real stacked excitation plus surrogate/data risk alignment; no unconditional guarantee |',
             '| What guarantees identifiability? | THEORY_NEEDED | Real Gram positive definiteness, distinct from population target identifiability |',
             '| Sensitivity to pilot density? | MODERATE_EXPERIMENT | New paired dF=8/16/32 plan only; conditioning and overhead jointly reported |',
             '| How much clean offline CSI? | MODERATE_EXPERIMENT | Separate Source/P/M budgets, independent N-stat blocks and origin uncertainty |',
             '| Is clean development CSI realistic? | WRITING_FIX | Controlled oracle preparation assumption; full sounding/simulation corpus cost explicit, noisy-statistics sensitivity conditional |',
             '| Where are modern adaptive/learning baselines? | MODERATE_EXPERIMENT | MFA/A-MMSE conditional shortlist; all fairness, licensing and reproduction gates precede implementation |',
             '| Outside 2×2 CDL-B/C? | MODERATE_EXPERIMENT | One genuinely unused CDL profile after history audit; 4×4 is LARGE_NEW_RESEARCH, deferred |',
             '| Is 120 km/h physical high-mobility robustness? | WRITING_FIX | Diagonal time-varying grid lacks ICI; full waveform ICI is OUT_OF_SCOPE for Paper 1 |',
             '| Target outside prepared affine space? | ALREADY_ANSWERED | Existing delay boundary demonstrates finite-space limitation; decomposition should quantify, not remove, this failure |',
             '| Why not update covariance online? | MODERATE_EXPERIMENT | Consider modern statistical mixture/CCF selection under same observations and preparation budget; full online Rdp estimation needs unavailable data truth |',
             '| Difference from subspace tracking? | WRITING_FIX | Özdemir abstract comparison only; detailed algorithm UNKNOWN, risk unresolved |',
             '', 'Top ten priorities: novelty closure; actual pilot/data-risk relation; random/endogenous coefficient error; Gram tails; offline cost; external comparator; joint compression risk; source-mean contribution; unused-profile generalization; observation-history fairness. This order is an advisor judgment, not a vote total or a numerical publication probability.']
    md('reviewer_attack_matrix','\n'.join(rows))
    dump('outputs/sp_u1_review_synthesis.json',{'reviewer_sha256':REVIEW_HASHES,
        'frozen_review_count':3,'concern_count':sum(len(r['concerns']) for r in reports),
        'consensus':consensus,'root_audit_separate':True,'reviewers_revised_after_freeze':False})

    md('theory_feasibility',r'''
    # Actual-algorithm coordinate theory: feasibility, not a manuscript theorem

    Verdict: **adaptation-specific propositions/application-specific analysis are feasible; a general new theorem is not established**. Generic affine LS, reduced-rank filtering and combination error analysis are prior art. [Azpicueta-Ruiz et al. 2010](https://doi.org/10.1109/ISCAS.2010.5538026), Sec II A/B, already adapts windowed affine combinations through output differences. [Gui 2014](https://arxiv.org/pdf/1407.6083v1) applies affine adaptive filters to CE. [2022 NLMS combination](https://doi.org/10.1016/j.sigpro.2022.108465) supplies correlated/nonstationary analysis context at abstract level; its exact proof is UNKNOWN. None licenses an independent-noise proof for our different stochastic pilot-CV design.

    ## 1. Canonical actual algorithm

    Frozen definitions: `src/pilot_cv.py`, `docs/t20a_final_method_definition.md`, `paper/sections/04_proposed_method.tex`. At each SNR, full Source W_s and two dense differences B_P,B_M are frozen; separate prepared-origin weighted rank-4 factors approximate corrections only. Four folds of the joint 64-entry received pilot vector predict withheld pilots from the other 48. Fold geometry is `(k//16+t//4)%4`. The estimator stacks current and four preceding frames, resets each trajectory, scores frames 4–9, and solves a real 2×2 Gram system with `np.linalg.solve`. No ridge, clipping, pseudoinverse or rejection fallback is present. Known noise variance and Source means are allowed; target means/statistics, truth and reporting label are not online inputs. Final coefficients are fitted with dense pilot predictors and then applied to compressed data predictors.

    For each frame/fold, let D=[ΔK_P z_o, ΔK_M z_o] and e=z_h−K_s z_o. Stack these as complex D_all,e_all; define the **real** design B=[Re D_all; Im D_all] and response r_R=[Re e_all; Im e_all]. Then c_hat=(BᵀB)⁻¹Bᵀr_R when the actual Gram is nonsingular. B is not the Gram itself. Five frames do not imply 320 independent complex observations.

    ## 2. Feasible statements and their limits

    | Object | Defensible statement | What it does not establish |
    |---|---|---|
    | Empirical coordinate identifiability | rank over the reals of B is 2 iff BᵀB is positive definite | Not global identification of target covariance/operator |
    | Conditioning | λ_min, λ_max and condition number of real Gram control worst-case response perturbation; column-normalized coherence separates excitation scale from collinearity | Pilot count alone does not imply conditioning or monotonic improvement |
    | Coefficient error | For any fixed reference c*, write r_R=Bc*+ε_R; then δc=(BᵀB)⁻¹Bᵀε_R exactly | ε_R is not automatically independent, centered or white |
    | Conditional covariance | With A=(BᵀB)⁻¹Bᵀ, E[δc\|B]=Aμ_{ε\|B}, Cov[δc\|B]=AΣ_{ε\|B}Aᵀ | σ²(BᵀB)⁻¹ requires explicit exogeneity and homoscedasticity; not the deployed guarantee |
    | Data propagation | B_P δα z+B_M δβ z is the exact coordinate-error vector | z and δc share current pilots; replacing their joint law by independent variances is unjustified |
    | Compression | Error is Σ_m c_hat_m(B_{m,4}−B_m)z; triangle/operator-norm bounds hold pointwise | Fixed-origin singular tails alone do not bound adaptive target error |
    | Representation | For a fixed population-optimal coordinate, target W_t may have nonzero residual outside span(B_P,B_M) | Two modules cannot represent arbitrary changes |

    **Real versus complex rank matters.** If the two complex columns satisfy d_M=i d_P, complex rank is one but real stacked rank can be two. Requiring complex rank two would unnecessarily change the theoretical statement for real α,β. Do not change the estimator to complex coordinates to simplify a proof.

    ## 3. Pilot loss is a surrogate

    Define separately the population pilot objective J_p(c)=E||e−Dc||² and data objective J_d(c)=E||h−[μ_ds+(W_s+c_P B_P+c_M B_M)(p−μ_ps)]||². Derive their respective real Hessians and linear terms. Their minimizers agree only under stated relations between those terms (and the mean contribution); equality is not implied by shared pilots, low rank or zero noise.

    A useful impossibility example has two valid joint channel distributions with identical pilot means/covariance but different data–pilot cross-covariance. Received pilots have identical laws; their best data-domain Wiener operators can differ. Thus arbitrary target operators are not identifiable from sparse pilots alone. Prepared restrictions may resolve a subset, but require a separate injectivity/risk-alignment argument. This is a proposed analysis, not an executed channel experiment.

    Within a fold, heldout AWGN is independent of observed-fold AWGN. Across folds, the same noisy pilot serves as response in one fold and regressor in another; errors share channels, and frames are temporally correlated. Model residuals can also depend on regressors. Conditional moments above are valid algebra, but the needed moments/bias terms are not supplied by an iid-noise assumption. No claim of unbiased coordinates, exact Gaussian uncertainty or 1/number-of-pilots variance decay is presently justified.

    ## 4. Exact vector decomposition, with mean bias

    Let p be the **noisy** received pilot observation, z=p−μ_ps, and let h_t*=μ_dt+W_t(p−μ_pt) be the target affine LMMSE estimate using actual target first/second moments. Put e_t=h_t*−h and b_μ=μ_ds−μ_dt+W_t(μ_pt−μ_ps). For a fixed reference c*, B(c)=c_P B_P+c_M B_M, the deployed estimate obeys:

    h_hat−h = e_t + {b_μ+[W_s+B(c*)−W_t]z} + B(c_hat−c*)z + Σ_m c_hat_m(B_{m,4}−B_m)z.

    This **vector equality** separates matched observation/prior residual, representation plus mean bias, coordinate error, and compression. Mean bias can alternatively be a fifth component. The squared norm equals the sum of component energies **plus every pairwise real cross term**. Four-vector Cauchy–Schwarz gives ||Σe_j||²≤4Σ||e_j||²; it does not give equality of four scalar MSE contributions. Adaptive terms are nonlinear in p, so ordinary LMMSE orthogonality to linear functions cannot eliminate all cross terms. Noise belongs to p and e_t; do not add it twice. No target-means advantage may be silently folded into deployed controls.

    ## 5. Origin-weighted SVD is not a target risk guarantee

    Frozen factors use centered C_{z,m}=R_{pp,m}+σ²I at the prepared origin. Actual Source-centered second moment there is C_{z,m}+(μ_pm−μ_ps)(μ_pm−μ_ps)ᴴ. At a new target it changes again. Keep the historical weighting unchanged, but state this distinction. For fixed c*, matrix second moments yield a quadratic prediction; for adaptive c_hat(p), mixed coordinate/input moments and module cross terms remain. A conditional bound may use ||c_hat|| and ||(B_{m,4}−B_m)z||, with its looseness explicit.

    ## 6. Stop criteria / next proof work

    Stop a theory-contribution claim if analysis only restates ordinary LS covariance/SVD error, if it requires target truth online, or if meaningful guarantees require modifying the frozen solve/window/folds. Keep useful explanatory derivations but label them accordingly. Proceed to an adaptation-specific claim only after checking affine-filter theory, proving actual risk-alignment conditions or their failure, and verifying vector/cross-term identities under explicit assumptions. No new theorem has been added to the manuscript, and no new numeric sentinel has run in SP-U1.
    ''')

    md('modern_baseline_gate',r'''
    # Modern baseline gate — conditional shortlist, no implementation authorized

    Recommend **two conditional candidates**: complex MFA (statistical) and A-MMSE v5 (learned linear filter). Reserve Mei 2021 CCF selection only if the statistical candidate cannot meet cost/reproduction gates. Maximum three, not all papers below. Implementation GO requires fair + reproducible + relevant; none currently clears all three for the joint 2×2 sparse-FDM protocol. Code-presence is not a successful reproduction.

    ## Audited practical information

    | Candidate | Relevance; code/license | Framework / Python / GPU | Cost and adaptation budget | Observations, labels, clean CSI | Pilot/model/dimension | Fairness decision |
    |---|---|---|---|---|---|---|
    | [MFA 2023](https://arxiv.org/pdf/2304.14809v2) | conditional mixture of compact LMMSE priors; [author cplx-mfa](https://github.com/benediktfesl/cplx-mfa), pinned 9f9c22a; BSD-3-Clause; fitting package, not full CE wrapper | Python package; exact current-environment compatibility untested; GPU not established as a requirement | EM on offline clean corpus; component count/latent rank/iterations selected only on new U-Dev; joint 4096 dimensions may cause dense-covariance memory costs; actual seconds UNKNOWN | clean offline corpus; received pilots online; no supervised target labels in posterior rule, but target corpus if training on target must be explicit | paper measured 64-antenna noisy vector; extension to masked joint 2×2 time-frequency vector needed | CONDITIONAL. Derive conditional observation map and account for full corpus/library budget; reproduce small author problem first; no claim of a reproduced OFDM baseline |
    | [A-MMSE v5 2026](https://arxiv.org/html/2506.00452v5) | learned linear whole filter / rank extension; [author repo](https://github.com/TaeJun1999/Attention-aided-MMSE), pinned fa2647f; **license not verified** | TensorFlow/Keras; ≥2.16 uses legacy tf_keras mode in README; Python compatibility untested; author v1 uses RTX4090; exact GPU need/runtime UNKNOWN | repo 150 epochs, batch 50, Adam 1.5e−4 per SNR; offline true-H MSE/Huber; exported fixed population filter; no verified online-update entry | true H offline labels; fixed filter + pilots online; target supervised training is additional information if offered | v1 TDL-A–E; v5 SISO COST2100, 72×14/.mat; default global-token export; current joint spatial grid differs | CONDITIONAL; license/data/version and small reproduction first. Joint port or per-link training changes spatial information; declare it. Never implement attention as our method |
    | [Mei 2021](https://doi.org/10.1049/cmu2.12250) | candidate CCF LMMSE interpolation-error selection; author code/license UNKNOWN | framework/Python/GPU UNKNOWN; likely matrix operations is an inference, not verified executable | covariance-candidate library; online candidate scores; exact code cost UNKNOWN | candidate correlations offline; pilot interpolation errors online; target-label policy UNKNOWN | inspected OFDM comb/block setting; 2D MIMO port unverified | RESERVE, conditional on full algorithm reproduction and matching library/observation budget |
    | [Channelformer 2023](https://arxiv.org/html/2302.04368v1) | received-pilot online weight adaptation; [author MATLAB repo](https://github.com/dianixn/Channelformer), pinned 6312c79, GPL-2.0 | mainly MATLAB code; OFDM.py is not full Python trainer; new environment and license-compatible integration required; exact GPU need UNKNOWN | offline clean init; online gradients/pruning; v1 reports 32,069 online parameters vs 117,659 full | online labels from special label pilots, not ground-truth payload in inspected path; denoising or 5-dB pilot boost | SISO; custom double-symbol DMRS; boosted slot energy ≈1.13 dB in author analysis | NOT shortlisted for fixed-1/32 primary comparison: extra pilot/label power and gradient budget. Can be related work or separately budgeted later |
    | [Lee/Choi PEFT 2026](https://doi.org/10.1109/OJCOMS.2026.3697928) | environment base + SNR adapter switching; code/license UNKNOWN | framework/Python compatibility UNKNOWN; author runtime on RTX4080 SUPER; necessity of GPU untested | 500 supervised target samples per adapter; four SNR adapters or one500 mixed-SNR adapter; backbone1,712,152 parameters,7.168 MB; LoRA192 KB nominal/0.196 MB serialized; K=4 total7.951 MB | Eq7 true-H supervised target NMSE; label-free environment-base selection UNKNOWN | NYUSim RMa→InF SISO; first-symbol comb-6,3072×14,512 pilots | RELATED WORK only: target-supervision/base-selection budget cannot presently align with pilot-only deployment |
    | [Ridge A-MMSE 2026](https://arxiv.org/html/2608.15945v1) | noisy-label bias and ridge denoising; code/license UNKNOWN | executable framework/Python/GPU UNKNOWN | noisy **full-grid sounding** and known noise; covariance/ridge targets | no clean-H labels in stated setting, but extra sounding is not free sparse pilots | COST2100/OFDM; input-dependent operator formulation differs from v5 fixed export | RELATED WORK/noisy-preparation context; not a substitute for the fixed-filter v5 code |
    | [PBCE 2025](https://arxiv.org/html/2504.16710v3) | parameter-conditioned Bayesian LMMSE; code/license UNKNOWN | framework/runtime/GPU UNKNOWN | channel-parameter inference; high-SNR/long-coherence analysis; joint FDM budget UNKNOWN | noisy observations; geometry/model knowledge; label requirement UNKNOWN | mmWave parametric model | not selected: model/interface mismatch and no verified executable |
    | [CSGMM 2025](https://arxiv.org/html/2502.18369v1) | structured noisy/undersampled prior EM | author executable/license/Python/GPU UNKNOWN | physical sparse covariance dictionary, iterative EM | compressed/noisy training samples, spatial prior knowledge | spatial ULA setting | not selected: new wideband-MIMO dictionary/preparation problem exceeds baseline port scope |

    ## A-MMSE version guard

    v1 and v5 are one evolving work, not two independent works. [v1](https://arxiv.org/html/2506.00452v1) uses a TDL/NR framing; v5 changes title and evaluates COST2100. Actual inspected repo learns one exported filter over its training population. Online plots are not evidence of a reproducible online adaptation path. Ridge A-MMSE's input-dependent formulation is a different work. Preserve these distinctions, dataset provenance and commit pins. Public availability without an explicit license does not authorize a code port; a request to authors requires separate messaging authorization.

    PEFT author-copy Table8 distinguishes **parameter count**, nominal payload and serialized storage. Its RTX4080 SUPER/batch32/input32×14×3072×2 measurements report LoRA54.813ms switching and7.838ms/batch inference versus full7.245ms/batch. These are author measurements, not our timing or a cross-system latency comparison. Merge avoids an extra inference path, not all switching cost. Our initial planning draft confused1.71M parameters with MB; source reinspection corrected it before final validation. No historical paper/results were edited.

    ## Fair port contract before any implementation

    Offline primary budget: the same permitted Source/P/M clean preparation trajectories, or a separately named Source-only arm. Count unique trajectories and all correlated frames; prohibit target U-Val H/labels, condition labels or new privileged target statistics. Equal online information: same noisy pilots, σ², power/mask, and the same causal history when comparing adaptation benefit. A fixed-filter model without history is an architecture comparison, not isolation of coordinate adaptation; include an explicitly budget-matched history control if feasible rather than claim equal information from unequal windows.

    Use joint spatial statistics if a model supports them; four per-link models lose joint information and multiply budgets, so disclose that port and do not call it a faithful joint equivalent. Train/validation splits must be by trajectory; the author A-MMSE frame-random split is not copied into our data. Freeze baseline hyperparameters using U-Dev only. Prove no ground-truth H enters inference. Small complex128 conditional-estimator tests and small author-setting reproduction must pass before joint integration. No author package was installed, trained or executed this round.

    If MFA needs full dense covariance/large EM that defeats the cost cap, stop the port and examine the reserve CCF selector. If A-MMSE license/data or faithful version cannot be resolved, stop that port and report the learning-baseline gap; do not call our own unrelated CNN a published baseline. This task recommends candidates, not a promise that either will be feasible.
    ''')

    md('offline_statistics_study_plan',r'''
    # Offline-statistics sensitivity plan — not executed

    Question: what preparation budget is needed for Source and the two correction directions to remain useful? This is an uncertainty/cost study of the same statistical method, not a claim that clean CSI is free or a new clean-CSI-free method.

    Historical preparation uses 500 clean trajectories ×10 correlated frames per origin, Source C30, P B30, M C120; sample ledger must identify each origin and overlap. A trajectory, not a frame or scalar coefficient, is the independent sample unit. Preserve original 500-trajectory NPZ and all old results. New reduced-budget statistics are separate **upgrade preparation artifacts**, never replacements. Keep source reference received power/SNR policy fixed; changing N must not change noise power as a hidden benefit.

    ## Primary design for later approval

    Candidate N_stat =25/50/100/250/500/1000. First cost-bounded stage recommends **50/100/250/500**, with independent replicate blocks and nested prefixes within a block. Proposed three blocks minimum for an exploratory cost curve, not a general precision guarantee; final replicate/sample count and compute cap must be preregistered after small engineering timing and before U-Val. N=25 is a stress point only if the smaller-budget curve changes a claim; N=1000 requires a benefit/cost question and separate approval. No full six-point factorial now.

    Two diagnostic arms separate N_source variation with module origins fixed at 500 from N_module variation with Source fixed at 500. Then a small common-N arm measures total acquisition budget. Means, centered Rpp/Rdp, solves, dense differences, origin-weighted factor quality, pilot predictors and inference must be tracked as a coherent artifact set for each arm. Same rank4, folds, unrestricted solve, window5, power and SNR definitions; no post hoc mean normalization, clipping, ridge or module choice. Preparation is not target adaptation: U-Val target statistics remain evaluation-only and inaccessible to deployment.

    Use common U-Val H/bits/unit-noise within comparisons and independent statistics blocks outside U-Val. Report covariance/operator change, principal correction-space change, coefficient conditioning, NMSE/BER, acquisition counts, memory and measured cost. Paired trajectory-level uncertainty across methods; outer independent preparation blocks quantify covariance-estimation uncertainty. Do not use a trajectory bootstrap of a single covariance fit to claim covariance-training replication. Data-mask energy normalization remains explicit. Failure/singular systems are recorded and trigger engineering/scientific review, never deleted.

    ## Noisy-statistics second layer (optional sensitivity only)

    A clean-channel corpus is a controlled preparation assumption. Evaluating noisy or estimated full-grid channels requires specifying acquisition/sounding cost, estimator correlation, known noise and first/second-moment bias. Subtracting a known observation-noise covariance and a PSD projection changes the statistics estimator: it must be a separately labeled sensitivity arm with a frozen formula, not a silent change to old NPZ. Sparse pilots alone do not reveal the full data–pilot cross-covariance without additional model assumptions. Do not pretend the clean/noisy arms have identical information costs.

    First assess analytically when independent noise permits covariance bias correction and when shared input/target noise biases Rdp, as illustrated by [Ridge A-MMSE](https://arxiv.org/html/2608.15945v1). A limited noisy sounding arm may test robustness of preparation. A learned/noisy online procedure without clean offline statistics is a different research problem and belongs to Paper 2.

    ## Decision value / stops

    A stable plateau within the smaller budget supports a bounded acquisition-cost statement; deterioration requires reporting the required budget and uncertainty. Noncentral mean sensitivity is separately diagnosed; do not silently refit means after seeing validation. If no curve can change the stated cost assumption, do not run it. Freeze interpretation and confidence reporting before U-Val; no arbitrary numerical stability threshold is added in SP-U1. No statistics fitting, generator call or sensitivity experiment has run.
    ''')

    md('pilot_conditioning_study_plan',r'''
    # Pilot overhead and real-coordinate conditioning — design only

    Pre-data hypothesis: reduced pilot information **may** reduce excitation/identifiability and worsen coordinate inference; the effect need not be monotonic because geometry, correlation, signal scale and validation loss change together. The study changes a claim about when pilot adaptation is useful; it is not a decorative curve.

    ## Legal geometry and arithmetic

    Nt=Nr=2, K=64,T=16. Primary dT=4 gives symbols [0,4,8,12]. FDM offsets [0,dF/2], owner pilot modulus1, other TX0. Data two-stream scaling1/√2 unchanged. The following counts are exact layout arithmetic, **not simulated performance**. In a later implementation they must be recomputed from the actual mask, never trusted from a field.

    | dF | offsets | pilot RE/TX | union RE / 1024 | union fraction | joint observed coefficients Nr×union | joint data coefficients 4×data RE |
    |---|---|---|---|---|---|---|
    | 8 | [0,4] | 32 | 64 | 1/16 | 128 | 3840 |
    | 16 | [0,8] | 16 | 32 | 1/32 | 64 | 3968 |
    | 32 | [0,16] | 8 | 16 | 1/64 | 32 | 4032 |

    dF=32 is a newly proposed **study condition**, not a silent update of the main four pilot options. Optional legal dT=2 or8 gives dF16 union fractions1/16 or1/64, but first stage uses dT4 only to avoid a full factorial. dT8 yields only two pilot symbols; geometry/excitation must be checked, not assumed fourfold-equivalent.

    `frozen_folds()` currently assumes the 64-entry dF16 layout. Future geometry-specific folds, gather order, predictor dimensions and exactly balanced train/holdout partitions require explicit small tests before integration. A proposed `(k//dF+t//dT)%4` partition must be enumerated for each geometry and checked for per-link coverage; it is not yet validated or authorized. Do not change only YAML and reuse stale operators/factors. Each geometry requires its own offline Source/module predictor preparation from the permitted corpus under a separately versioned protocol.

    ## Paired evidence design

    New U-Dev sets prepare geometry artifacts and verify masks/folds; new U-Val sets evaluate fixed rank4, window5 and dense coordinate fitting. Use identical trajectories, master bits and unit noise across layouts; only legal RE mapping and corresponding operator preparation change. Keep raw H unnormalized and the frozen source-ensemble noise reference, not per-mask/per-frame received-power calibration. All methods in a layout see the same pilots/history/noise, and total energy accounting includes owner-only pilots and union data loss.

    Report actual overhead; real Gram λ_min/λ_max/condition number; normalized column coherence; fit failure/nonfinite counts; α/β distributions; pilot loss; data NMSE/BER; common-data-intersection NMSE alongside mask-specific NMSE so a changing data mask is not misread as an estimator gain. Examine coefficient/risk relations using independent trajectories and paired confidence intervals, not hundreds of correlated scalar rows. No new rank, module or stabilizer is selected from old independent results. Predefine all aggregation and U-Val comparisons after theory/engineering checks and before U-Val access.

    ## Consequences, fixed before results

    Support: bound the adaptation claim to demonstrably excited layouts and quantify the overhead trade-off. Contradiction: remove a simple pilot-density explanation and report which geometry/correlation effects invalidate it; do not silently change folds or capping. Mixed: present conditional performance/excitation evidence, without a universal monotonic law. If a pattern cannot alter the claim, it loses priority. Gram rank proves only the realized surrogate solve, not population target-operator identification. Density alone cannot remove outside-space representation error. No layout experiment has run in SP-U1.
    ''')

    md('ood_study_gate',r'''
    # Generalization and scaling gate — no new channel generation

    Prefer **one genuinely unused CDL profile at 2×2** over 4×4 for the current statistical-mismatch question. Proposed CDL-A at 100ns,30km/h and unchanged array/orientation is a candidate, not a verified unused condition: eligibility requires a complete history/config/manifest/design audit. Prior TDL-A use is different from CDL-A, but does not automatically prove the CDL candidate is fresh. If previously used to prepare, choose hyperparameters or design, it cannot be called design-unseen OOD. Obtain an eligible condition definition before generating data; do not pick the next profile based on old or new performance.

    Source/P/M library, rank4, window5, fitting rule, σ² policy and geometry remain frozen for this challenge. New target statistics, if generated later solely to construct a Matched evaluation reference, must never reach the estimator or basis; this privilege is separately counted and labeled. Hypothesis: a new profile may lie inside or outside the two-direction correction class. Primary reporting: source-relative deployed benefit, representation and coordinate limitations, paired trajectory uncertainty; preserve every outcome. Thresholds/condition IDs/replicates/seed manifests must be approved and frozen before U-Val. No unconditional all-environment claim follows from a single new profile.

    | Option | Scientific/reviewer value | Compute | Implementation risk | Decision |
    |---|---|---|---|---|
    | New eligible CDL profile, fixed 2×2 | directly tests prepared-space transfer | additional independent corpus/replay/statistical reference; timing UNKNOWN | history eligibility, reference-only target stats | recommended conditional |
    | 4×4 at equal union overhead | tests spatial scaling and fixed-rank capacity | substantially larger operators/statistics and baseline retraining | Nt2 assumptions, joint layout, power and link tests | defer unless a scaling hypothesis merits it |
    | Full waveform high-speed ICI | asks whether diagonal-model benefit survives interference | full waveform, channel, receiver and reference work | changes observation/noise/model problem | Paper 2 |

    ## 4×4 dimensional estimate, not a run

    At equal union pilot fraction1/32, one legal candidate is Nt=Nr=4,dF32,dT4, offsets[0,8,16,24]. Union32RE, joint pilot dimension128, joint data dimension16×992=15872. Full Source entries=15872×128=2,031,616, **8×** current3968×64=253,952. Two rank4 corrections store 2×4×(15872+128)=128,000 complex coefficients versus32,256 (~3.97×). Rdp grows by8×; Rpp by4×. These are algebraic sizes, excluding full-covariance learning baseline/temporally augmented predictors.

    Every stream would need1/√4 under total power1, geometry-specific statistics/folds, and revised receive reference/antenna indexing under a new system protocol. Current hardcoded 2×2 interfaces do not become a validated 4×4 method by toggling nt/nr. A larger figure proving the program runs is insufficient value. Any scaling work needs a prespecified rank/representation question and regression against 2×2, not post hoc rank tuning.

    ## Evidence split and old-test firewall

    U-Dev: new namespace for engineering, statistics replication and baseline choice; trajectory/bits/noise/training seeds remain conceptually separate. U-Val: separately frozen independent trajectories for prespecified upgrade claims. U-Final only if material new selection/method changes make an additional final evaluation necessary; do not invent a formal test per study. Old T20B remains existing final evidence/reporting-only forever, including its recovered deterministic path. It never becomes U-Dev and never selects new rank, pilot density, baseline hyperparameters, theory parameter or module. Namespace disjointness and design-history audit are gates, not assumptions.
    ''')

    candidates=[
      ('Coordinate theory',5,5,4,4,3,1,1,'1: actual-risk analysis precedes experiment claims'),
      ('Pilot conditioning',5,5,3,5,3,3,2,'2: tests the sparse-information claim'),
      ('Offline statistics sensitivity',4,5,2,5,3,3,2,'3: identifies acquisition/replication cost'),
      ('Modern baselines',4,5,2,5,4,4,3,'4: necessary external comparators, feasibility first'),
      ('New eligible OOD profile',4,4,2,4,2,3,2,'5: direct prepared-space boundary'),
      ('Runtime benchmark',3,4,1,3,2,1,1,'6: honest measured cost, after baseline integration'),
      ('4×4 scaling',3,3,2,3,5,5,4,'defer: expensive unless new scaling hypothesis'),
      ('ICI/high mobility',4,3,3,4,5,5,5,'Paper 2: changes physical observation problem'),
      ('Clean-CSI-free extension',4,4,3,4,5,4,5,'Paper 2: changes acquisition/statistical assumption')]
    rows=['# Upgrade value/cost matrix','','Internal advisor planning estimates, not measured execution costs or acceptance probabilities. Value axes: 5 higher benefit. Cost/scope axes: 5 higher burden/risk. Do not add all seven scores as if they had the same sign. Priority reflects claim relevance and dependencies.','',
          '| Candidate | Scientific | Reviewer | Novelty potential | Evidence | Implementation cost | Compute cost | Scope creep | Priority / reason |','|---|---|---|---|---|---|---|---|---|']
    rows += ['| '+' | '.join(map(str,row))+' |' for row in candidates]
    rows += ['','No theory novelty is established by its score. Conditional code/licensing or statistical-memory blocks can change feasible scheduling, not retroactively change scientific hypotheses. Timing must be measured by separately authorized small engineering runs before a compute allocation is promised.']
    md('upgrade_value_cost_matrix','\n'.join(rows))
    dump('outputs/sp_u1_upgrade_cost_estimates.json',{'scale':'ordinal_1_to_5','measured_compute':False,
        'axes':['scientific','reviewer','novelty_potential','evidence','implementation_cost','compute_cost','scope_creep'],
        'candidates':[{'name':r[0],'scores':list(r[1:8]),'priority_reason':r[8]} for r in candidates]})

    md('paper1_paper2_boundary',r'''
    # Paper 1 / Paper 2 boundary

    Paper 1 remains sparse-pilot analytical adaptation of prepared statistical operators with full Source and compressed corrections. Useful additions: actual coordinate-risk/conditioning analysis; preparation-budget uncertainty; a small legal pilot-overhead study; at most2–3 fair external baselines; one eligible fixed-method OOD profile; measured latency/storage. All are conditional on new evidence protocols. No new method component is authorized here.

    Paper 2 candidates: online unknown-statistics recovery without clean offline corpus; new correction modules/basis learning; environment-label inference; neural coefficient predictors/attention/GAN/RL; payload semi-blind adaptation; full waveform ICI/mobility receiver; continual environment switching; automatically growing library; extensive 4×4 scaling as a new capacity question. These change information assumptions or the research problem. Their possible value does not justify including them now.

    Two directions are a prepared hypothesis, not a physical-factor decomposition. Keep delay/outside-space failures and mean effects visible. Do not add a third module to repair a tested failure. Representation analyses use evaluation-only target information; deployed coordinates remain pilot-only. Author learning architectures may be reproduced as named comparators after approval, never silently become our estimator. The whole adaptive map is nonlinear even though the frozen filters and conditional prediction are linear. No manuscript changes this round.
    ''')
    md('title_positioning_review',r'''
    # Title and scientific positioning review

    Current broad project title contains “learning-enhanced”. The frozen final deployed method is analytical real-LS adaptation of empirical statistical filters; it is not a trained neural estimator. Empirical preparation is statistical learning in a broad sense, but a neural reading of that phrase would misdescribe the core. No project or manuscript title is renamed in SP-U1.

    Recommended title direction: **Sparse-Pilot Adaptation of Frozen LMMSE Operators with Low-Rank Affine Corrections**. Alternative if actual-risk analysis becomes central: **Pilot-Driven Affine Adaptation of Mismatched LMMSE Channel Estimators: Conditioning and Correction Compression**. Both describe the object rather than assert a priority or all-condition performance.

    Primary venue positioning: statistical signal processing, adaptive linear channel estimation and signal processing for communications. “Linear” refers to the prepared Wiener operators, not the overall input-dependent estimator. Avoid interpreting profile/mobility directions as separable propagation physics or rank4 as the rank of the full estimator. “Low pilot overhead” needs the proposed overhead study and precise1/32/current-geometry qualification; no arbitrary-pilot claim. Title choice should follow advisor approval and the eventual evidence, not old formal-test numbers.
    ''')
    md('projected_contributions',r'''
    # Projected contributions — conditional, not yet achieved

    ## Present defensible claim level

    A bounded engineering/statistical study integrates a full frozen Source Wiener operator, two prepared finite-difference corrections, sparse-pilot real affine fitting and correction-only compression under an explicit offline/online information contract. Affine combinations, LS mixing, reduced-rank filtering and libraries are established individually. Within the literature examined, no direct match was identified in the verified set for the entire contract; potential overlap remains, particularly the abstract-only Özdemir algorithm. This is not a priority clearance.

    ## If theory succeeds

    Contribution candidate1: adaptation-specific identifiability/conditioning and data-risk analysis for the actual endogenous cross-pilot design, including a risk-alignment condition or impossibility boundary and coordinate-amplified compression/mean error. Basic LS covariance alone remains explanatory. Exact vector identities and conditional bounds must not be sold as independent scalar-MSE additivity or a broad new theorem.

    ## If new evidence supports it

    Candidate2: a bounded pilot-information/conditioning and preparation-cost characterization with independent preparation replication and explicitly counted clean sounding assumptions. Candidate3: equal-information comparison with feasible modern statistical/learned-linear controls and fixed-method transfer to a truly unused profile. These are empirical contributions with tested limits, not architecture inventions.

    If evidence is mixed, report the operating restrictions and failures. If pilot surrogate does not track data risk, narrow the inference claim. If smaller preparation budgets fail, report acquisition requirements. If stronger controls remove gains, do not add tuning or modules to reclaim them. Existing T20B remains independently interpretable original final evidence; upgrade claims come from new approved data. Never infer a broad physical mechanism from coordinate geometry or high speed without ICI.
    ''')
    md('recommended_upgrade_plan',r'''
    # Upgrade plan freeze for advisor review

    State: **PROPOSAL_FROZEN_PENDING_ADVISOR_REVIEW**. Planning completion is not authorization to implement or run SP-U2. Original method/protocol/results/manuscript byte-unchanged. Submission readiness and acceptance likelihood are not declared.

    ## Foundation and evidence contract

    Working research question: can a compact prepared operator space be inferred from sparse received pilots when Source Wiener statistics mismatch? Working hypotheses, before new results: realized real-design excitation is necessary for stable coordinates; pilot surrogate and data risk agree only under additional structure; adaptive coordinates amplify compression residuals; preparation-budget uncertainty matters; fixed two-direction transfer has an outside-space boundary. These are testable propositions, not known outcomes.

    Canonical objects, mean convention and actual solve are fixed in theory_feasibility. Existing tests and all tracked files are provenance-checked. Root post-review forensic audit is separate from three immutable reviewers. Advisor claim ceilings: finite tested conditions; analytic filters with nonlinear coordinate fitting; clean offline statistical preparation counted; diagonal channel excludes ICI; no priority assertion. Do not replace scientific claims with internal task IDs in eventual prose.

    ## Packages / ordered dependencies

    | Package | Contents | Conditional value | Planning cost |
    |---|---|---|---|
    | Minimum | closest-work closure; actual-algorithm theory; statistics ledger; one cost-bounded pilot/conditioning design; baseline feasibility gate | sharper but still limited external evidence; not promised journal-ready | about3–4 researcher-weeks, compute allocation unknown |
    | **Recommended** | minimum + offline-statistics replication/sensitivity + two feasible modern controls + one eligible new-profile challenge + measured runtime/storage | directly addresses most independent reviewer concerns | about6–9 researcher-weeks; small timing gates needed before compute budget |
    | Aggressive within Paper1 | recommended + additional preparation replication/noisy-statistics sensitivity and tightly motivated4×4 scaling only if theory asks a scaling question | wider cost/capacity evidence, diminishing scope value | about10–14 researcher-weeks; high memory/retraining risk |

    Estimates are work-planning ranges, not measured wall-clock durations. Recommended distribution: theory/closure ~1–2weeks; baseline gates/ports ~2–3; preparation+pilot protocols/evidence ~1–2; profile/runtime+integration ~1–2, with overlap possible. Baseline infeasibility can extend or halt this schedule. No deadline or GPU-hour promise from old task timings.

    Dependency order: (1) actual coordinate-risk proof gate; (2) modern-control small reproduction/license/information gates and new namespace audit; (3) preregister preparation/pilot/U-Val designs; (4) approved U-Dev, then frozen U-Val, with trajectory-level uncertainty; (5) eligible OOD/runtime and claim reconciliation; (6) manuscript revision only after advisor-approved evidence. U-Final is conditional on substantive new selection; old T20B is always reporting-only.

    ## Stops

    Stop any scientific execution in this round. Later: stop on provenance drift, seed/data overlap, true-H/label leakage, unauthorized source/metric changes, non-reproducible baseline, undecidable license, invalid folds/power/noise, actual solver/test failure, or unapproved target-statistics access. Preserve every failed outcome; no seed replacement, clipping or post hoc threshold. Stop a contribution claim when closest-work distinction stays unresolved or theory only restates generic LS. Stop an experiment with no possible claim-changing decision. Never repair outside-space failure by introducing a new module under this plan. Full ICI or clean-CSI-free development requires a separate Paper2 research decision.

    ## Exact next task recommendation (not started)

    **SP-U2A: Actual-Algorithm Coordinate-Risk Theory Gate.** Read affine-combination theory; formally derive real empirical identifiability, conditional biased coefficient moments, surrogate/data-risk relation and counterexample, exact vector/mean/cross-term decomposition, and adaptive correction-compression bounds for unchanged `pilot_cv.py`. Deliver proof notes and a prior-art classification; only after that task is explicitly approved, small complex128 synthetic identity/counterexample tests may be designed and executed. No CDL generation, no target-statistics fitting, no new modules, no baseline training, no formal-test reuse and no manuscript theorem insertion in that task without further authorization. Exit with PROVED / CONDITIONAL / NOT_ESTABLISHED by statement, not a forced overall scientific success.
    ''')
    md('advisor_review_packet',r'''
    # SP-U1 advisor review packet

    ## A. Signal Processing fit

    Statistical signal processing/adaptive linear CE fits the official estimation/filtering/communications scope. A useful actual-risk analysis and cost-controlled evidence are the priority. Shop rules verified; full author guide remains403/UNKNOWN. Current manuscript is pre_signal_processing_upgrade_draft; no readiness/acceptance claim.

    ## B. Ten most important closest works

    1. Özdemir2006, [adaptive low-rank MIMO-OFDM LMMSE](https://doi.org/10.1109/TWC.2006.04713): tracking/subspaces; ABSTRACT_ONLY, equations UNKNOWN.
    2. Azpicueta-Ruiz2010, [LS affine multi-filter adaptation](https://doi.org/10.1109/ISCAS.2010.5538026): difference-output/windowed LS is established.
    3. Edfors1996, [SVD OFDM LMMSE report](https://www.diva-portal.org/smash/get/diva2:997763/FULLTEXT01.pdf): full-filter reduced rank, verified URN exception.
    4. Mei2021, [CCF selection](https://doi.org/10.1049/cmu2.12250): pilot-interpolation statistical selection.
    5. Gui2014, [affine sparse-filter CE](https://arxiv.org/pdf/1407.6083v1): adaptive CE combination precedent.
    6. A-MMSE2025/2026, [v5 learned linear filter](https://arxiv.org/html/2506.00452v5): fixed exported full filter/rank extension; author code license unresolved.
    7. Fesl2023, [MFA conditional mixture](https://arxiv.org/pdf/2304.14809v2): compact learned prior and posterior convex LMMSE.
    8. Channelformer2023, [receiver training](https://arxiv.org/html/2302.04368v1): special label pilots/boost or denoising, not same overhead.
    9. Lee/Choi2026, [two-tier PEFT](https://doi.org/10.1109/OJCOMS.2026.3697928): supervised target adapters and SNR switching, different information contract.
    10. Turan2024, [GMM adaptive pilot design](https://arxiv.org/pdf/2403.17577v1): statistical mixtures and adaptive observations, different fixed-layout question.

    ## C. Top five novelty collision risks

    Affine/windowed LS; low-rank adaptive tracking; generic full-filter SVD; posterior filter mixtures; compact environment libraries. Concept novelty LOW for the isolated elements. Exact whole-contract overlap remains unresolved; do not assert priority. Candidate contribution is integrated adaptation-specific analysis plus bounded new evidence.

    ## D. Top ten reviewer attacks

    What distinguishes the closest prior algorithms? Does pilot loss minimize data risk? Can endogenous/fold-overlap noise bias coordinates? What real excitation guarantees identifiability? What clean preparation cost is required? Where is a fair modern comparator? Do adaptive coordinates amplify rank error? Does mean bias dominate? Is any profile truly design-unseen? Does five-frame history, rather than adaptation, account for a comparison? Three frozen reports contain44 concerns; root mandatory coverage and audit are separately labeled.

    ## E. Theory feasibility verdict

    Feasible: real Gram criterion, conditional biased LS moments, surrogate/data-risk conditions or counterexample, exact vector decomposition with mean/cross terms, and conditional coordinate/compression bounds. A new general theorem is NOT_ESTABLISHED. Unconditional iid coefficient covariance and scalar additive MSE are not valid defaults. Next proof gate before evidence expansion.

    ## F. Modern baseline recommendations

    Conditional shortlist: **MFA** statistical posterior mixture + **A-MMSE v5** learned linear filter; reserve MeiCCF only if needed. MFA is partial BSD-3 code and needs joint masked-CE validation; A-MMSE lacks verified license/dataset/runtime reproduction. Channelformer extra labels/RE/power and PEFT target supervision prevent equal-budget primary use. No implementation GO yet.

    ## G. Offline-statistics study verdict

    Valuable: Source versus module sample quality and independent covariance-training uncertainty. Begin proposed50/100/250/500 nested trajectories within independent blocks; separate acquisition cost and correlated frames.25/1000 conditional. Noisy full-sounding statistics only a sensitivity arm, not a new clean-CSI-free method. Freeze source ensemble noise reference.

    ## H. Pilot-conditioning study verdict

    Valuable: legal dF8/16/32,dT4 gives union fractions1/16,1/32,1/64. Need geometry-specific fold/operator preparation, actual mask accounting, real Gram/coordinate diagnostics and paired data-risk comparison. Hypothesis allows nonmonotonic effects. Outcomes must change or restrict the low-pilot claim. No runs.

    ## I. OOD study verdict

    Prefer one genuinely unused2×2 CDL profile; CDL-A is only a candidate pending complete design-history audit. New U-Dev/U-Val, never old T20B for selection.4×4 increases Source entries8× at equal overhead and requires power/layout/link/statistics/baseline work; defer without a scaling question. ICI is Paper2.

    ## J. What NOT to add to Paper1

    New module, neural coordinates, adaptive basis, clean-CSI-free online covariance recovery, payload adaptation, full ICI receiver, continual environment switching or a large spatial-scaling project. Preserve finite-space failures and raw channel/SNR definitions. No manuscript/BibTeX edits now.

    ## K. Recommended upgrade package

    Actual coordinate-risk theory + modern-control gates + preparation and pilot studies + one eligible profile + measured runtime. Dependencies and stops in recommended_upgrade_plan. Freeze for advisor review; execution requires the next explicit task. Formal original results remain reporting-only.

    ## L. Estimated research cost

    Minimum3–4 researcher-weeks; recommended6–9; aggressive10–14. Planning estimates only, no measured GPU-hour claim. Early code-license/data/memory/timing gates can block a candidate; do not compensate by adding unrelated architectures. Largest burden is faithful joint baselines, not documentation.

    ## M. Exact next task recommendation

    SP-U2A Actual-Algorithm Coordinate-Risk Theory Gate: proof notes for unchanged real pilot-CV solve, conditional moments/risk relation, mean/cross-term identity and adaptive compression; scoped small synthetic tests only after explicit task authorization. No CDL/channel fitting/training/formal-test access or manuscript modification. SP-U1 stops here.
    ''')
    dump('outputs/sp_u1_upgrade_gate.json',{
        'task':'SP-U1','state':'PROPOSAL_FROZEN_PENDING_ADVISOR_REVIEW',
        'manuscript_state':'pre_signal_processing_upgrade_draft',
        'old_t20b_reporting_only':True,'old_t20b_used_for_new_design':False,
        'scientific_execution_authorized':False,'new_experiments_run':False,
        'source_method_changed':False,'manuscript_changed':False,
        'theory_verdict':'ADAPTATION_SPECIFIC_ANALYSIS_FEASIBLE_NOT_PROVED',
        'recommended_baselines':['mfa','ammse'],'reserve_baseline':'mei2021',
        'baseline_implementation_go':False,'new_ood_eligibility':'CONDITIONAL_HISTORY_AUDIT',
        'package':'recommended','next_task_recommendation':'SP-U2A Actual-Algorithm Coordinate-Risk Theory Gate',
        'next_task_started':False,'all_checks_passed':False,'validation_state':'PENDING_ACTUAL_TESTS'})
