"""SP-U9A editorial consolidation of frozen evidence only (stdlib).

No channel simulator, estimator, random generator, bootstrap, scientific fitting,
manuscript editing or figure generation is called by this task.
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/sp_u9a'

def load(path): return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,value):
 p=ROOT/path;p.parent.mkdir(exist_ok=True,parents=True)
 p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write_text(path,value):
 p=ROOT/path;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(value.rstrip()+'\n',encoding='utf-8')
def write_csv(path,rows):
 with (ROOT/path).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def md_rows(rows):
 return '\n\n'.join('### '+str(row.get('Claim ID',row.get('Risk ID',row.get('ID',row.get('task ID','Record')))))+'\n\n'+
  '\n'.join('- **'+k+'**: '+(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else str(v)) for k,v in row.items()) for row in rows)

FIELDS=['Claim ID','Candidate claim','Allowed wording','Forbidden wording','Evidence level','Evidence source','Condition','Metric','Primary/secondary','Independent unit','Sample size','Statistical support','Known counter-evidence','Boundary/limitation','Reviewer attack','Needed manuscript location','Figure/table support','Status']
def claim(*values):
 assert len(values)==len(FIELDS)
 return dict(zip(FIELDS,values))

CLAIMS=[
 claim('C1','Final LR4 improves over Source under prepared mismatch',
 'Under the preregistered Profile, Mobility and Composed conditions, LR4 improves pooled data-channel NMSE over Source.',
 'General mismatch robustness; gains for every realization; superiority to all estimators',
 'A FORMAL','outputs/t20b_formal_results.json#bootstrap_results; outputs/t20b_formal_bootstrap.csv',
 '2x2; B30/100, C120/100, B120/100; source C30/100; dF16/dT4; SNR10/20',
 'Source NMSE dB minus LR4 NMSE dB','Primary endpoint A, six cells',
 'trajectory','1000/condition; 10 frames, 4–9 scored; 5000 total across five conditions',
 'Existing paired trajectory percentile bootstrap, 10000 draws; one-sided Bonferroni across ten T20B cells, lower quantile .005. Adjusted LCBs all positive.',
 'ID loss; Boundary and unseen matched gaps; no modern learned performance comparator',
 'Conditional on fixed offline library, tested geometry and source-power reference; prepared distribution is not unseen profile',
 'R4,R5,R16,R17','Abstract formal slot; Introduction evidence; Results main','F2,F3,T3','SUPPORTED_WITH_LIMITS'),
 claim('C2','Unrestricted affine fit helps Composed B120 beyond Source/Convex',
 'In held-out B120/100 composition, LR4 improves over Source and Convex under the preregistered comparisons.',
 'Affine extrapolation universally better; physical covariance lies outside a hull; any arbitrary composition',
 'A FORMAL','outputs/t20b_formal_bootstrap.csv; outputs/t20b_formal_results.json#bootstrap_results',
 'B120/100; SNR10/20; identical received inputs and source means',
 'Source−LR4 and Convex−LR4 pooled NMSE dB','Primary A/B',
 'trajectory','1000, paired methods/SNR; six scored frames',
 'Source gains 9.009532/11.773631 dB (LCB8.839314/11.652903); Convex gains3.206497/4.835934 dB (LCB3.106160/4.769255). Frozen family control, no new tests.',
 'Matched advantage remains about2.002/6.147 dB; ID degradation; failed physical separability explanation',
 'Establishes a design contrast inside one library/objective, not a physical mechanism',
 'R1,R2,R6','Method constraint; Results Composed; Discussion','F3,T3','SUPPORTED_WITH_LIMITS'),
 claim('C3','Rank4 stays near Dense across the tested main conditions',
 'The preregistered Composed compression penalties meet the 0.5-dB criterion; compression behavior is condition dependent.',
 'Rank4 universally lossless; channel rank equals4; weighted singular tail guarantees adaptive target MSE',
 'A FORMAL + A SEALED','outputs/t20b_formal_results.json; outputs/sp_u7b_r1/analysis.json#results.secondary',
 'T20B all five conditions and sealed CDL-A120/100; SNR10/20',
 'LR4−Dense pooled NMSE dB','Primary C only B120; ID/Profile/Mobility/Boundary and OOD differences descriptive',
 'trajectory','1000/condition; fixed shared coefficients Dense/LR4',
 'B120 penalty .225399/.405946; adjusted UCB .243135/.436599 <.5. Boundary20 point1.086044; CDL-A differences+.125492/−.620492 with frozen95% CIs [.110084,.141100]/[−.648244,−.592253].',
 'Boundary20 exceeds margin; CDL-A10 nonzero compression loss and20 improvement; full target gap not isolated compression',
 'Margin is preregistered only for Composed. Other cells cannot retroactively become noninferiority endpoints.',
 'R7,R9,R11','Method compression; Results main and limits','F3,F6,T3,T4','MIXED'),
 claim('C4','ID has modest central loss and realization tails',
 'On source-domain inputs LR4 has a modest pooled NMSE loss; development realizations also show nontrivial tails.',
 'No source-domain degradation; negligible worst case; proven safety on all ID realizations',
 'A FORMAL + B DEVELOPMENT','outputs/t20b_formal_results.json#nmse_results; outputs/sp_u4b_analysis.json; outputs/sp_u5b_ID_tradeoff.csv',
 'C30/100 ID; fixed formal density and development density/budgets',
 'Formal pooled loss; separately development median/P95/maximum paired trajectory dB loss','Descriptive/secondary',
 'trajectory','Formal1000; development100 reused/condition; five subset preparations are not500 independent tests',
 'Formal pooled loss .218084/.122477 dB; U4B method median range .084931–.209646, maximum1.147027; U5B dF32 common-domain median .245867/.194221, maxima1.795421/1.632217. No new formal inference.',
 'ID lower-budget and lower-density tails; negative loss/gain realizations retained',
 'Formal pooled central and development distribution metrics are distinct estimands/populations; never merge.',
 'R8,R16','Results ID; main limitations; appendix tails','T3,F5; appendix U4B-E/U5B tails','SUPPORTED_WITH_LIMITS'),
 claim('C5','Adaptation survives lower pilot overhead with tradeoffs',
 'Development tests suggest retained Composed adaptation benefit at 1/64 pilot occupancy, alongside worse absolute NMSE, extra LR4 sensitivity and ID loss.',
 'Robust to pilot scarcity without qualification; pilot density alone causally explains loss; link throughput improvement',
 'B DEVELOPMENT','outputs/sp_u5a_analysis.json; outputs/sp_u5a_table1_geometry.csv; outputs/sp_u5b_analysis.json; outputs/sp_u5b_control.csv; outputs/sp_u5b_extra_sensitivity.csv; outputs/sp_u5b_gains.csv',
 'dF8/16/32, dT4, occupancies1/16,1/32,1/64; ID/Composed; SNR10/20; U5B common dF8 data domain',
 'Paired trajectory NMSE penalties/gains; common-domain score control','Development descriptive; BER secondary',
 'trajectory','100/condition shared across densities/SNR; shared offline500/condition',
 'U5A low-overhead Composed minimum individual gain3.359111 dB; rank failures/fallback0. U5B both SNRs all Composed extra penalties positive; common absolute reduction penalties LR4 med2.212372/1.710697 versus Source.349353/.614199.',
 'Absolute NMSE worsening, LR4 extra penalty, broader ID tails; folds/operators and geometry co-vary',
 'U5B changes score mask only, reuses coordinates/predictions. It removes scoring-domain ambiguity, not every geometrical confound. No formal validation or threshold.',
 'R10,R8,R16','Development sensitivity subsection, not abstract general robustness','F4; appendix control/tails','SUPPORTED_WITH_LIMITS'),
 claim('C6','Offline budget/subset evidence supports benefit with finite preparation risk',
 'Development subset realizations retain Composed gains under the tested budgets, with preparation variation and low-budget tail risk.',
 '50 trajectories are sufficient; sample efficient in general; 500 independent subset replications; guaranteed monotone central risk',
 'B DEVELOPMENT','outputs/sp_u4a_analysis.json; outputs/sp_u4b_analysis.json; outputs/sp_u4b_summary.csv; outputs/sp_u4b_table6_hypotheses.csv',
 'Budgets50/100/250/500 per preparation environment; five nested subset realizations on one finite corpus; ID/Composed SNR10/20',
 'Median trajectory NMSE/gain; across-preparation IQR and signed paired P95 tails','Development descriptive',
 'evaluation trajectory, with subset realization a separate preparation axis','100/condition reused;5 subset realizations, shared N500 reference; not N500 independent evaluation',
 'All30 Composed gain medians positive; minimum individual gain3.976139. Predominantly increasing low-budget LR4 tails19/20 series, one counterexample; maximum Composed budget penalty .887053 dB.',
 'Central ordering mixed; subset overlap, low-budget variability and tails; Source/corrections/coefficient geometry co-vary',
 'No population sample-complexity theorem, no independent external validation, no new CI/p-value.',
 'R4,R5,R16','Development offline-budget subsection; limitations','F5; appendix U4B tables','SUPPORTED_WITH_LIMITS'),
 claim('C7','Sealed independent unseen-profile transfer at one target setting',
 'Independent sealed validation supports Source-relative NMSE improvement at CDL-A/120/100 under the frozen protocol.',
 'Universal OOD; arbitrary channel robustness; near-Matched unseen estimation; validation reused for tuning',
 'A SEALED','outputs/sp_u7b_r1/analysis.json; outputs/sp_u7b_r1/table_A.csv; outputs/sp_u7b_r1/table_B.csv; outputs/sp_u7b_r1/final_verification.json',
 'CDL-A120km/h/100ns; 2x2; dF16/dT4; SNR10/20; seven frozen methods; rank4',
 'Source−LR4 pooled energy NMSE dB','Preregistered two-cell primary; remaining comparisons secondary',
 'trajectory','1000 evaluation independent trajectories; separate500 Matched-reference trajectories; ten frames score4–9',
 'Gains10.969857/15.722409 dB; one-sided2-cell Bonferroni LCB10.793108/15.591996 positive; saved10000 paired draws;15/15 integrity gates PASS. Conditional on fixed reference.',
 'Single profile/speed/delay combination; large Matched gap, mixed compression, BER ranking',
 'USED validation permanently reporting-only; old invalidated U7B excluded; target condition label never deployed input.',
 'R11,R12,R17,R18','Independent validation subsection; abstract OOD slot','F6,T4','SUPPORTED_WITH_LIMITS'),
 claim('C8','OOD matched gap shows partial transfer',
 'Despite Source-relative gains, LR4 remains2.85/5.54 dB above the finite Matched reference at the tested unseen setting.',
 'Near-Matched; ideal Bayesian oracle gap; all remaining gap is representation error',
 'A SEALED','outputs/sp_u7b_r1/analysis.json#results.secondary.LR4_minus_Matched; outputs/sp_u7b_r1/table_B.csv',
 'CDL-A120/100 SNR10/20','LR4−Matched pooled NMSE dB','Secondary descriptive',
 'evaluation trajectory conditional on fixed Matched statistics','1000 evaluation,500 reference trajectories',
 'Points2.846553/5.543975; frozen paired95% CI[2.772704,2.920958]/[5.411672,5.674829]. Reference-training uncertainty not resampled.',
 'Large advantage over Source coexists with gap to Matched; mean/statistical estimation/compression/surrogate differences coexist',
 'Matched is finite-statistics affine LMMSE, not a true population or all-window oracle.',
 'R11,R18','OOD results and main limitations','T4,F6','SUPPORTED_WITH_LIMITS'),
 claim('C9','Secondary frozen-ZF BER benefit, metric-dependent ordering',
 'LR4 improves BER relative to Source with the frozen ZF receiver; post-hoc diagnostics show metric-dependent rankings and descriptive associations with inverse noise enhancement.',
 'General receiver-level BER superiority; NMSE ranking determines BER; Perfect CSI ZF universally minimizes BER; diagnostic correlation proves cause',
 'A SEALED SECONDARY + D POST-HOC','outputs/sp_u7b_r1/analysis.json#results.ber; outputs/sp_u8a/analysis.json',
 'Same CDL-A target, GrayQPSK, uncoded per-RE square ZF; SNR10/20; no ICI',
 'Pooled BER; paired BER difference; singular-value/inverse diagnostics','BER secondary; U8 strictly post-hoc, primary NMSE impact NONE',
 'trajectory','1000;6 scoredframes;23808bits/trajectory/SNR;23808000 pooled bits are not independent N',
 'Source BER .383318/.221926; LR4 .376263/.200900; frozen LR4−Source95%CIs entirely negative. Perfect post-hoc BER .369093/.185657; LS .362667/.221648; Matched .389983/.198533. No new bootstrap.',
 'LS beats Perfect at10dB; Matched has better NMSE but worse BER at10dB; unregularized ZF conditioning; correlations descriptive',
 'Receiver identity retains cross term; finite realization/conditioning is not an estimator-level BER theorem.',
 'R13,R14','Short receiver limitations note; appendix diagnosis','Appendix formal BER and U8 figures','SUPPORTED_WITH_LIMITS'),
 claim('C10','Rank4 reduces arithmetic and correction storage versus Dense',
 'Under the specified count convention LR4 has a lower complex-MAC arithmetic proxy and correction storage than Dense Affine.',
 'Measured faster; lower energy/latency; hardware efficiency; cheaper than Source total inference',
 'C ENGINEERING / ANALYTICAL COUNT','outputs/t20b_formal_results.json#complexity; paper/tables/05_complexity.tex; paper/sections/04_proposed_method.tex#Arithmetic-Complexity',
 '3968x64 data map; two rank4 corrections; full Source; dense cross-pilot fit retained',
 'complex MAC/window-frame; complex coefficients/bytes','Analytical implementation proxy; no statistical endpoint',
 'specified operator application','N not applicable',
 'Source253952; fitting46080; corrections32256; LR4 total332288 vs Dense807936,58.87199% proxy reduction; total/Source1.308468; correction factors32256 vs507904 complex coefficients.',
 'Total LR4 costs more than Source; scalar products/Gram/solve/detection/memory/device transfers omitted; no comparable runtime/energy measurement',
 'Correction storage only, not complete deployed footprint; SNR/library multiplicity and offline acquisition not counted by these two-factor payloads.',
 'R15','Method cost; Results complexity; abstract cost slot','T5','SUPPORTED_WITH_LIMITS'),
 claim('C11','Modern prior-art and reproduction feasibility are distinct from performance',
 'Koller2022 GMM/Fesl2023 MFA are closest statistical works; MFA and A-MMSE project-shape performance reproduction remains unestablished after frozen feasibility gates.',
 'Our method outperforms MFA/A-MMSE; gate failure proves their algorithms inaccurate; faithful A-MMSE reproduction completed',
 'C ENGINEERING + FROZEN PRIOR ART','docs/sp_u2a_prior_art_amendment.md; outputs/sp_u3b_r1_decision.json; outputs/sp_u3b_r1_A_gate.json; outputs/sp_u6a_decision.json; outputs/sp_u6a_resource_gate.json; outputs/sp_u6a_ammse_version_ledger.json',
 'MFA full project-dimensional training; A-MMSE v5/public commitfa2647f7fd0411bdb6a33ad9705cadb134ab3793',
 'Training/resource/semantic compatibility, not target performance','No comparative scientific endpoint',
 'engineering fit or static parameter bound','MFA medium fit PASS; full A stopped at3600.26s time cap; B notrun. A-MMSE optimizer/network project step NOT_RUN.',
 'MFA semantics-preserving memory patch mediumPASS/fullgateFAIL. A-MMSE source-derived project decoderfloat32 lower bound7168.025GiB vs8GiBcap, not measured allocation. Public repo global-token training differs from v5 pilot-fed generator; fixed export aligns intent only.',
 'No learned/generative performance comparison; novelty not cleared by field distinctions; Özdemir ABSTRACT_ONLY/OPEN RISK; license open',
 'Frozen current implementation examined, not all possible faithful implementations; resource failure not general baseline infeasibility.',
 'R1,R2,R3','Related Work closest comparison; limitations; appendix feasibility','T2; appendix gates','SUPPORTED_WITH_LIMITS'),
 claim('C12','Exact algorithm theory is useful but failed mechanisms must be removed',
 'Exact noisy-design fitting equations, distinct pilot/data coordinate oracles, identifiability and propagation identities clarify the frozen estimator; diagnostics limit mechanism interpretation.',
 'Conditioning causes degradation; coordinate noise necessarily worsens channel MSE; current frame dominates; simple recency; physically independent profile/mobility factors; homoscedastic OLS covariance by default',
 'B MATHEMATICAL / DEVELOPMENT + E NEGATIVE','docs/sp_u2a_equation_inventory.md#E05-E11; docs/sp_u2a_coordinate_theory.md#O01-O04,T09-T16,D01-D05; docs/sp_u2a_assumption_audit.md; outputs/sp_u2b_table3_hypothesis_audit.csv; outputs/sp_u2c_table3_hypothesis_audit.csv; outputs/sp_u2d_table4_hypotheses.csv',
 'Fourfold48-to16 complex maps, five causal current-inclusive frames,64 pilot coefficients; G640x2 real',
 'Residual identity; rank/perturbation; current-pilot output quadratic and error cross terms','Exact mathematical identities plus separate descriptive falsification',
 'identity domain; development trajectory for associations','U2B100;U2C/U2D20 fixed trajectories, six overlapping windows and20 noise repeats; not2400 independent channels',
 'z=Gc*+epsilon(c*) exact residual identity; G depends on same noisy observations. RankR(G)=2 needed. Delta h=X4(u_current)Delta c; energy=Delta c^T Re(X4^H X4)Delta c. Full residual energy includes baseline/error cross terms.',
 'U2B/U2C weak condition-vs-signed-loss links; U2D two-SNR current advantage/recency fails; no physical separability theorem',
 'Bounds explain excitation sensitivity, not outcome causation/calibrated uncertainty; empirical pilot minimizer differs from full-channel MSE oracle.',
 'R1,R6,R7,R13','Method compact exact equation; appendix assumptions; discussion one bounded paragraph','F1; appendix U2 inventories','SUPPORTED_WITH_LIMITS'),
]
CLAIMS += [
 claim('C11-P','LR4 outperforms MFA/A-MMSE','No performance superiority statement is supported.',
 'Our method outperforms MFA/A-MMSE','C ENGINEERING, NO PERFORMANCE EVIDENCE',
 'outputs/sp_u3b_r1_decision.json; outputs/sp_u6a_decision.json','Project-shape external comparators',
 'comparative target NMSE/BER','No endpoint executed','N not applicable','NOT_RUN / no valid comparison N',
 'No trained/evaluated comparable external performance output.',
 'Feasibility gates and source-recipe gaps are not performance outcomes.',
 'Uncompleted rather than scientifically defeated baselines.','R3','Limitations, not performance table','TA1','NOT_TESTED'),
 claim('C12-N','Condition/noise/current/recency/physical-factor causal mechanism story',
 'Strong versions of these mechanism hypotheses were not supported by the frozen diagnostics.',
 'Conditioning causes degradation; coordinate noise causes degradation; current frame dominates; simple recency; physical factor separability',
 'E NEGATIVE / B DEVELOPMENT',
 'outputs/sp_u2b_table3_hypothesis_audit.csv; outputs/sp_u2c_table3_hypothesis_audit.csv; outputs/sp_u2d_table4_hypotheses.csv; docs/sp_u2a_assumption_audit.md',
 'Tested Composed diagnostic conditions/SNR10/20','signed error and cross terms/age contrasts','Descriptive falsification',
 'trajectory','U2B100; U2C/U2D20; repeats/windows nested',
 'U2B H4 signed-loss relation weak; U2C conditional penalty often negative; U2D current advantage not stable across both SNRs and recency monotonic counts tiny.',
 'Excitation sensitivity and exact propagation identities still stand; negative evidence does not establish universal absence of any mechanism.',
 'Reject the requested strong narrative only; no theorem of general impossibility.',
 'R6,R13','One main limitations sentence; appendix','FA3','REJECTED'),
]

def risk(rid,severity,q,why,evidence,weakness,answer,newexp,response,wording,action,claims):
 return {'Risk ID':rid,'Severity':severity,'Reviewer question':q,'Why reasonable':why,
 'Current evidence':evidence,'Current weakness':weakness,'Can answer now?':answer,
 'Requires new experiment?':newexp,'Recommended manuscript response':response,
 'Claim wording adjustment':wording,'Action':action,'Claim IDs':claims}

RISKS=[
 risk('R1','HIGH','What is novel relative to GMM/MFA posterior-weighted conditional LMMSE?',
 'Pilot-dependent mixture of affine conditional estimators is explicit closest prior art.',
 'U2A prior-art table distinguishes posterior simplex vs unrestricted cross-pilot coordinates, source centering and correction-only rank4.',
 'Field differences do not establish sufficiently original scientific advance; exact collision remains open.',
 'Partially: algorithm contrast yes, novelty sufficiency no','Not executed; advisor may request targeted comparison/ablation evidence; literature-level positioning may suffice.',
 'Promote Koller2022/Fesl2023 to closest group; use verified field comparison; ask mentor to judge advance before U9B.',
 'A specific pilot-driven correction implementation; never first observation-dependent mixing.',
 'PAPER_RISK_OPEN','C2,C11,C12'),
 risk('R2','HIGH','How does this differ from adaptive MMSE, candidate selection and adaptive low-rank LMMSE?',
 'LS affine-filter adaptation and low-rank adaptive CE have precedents.',
 'U1/U2A frozen literature; current Hard/Convex/Dense internal comparisons; exact source-correction fit.',
 'Özdemir2006 remains ABSTRACT_ONLY / OPEN RISK; cannot exclude algorithm-level overlap.',
 'Partially; known distinctions yes, full priority clearance no','No automatic experiment; full-text/algorithm confirmation or mentor-approved narrowing needed first.',
 'Disclose abstract-only verification boundary; limit novelty to tested pipeline rather than inventing full-text differences.',
 'No first/SOTA or blanket departure from all adaptive LMMSE.',
 'PAPER_RISK_OPEN','C1,C2,C11'),
 risk('R3','HIGH','Why are modern learned/generative methods absent from performance comparison?',
 'Current comparisons mainly contrast a single prepared linear library, not strongest external methods.',
 'U3B-R1 full fit time gateFAIL; U6A static faithful-shape resourceSTOP-D; no target scores.',
 'Engineering gates cannot establish competitive performance; omission is material for submission positioning.',
 'Can explain noncompletion, cannot establish comparative competitiveness','Possible additional evidence only if mentor authorizes a later task; no branch reopened here.',
 'Report attempted contracts and stop reasons without fabricating performance; mentor decides whether a bounded method paper is adequate.',
 'No outperforms MFA/A-MMSE or state-of-the-art claim.',
 'PAPER_RISK_OPEN','C11'),
 risk('R4','HIGH','Is the clean offline CSI requirement operationally attainable?',
 'Dense empirical preparation is a substantial information/acquisition assumption.',
 'Method/system info contract; U4 budget sensitivity tests clean simulated preparation.',
 'No noisy-calibration/library-acquisition validation.',
 'Yes for assumption and tested contract; practical acquisition not established','Not needed for bounded simulated prepared-library claim; required only to claim operational acquisition robustness.',
 'Disclose clean simulation preparation and unknown acquisition representativeness in main limitations.',
 'With an available prepared library, not end-to-end CSI-free preparation.',
 'DISCLOSE_AS_LIMITATION','C1,C6,C10'),
 risk('R5','MEDIUM','Why use environment-labelled B30/C120 preparation?',
 'Library is informed offline even though deployment has no reporting label.',
 'Frozen source/B30/C120 library; protocol disjoint evaluation; deployment pilot-only interface.',
 'Not unsupervised environment discovery; corpus coverage assumed.',
 'Yes','Not needed for current scope.',
 'Separate offline environment knowledge from online condition-blind fit; enumerate preparation data.',
 'No online target label, with offline labelled preparation.',
 'ANSWER_WITH_EXISTING_EVIDENCE','C1,C2,C6'),
 risk('R6','MEDIUM','Why only two correction directions?',
 'Two-dimensional span may not cover unrelated shifts.',
 'Frozen final method, Composed benefit, Boundary/OOD residual gaps; failed physical separability.',
 'No optimal library-size theorem or universal coverage.',
 'Yes for design; optimality not established','Not needed unless optimal/universal span claim retained.',
 'State compact chosen library and coverage limits; no latent-factor interpretation.',
 'Two prepared construction-origin directions, not two independent physical factors.',
 'DISCLOSE_AS_LIMITATION','C2,C8,C12'),
 risk('R7','MEDIUM','Why rank4 and what does its fidelity guarantee?',
 'Development choice may trade fidelity for arithmetic; weighted norm is not target risk.',
 'T20B primary compression endpointC passes; Boundary20 fails broader fidelity; OOD differences mixed.',
 'No universal optimal rank or target-error guarantee.',
 'Yes for frozen empirical choice','No rank sweep here; needed only for an optimal-rank claim.',
 'Report preregistered margin, fixed selection origin and both counterexamples; do not retrofit criteria.',
 'Condition-dependent correction compression fidelity.',
 'ANSWER_WITH_EXISTING_EVIDENCE','C3,C10'),
 risk('R8','MEDIUM','Does adaptation harm source-domain inputs?',
 'Unrestricted coordinates fitted to noisy pilots can lose to Source.',
 'T20B pooled IDloss .218084/.122477dB; U4/U5 separate realization tails.',
 'No worst-case safeguard; no zero-loss guarantee.',
 'Yes','Not needed; do not add ridge/clipping/safeguard.',
 'Retain IDcentral loss and tails beside mismatch gains, with correct evidence labels.',
 'Modest central loss with realization tails.',
 'ANSWER_WITH_EXISTING_EVIDENCE','C4'),
 risk('R9','MEDIUM','Why does C30/200 Boundary expose a large residual and compression loss?',
 'Unseen nominal delay tests representational/compression transfer.',
 'T20B Boundary Source gains2.550301/8.509307; Matched gaps5.990482/5.854487; compression−.238148/+1.086044.',
 'Gap cannot isolate source means, finite statistics, surrogate, span and compression effects; delay rescue stopped.',
 'Yes for boundary observation; complete cause not established','Not needed for disclosed boundary; causal decomposition would need later authorized work.',
 'Main limitations, appendix all methods; no added delay module.',
 'Partial adaptation under the tested delay shift.',
 'DISCLOSE_AS_LIMITATION','C3,C8,C12'),
 risk('R10','MEDIUM','Is low-pilot robustness supported after controlling data-domain changes?',
 'Geometry, dimension and scored data masks change with overhead.',
 'U5A true occupancies and U5B common-domain control; lower-overhead gains retained, extra LR4 sensitivity.',
 'Only100 development trajectories/two conditions and3 densities; remaining geometry changes not isolated.',
 'Yes for bounded development finding','Not needed for sensitivity claim; no new formal density experiment.',
 'State occupancy not just spacing, report absolute deterioration and IDcost; label development.',
 'Benefit remains under tested lower density with substantial tradeoffs.',
 'DISCLOSE_AS_LIMITATION','C5'),
 risk('R11','MEDIUM','Does unseen-profile gain imply close-to-Matched estimation?',
 'Large gain from a weak mismatched Source is not an absolute-quality guarantee.',
 'R1 Sourcegains10.969857/15.722409; Matched gap2.846553/5.543975.',
 'Near-Matched interpretation unsupported.',
 'Yes','Not needed; existing gap directly answers.',
 'Put gain and Matched gap together in OOD table and abstract evidence skeleton.',
 'Partial unseen-profile transfer at CDL-A120/100.',
 'ANSWER_WITH_EXISTING_EVIDENCE','C7,C8'),
 risk('R12','MEDIUM','Can one unseen profile support general OOD robustness?',
 'Single target condition is not a population of OOD domains.',
 '1000 independent realizations within one CDL-A120/100 condition, sealed gatesPASS.',
 'No other unseen profile/array/speed/delay combinations tested independently.',
 'Yes by narrowing','Additional domains would be required for broader claim, not authorized here.',
 'Explicitly disclose one target setting; avoid arbitrary-domain framing.',
 'One independent unseen-profile validation, not universal OOD.',
 'DISCLOSE_AS_LIMITATION','C7'),
 risk('R13','MEDIUM','Why do BER and NMSE rankings disagree, including Perfect/LS/Matched?',
 'ZF channel inverse and nonlinear hard decisions have a different risk than channel MSE.',
 'R1 frozenBER and U8 replay, singular-value/inverse diagnostics; exact mismatch/noise/cross identity.',
 'Diagnostic associations are not a receiver-general causality result; unregularized receiver limited.',
 'Yes for bounded sanity explanation','No new receiver to be added.',
 'BER secondary; U8 explanatory appendix; retain all rankings and Perfect reference rather than hide anomaly.',
 'Descriptively associated with ZF inverse noise enhancement, not general BER superiority.',
 'ANSWER_WITH_EXISTING_EVIDENCE','C9,C12'),
 risk('R14','HIGH','How credible is high-mobility evaluation without explicit ICI?',
 'Symbol-wise frequency response excludes within-symbol Doppler interference.',
 'System03 and Discussion d10 explicitly state ideal synchronized no-ICI abstraction.',
 'No waveform high-mobility robustness demonstration.',
 'Yes for abstraction; waveform claim no','Not needed for channel-statistical claim; full waveform claim would require new evidence.',
 'Place no-ICI assumption upfront and limit mobility meaning to temporal channel statistics.',
 'Temporal-statistics mobility shift within the stated abstraction.',
 'DISCLOSE_AS_LIMITATION','C1,C7,C9'),
 risk('R15','MEDIUM','Are MAC reductions evidence of lower latency/energy?',
 'Hardware cost depends on memory, solver and device transfers.',
 'Frozen cost count332288 vs807936; table excludes uncounted components.',
 'No matched benchmark or measured runtime/energy; more total arithmetic than Source.',
 'Yes by exact proxy definition','Not needed for proxy claim; hardware superiority needs later benchmark.',
 'Report scope of counts, correction storage only and full Source retained.',
 'Arithmetic/storage proxy, not measured deployment efficiency.',
 'ANSWER_WITH_EXISTING_EVIDENCE','C10'),
 risk('R16','HIGH','Were development and formal/validation outcomes kept separate?',
 'Evidence selection after unseal can falsely appear to independently validate tuning.',
 'T20B USED formal; R1 USED validation; U4/U5 development; U8posthoc; frozen manifests and no retuning.',
 'Current manuscript predates Phase2; hierarchy must be made explicit per display.',
 'Yes with provenance','No new statistics; retain data-use records.',
 'Use source-specific labels and immutable provenance; oldU7B NEVER SCIENTIFIC EVIDENCE; disclose T20B interrupted deterministic recovery.',
 'Development suggests, sealed supports under registered criterion.',
 'ANSWER_WITH_EXISTING_EVIDENCE','C1,C4,C5,C6,C7,C9'),
 risk('R17','HIGH','Did target clean H, labels or Matched statistics leak into deployed fitting?',
 'Truth-assisted references and online pilot adaptation coexist in evaluation scripts.',
 'T20B no target label/statistics input; R1 15gates including500reference/1000eval separation, adapter interface and hash witnesses.',
 'Online current noisy pilots are allowed and make adaptation nonlinear; historical proof is not a new independent raw audit.',
 'Yes for frozen gate records','No new scientific run required.',
 'Explain deployment info contract and separate truth-assisted diagnostic oracle/Matched paths; report existing gate assurance scope.',
 'No target clean CSI or environment label enters deployment; offline labelled library remains.',
 'ANSWER_WITH_EXISTING_EVIDENCE','C7,C12'),
 risk('R18','MEDIUM','Is Matched a true oracle and does CI include its training uncertainty?',
 'Finite independently estimated statistics and target means affect reference error.',
 'R1 separate500 trajectories; uncertainty_scope conditional fixed reference; source reference differs from deployedmeans.',
 'Reference-training uncertainty not resampled; not ideal conditional mean.',
 'Yes','No new bootstrap; broader oracle-risk claim would need different reference design.',
 'Rename consistently Matched reference; report conditional CI and finite preparation size.',
 'Finite-statistics target-informed LMMSE reference, not Bayes-optimal oracle.',
 'DISCLOSE_AS_LIMITATION','C8'),
]
RISKS.sort(key=lambda r:({'HIGH':0,'MEDIUM':1,'LOW':2}[r['Severity']],int(r['Risk ID'][1:])))

TERMS=[
 {'canonical':'Source','definition':'frozen source-statistics affine LMMSE','variants':'Source LMMSE, Ws','decision':'full source filter retained'},
 {'canonical':'LR4 Affine','definition':'source plus two rank4 data correction implementations','variants':'Final LR4 Affine, Low-rank Affine','decision':'same method; rank is correction rank'},
 {'canonical':'Dense Affine','definition':'same online coordinates, uncompressed corrections','variants':'Dense','decision':'compression comparator'},
 {'canonical':'Matched reference','definition':'independent finite target-statistics/means LMMSE','variants':'Matched LMMSE reference, oracle','decision':'do not call ideal/Bayesian oracle'},
 {'canonical':'Pilot-LS observation p; source-centered u','definition':'64 complex received pilot coefficients; u=p−mu_ps','variants':'pilot_cv code z; paper z','decision':'distinguish response z640 from u64'},
 {'canonical':'G design; H=G^TG Gram','definition':'real640x2 design and2x2 Gram in U2A','variants':'paper currently G_j denotes2x2 Gram','decision':'proposed notation update only; no source edits'},
 {'canonical':'B_P/B_M','definition':'profile-origin/mobility-origin operator differences','variants':'profile factor, mobility factor','decision':'origin labels, not independent latent factors'},
 {'canonical':'pooled NMSE dB','definition':'10log10(sum error energy/sum truth energy)','variants':'mean trajectory NMSE dB','decision':'distinct estimands; preserve label'},
 {'canonical':'trajectory','definition':'independent statistical unit; frames/windows nested','variants':'frame N, bit N, repeated windows','decision':'counts not independent sample sizes'},
]

ARCHITECTURE=r'''# SP-U9A Manuscript architecture audit (proposals only)

No manuscript, figure or bibliography bytes are changed. Current root is paper/main.tex, metadata in paper/metadata.tex. Eight sections: Introduction, Related Work, System Model, Pilot-Driven Low-Rank Affine Adaptation, Experimental Setup, Results, Discussion, Conclusion. Appendix: notation, full numerical results, selected-development placeholders. Abstract, Introduction and Conclusion are incomplete skeletons, not existing expansive claims.

## Verified current anchors and actions

| Anchor | Action | Reason and bounded proposal |
|---|---|---|
| main.tex abstract comments; sections/01_introduction.tex six comments | ADD | Write later from evidence skeleton; no actual first/SOTA/general/universal claims located in current Introduction. |
| metadata.tex PaperTitle TODO | REWRITE | Remove TODO only in future U9B; use statistical pilot-driven title. |
| Related Work r01/r02 | KEEP | Classical LS/LMMSE foundation; retain information assumptions. |
| Related Work r03/r04 | REWRITE | Adaptive/low-rank precedents incl Özdemir abstract-only boundary; do not infer unverified algorithm details. |
| Related Work r05/r06 | REWRITE + ADD | Retain selection/affine and Turan2024; add tier1 Koller2022 GMM and Fesl2023 MFA, with actual field-level contrast. |
| Related Work r07/r08 | MOVE/REWRITE | Separate learned CE and learned linear-filter/A-MMSE; no neural framing for our analytical estimator. |
| Related Work r09/r10 | REWRITE | Difference = explicit Source+prepared directions+analytical cross-pilot coordinates+correction-only compression. No novelty clearance from multiple statistics. |
| System Model observation/noise/pilot definitions | KEEP | 2x2,64x16,1.92MHz=64x30kHz,no ICI,source receive-power reference. |
| Method Source, prepared operators and affine equations | KEEP | Exact frozen definitions. BP/BM already explicitly construction-origin labels; not a defect to invent. |
| Method coordinate Gram G_j, lines100–104 | REWRITE | Existing Gram notation collides with U2A design G. Proposed G640x2,H2x2; both unambiguously defined. |
| Method pilot fitting | ADD | Five causal current-inclusive frames,4folds; no fold independence, no ridge/clip/fallback; noisy design and residual. |
| Method weighted compression | KEEP + ADD | Weighted operator norm optimality only; no target adaptive MSE guarantee; same Dense/LR4 coordinates. |
| Method arithmetic; Results complexity table | KEEP/REWRITE | Label arithmetic proxy; count exclusions explicit, total greater than Source; storage only two corrections. |
| Setup channel separation/metrics | KEEP + ADD | Source/offline/held-composition/boundary/unseen split table and per-task evidence labels. Primary family definitions kept separately T20B/R1. |
| Results prepared shifts/Composed/Boundary | KEEP | Valid formal results; preserve failures. Rewrite broad opening of BER subsection as frozen-receiver, secondary observation. |
| Results compression | KEEP + ADD | Registered Composed .5dB criterion plus Boundary20 and OOD10/20 mixed behavior. |
| Discussion d01/d02 | KEEP/REWRITE | Empirical unrestricted-coordinate advantage, not covariance hull or causal physical explanation. |
| Discussion d03/d04 | KEEP | Already origin-not-physics and rejected perturbation-separability language. Add one U2 negative sentence, omit diagnostic cascade. |
| Discussion d05/d06 | KEEP | Clean offline assumption and untested operational acquisition explicit. |
| Discussion d07/d08 | REWRITE | Add single sealed CDL-A setting and finite Matched gaps; preserve delay boundary. |
| Discussion d09 | MOVE | Legacy delay development details to compact appendix only if frozen source mapping available; no delay rescue branch. |
| Discussion d10/d11 | KEEP + ADD | No ICI,MAC not latency,BER dependence; short U8 explanation with post-hoc label. |
| Discussion d12/d13 | REWRITE | Compress future work; recommendations do not authorize more tasks or imply performed studies. |
| Conclusion placeholder | ADD | Later bound claims to C1/C2/C3/C7/C10 and limitations; no new inference. |
| Appendix full eight-method formal tables | KEEP | Reproducibility; no removal of negative rows. |
| Appendix development placeholders | REWRITE/MOVE | Replace later with U2 exact assumptions,U4/U5 distributions,feasibility and U8 receiver notes. |

## Introduction phrase audit

Actual wording is comments-only: statistical CE with sparse pilots; mismatch across CDL profile/speed; selection/affine/low-rank established; offline clean library and online analytical coordinates; bounded contributions after review. No actual prose occurrence of first, SOTA, general, universal, excessive deep-learning claim was found in these six comments. The low-pilot context is appropriate motivation but a universal low-pilot robustness result is not established. Recommended scope: fixed sparse pilot geometry, development sensitivity across3 occupancies, prepared-library requirement, single unseen target.

## Related Work proposed four groups

A classical/statistical/adaptive LMMSE and candidate/affine filter combination. B generative mixtures: Koller2022 GMM,Fesl2023 MFA first-tier, Turan2024 related. C learned channel estimators, distinct training/CSI/online compute budgets. D learned linear filters/A-MMSE v5 and pinned public implementation; disclose global-token recipe gap and reproduction gate. No new literature search undertaken; use frozen U1/U2A/U6 sources. Özdemir remains ABSTRACT_ONLY / OPEN RISK.

## Theory allocation

MAIN TEXT: fold predictors and noisy design, real normal equation, dimensions640x2, current-inclusive causal budget; distinguish empirical held-pilot objective from channel-MSE oracle; exact Delta h=X4(u_current)Delta c and quadratic form.

APPENDIX: U2A E10/E11 noisy design construction; rank and lambda_min excitation; deterministic joint design/response perturbation T11/T12; conditional general covariance with actual assumptions; four vector errors and six cross terms; precise population vs realization vs empirical oracles. Tie each bound to actual maps; avoid a standalone OLS tutorial.

OMIT: classic tau²(G^TG)^−1 covariance absent homoscedastic/endogeneity assumptions; conditioning-causes-loss, noisy-coordinate-causes-loss, current-state uniqueness, simple recency, physical-factor separability; numerical correlation tables used to imply causal theory. Failed diagnostic results may be disclosed in appendix, never restated as theorems.

## Exact theory summary for future writing

For each frame/fold, D=[Delta K_P S_O u,Delta K_M S_O u] is16x2 complex, r=(S_H−K_s S_O)u is16 complex. Four folds/fiveframes stack320 complex rows; real expansion G=[ReD;ImD]640x2,z=[Rer;Imr]640. H=G^TG and b=G^Tz reproduce the frozen real normal solve. z=Gc*+epsilon(c*) is an exact residual representation for declared c*, not an exogenous independent-error data-generating model. Both G and residual reuse observation noise across folds/windows.

At fixed current u, X4(u)=[B_P4u,B_M4u]; Delta h=X4(u)Delta c, Q=Delta c^T Re(X4^H X4)Delta c. Relative to a reference error e*, loss change Q+2Re<e*,Delta h> may be negative. If current pilot itself changes, the identity additionally includes the changed u term and cannot be reduced to coefficient movement alone. Bounds condition on adequate excitation, not a fitted rejection threshold. Four errors representation/coordinate/compression/observation-noise are vector definitions; their total energy includes six pairwise cross terms unless orthogonality is established. Population pilot-loss and channel-MSE minimizers need not coincide.

## Proposed experimental sequence

5.1 Main formal T20B, ID and Boundary included. 5.2 Independent sealed CDL-A R1, Source gains and Matched gap together. 5.3 Pilot overhead U5 development/common-domain control. 5.4 Offline budget/subset U4 development. 5.5 Frozen arithmetic/storage proxy. 5.6 Limitations and short frozen-ZF BER note; U8 details appendix.

Reason for placing R1 before U4/U5: readers see the two independent validation blocks together before supporting development sensitivities; evidence classes remain explicit, not merged. No new significance tests for development displays.

## Contribution candidates (not inserted in manuscript)

1. A fully specified Source-anchored affine adaptation procedure with two prepared operator directions and analytical four-fold/five-frame received-pilot coordinates, without online target CSI/covariance/label. C2,C12; novelty wording subject to R1/R2.
2. Correction-only rank4 implementation retaining full Source and dense coordinate fitting, with a registered Composed fidelity result and58.87% matrix-vector arithmetic-proxy reduction versus Dense; explicit transfer counterexamples. C3,C10.
3. Independent formal and single unseen-profile validation of Source-relative estimation benefit, retaining ID loss, Boundary weakness and finite-Matched gaps. C1,C2,C4,C7,C8.
4. Existing development evidence characterizing true pilot-occupancy and finite offline-preparation tradeoffs, with clear separation from formal validation and frozen-receiver BER diagnosis. C5,C6,C9.

## Abstract evidence skeleton only

Problem [C1,C4]: sparse-pilot linear estimation with mismatched frozen statistics. Method [C2,C12]: source+prepared corrections, analytical cross-pilot coordinates, correction-only rank4. Formal [C1,C2,C3]: tested primary improvements and Composed registered compression margin. OOD [C7,C8]: one independent CDL-A120/100 setting, gains plus2.85/5.54dB Matched gap. Cost [C10]:58.87% specified MAC proxy reduction versus Dense, no timing. Limitation [C3,C4,C5,C6,C8,C11]: clean prepared library, ID loss, restricted transfer, uncompleted modern performance comparisons. This is not a final abstract.

## Introduction story

Problem → sparse observations and mismatched source statistics. Gap → updating a full target statistical model can require information/computation not available online; observation-dependent combination itself is established. Hypothesis → a compact prepared correction span with an explicit cross-pilot objective can provide bounded adaptation. Method → exact frozen analytical coordinates and correction-only factors. Evidence → T20B then single sealed CDL-A, supported development sensitivities. Boundary → finite library/clean preparation, ID/compression/Matched gaps, missing modern comparative scores,no ICI,no latency claim. Do not start with architecture or neural framing.

## Title decision and submission fit

Thesis title 《低导频开销条件下的学习增强 MIMO-OFDM 信道估计方法研究》 remains unchanged. Learning-enhanced may describe a broader thesis but can mislead for Paper1, whose procedure uses statistical preparation and analytical adaptation, not a neural online estimator. Current English working string is TODO: Paper title --- Pilot-Driven Low-Rank Affine LMMSE Adaptation. Recommended future Paper1 candidates:
1. Pilot-Driven Low-Rank Affine LMMSE Adaptation for MIMO-OFDM Channel Estimation.
2. Source-Anchored Affine Channel Estimation with Low-Rank Statistical Corrections.
3. Cross-Pilot Coordinate Fitting for Prepared Low-Rank MIMO-OFDM Channel Adaptation.
Preferred candidate1, closest to frozen working title and explicit operator type. No title source changed.

Signal Processing fit judgment uses the frozen2026-10-04 official scope record in docs/sp_u1_signal_processing_venue_profile.md: statistical processing,estimation,filtering,communications. Strong fit: explicit adaptive estimation objective, implementable correction compression, rigorous evidence/provenance and honest boundaries. Biggest scientific-fit risk: whether the specific combination clears substantial prior-art overlap and lack of modern comparator performance. Generic Nature review axes are not Elsevier criteria. No acceptance prediction, no submission format/guide policy claim. No new literature or current-policy search needed for this bounded fit judgment; guide recheck belongs to any later submission task.
'''

LANGUAGE=r'''# SP-U9A Statistical and language audit

| Evidence | Allowed wording | Not allowed |
|---|---|---|
| A formal/sealed | supports/demonstrates within tested setting; significant under the preregistered criterion only for designated cells | universal/general; calling a secondary CI a new registered endpoint |
| B development | suggests/indicates/development evidence/observed under tested settings | independent validation; confirmatory significance absent registered inference |
| C engineering | gatePASS/STOP,static resource bound,not established | inferior accuracy,measured latency from MAC count,failed baseline algorithm |
| D post-hoc | consistent with/descriptively associated with/diagnostic evidence | causes,explains all degradation,changed primary decision |
| E negative | strong hypothesis not supported/contradicted within tested diagnostic setting | universal impossibility of all other mechanisms |

Statistical reporting checklist for every proposed numerical display: independent unit, independent N, scored frames nested, exact aggregation/estimand, CI/test, paired relationship, primary/secondary/evidence class, fixed offline reference uncertainty scope, full condition/SNR and no exclusions.

T20B: N1000 trajectories per condition, five conditions (5000 total),10frames/trajectory,frames4–9 scored. NMSE pools energies before logarithm; BER pools errors/bits. Existing10000 paired trajectory draws.95% descriptive percentile intervals differ from one-sided Bonferroni across10 primary cells (.005/.995). EndpointC margin .5dB only Composed. ID/Boundary/BER descriptive. Interrupted execution was deterministically completed with exact prefix/scientific equivalence, not a new independent replication; preserve existing recovery provenance.

R1: N1000 evaluation trajectories,500 separate Matched-reference trajectories; six scoredframes.10000 saved paired draws shared seven methods/twoSNR. Primary Source−LR4 two-cell familyalpha.05,one-sidedcellalpha.025. Other NMSE/BER95% CIs descriptive; conditional on fixed500-trajectory reference, no uncertainty over retrained reference. OOD N counts realizations within one domain, not1000 independent domains.

U4A/U4B: N100 evaluation trajectories per condition reused;5 subset preparations on one finite500-trajectory corpus are a separate axis, not N500 independent tests. OriginalN500reference shared; nested/overlapping subsets. Median trajectory dB, IQR across preparations and paired penalties distinct from pooled energy NMSE. No formal CI/p-values.

U5A/U5B: N100 per condition; same underlying trajectory across densities/SNR/methods; dF8/16/32 occupancies1/16,1/32,1/64. Native data domain changes; common-domain U5B controls score mask only with frozen predictions/coordinates. Paired penalty median differs from difference of medians; both preserved as separate fields. No new formal inference.

U2B N100; U2C/U2D N20 fixed channel trajectories, six overlapping causal windows and20 noise repeats. Windows and noise repetitions are nested diagnostic observations, not independent channel N. No causal interpretation of Spearman or sign counts.

U8 N1000 reused R1 trajectories.23808000 pooled bits/SNR and5952000 true channel matrices are computational records, not independent sampleN. Fixed-ID exemplars are predesignated diagnostics, not inferential population; no p-value/bootstrap/retest.

Current table audit: tables01_system/evaluation_conditions are descriptive protocol tables so CI/paired/primary not applicable. tables02_nmse/03_ber label the metric,95% CI and paired trajectory percentile bootstrap, but standalone table captions/footnotes omit explicit N1000 and frames4–9 (available in Experimental Setup and figure captions); add those plus primary/descriptive label and fixed-library conditional scope. The BER table copies an irrelevant Perfect CSI zero-NMSE note; replace later with a receiver/bit-denominator note, without changing data. tables04_endpoints has family10,.005/.995,.5criterion; add trajectoryN and aggregation pointer if standalone. tables05_complexity N/CI/test/paired not applicable and omitted costs stated. tables06_summary currently deltas/uncertainty pointer but needs standalone N,pooled-difference definition and descriptive-versus-primary note. Compact02/03 variants duplicate full appendix and are not additional evidence; their standalone N/paired/unit metadata are incomplete and full CI is deferred. result_numbers/summary_numbers are generated macros, not new evidence.

Existing frozen CSV/JSON digits are preserved. Rounded prose numbers are display only. Differences between pooled dB,median paired trajectory difference,and difference of medians are not EVIDENCE_CONFLICT unless definitions and populations are identical. Actual same-cell disagreements are logged as EVIDENCE_CONFLICT with both sources; never silently resolved.
'''

NEGATIVE=[
 {'ID':'N1','evidence':'U2B/U2C/U2D and current Discussion d04','placement':'APPENDIX + one MAIN LIMITATIONS sentence','message':'Condition number alone, noise-coordinate movement→loss,unique current-frame advantage and simple recency are not established; no physical-factor decomposition. Keep exact propagation and its cross term.'},
 {'ID':'N2','evidence':'T20B Boundary10/20','placement':'MAIN LIMITATIONS and formal figure/table','message':'Source benefit coexists with5.990/5.854dB Matched gaps;20dB compression loss1.086dB; no delay rescue.'},
 {'ID':'N3','evidence':'U3B-R1 and U6A resource gates','placement':'MAIN LIMITATIONS + APPENDIX contract/gates','message':'Modern baseline performance unestablished; resource gates are feasibility limits, not scientific inferiority.'},
 {'ID':'N4','evidence':'T20B ID + U4/U5 distributions','placement':'MAIN RESULTS + APPENDIX tails','message':'Modest central ID loss and realization tails; no no-degradation guarantee.'},
 {'ID':'N5','evidence':'U5A/B native/common controls','placement':'MAIN sensitivity subsection','message':'Low-occupancy benefit remains but absolute/extra LR4 loss and ID tails worsen. No scarcity robustness theorem.'},
 {'ID':'N6','evidence':'R1 sealed Matched gap/compression','placement':'MAIN OOD RESULTS + LIMITATIONS','message':'2.85/5.54dB finite-Matched gap and mixed compression, one unseen domain only.'},
 {'ID':'N7','evidence':'U8 frozen-ZF diagnostic','placement':'MAIN short receiver note + APPENDIX','message':'BER ranking differs from NMSE and Perfect/Matched ranking; descriptive inverse-noise dependence, no causal/general BER claim.'},
 {'ID':'N8','evidence':'U4B subset replication','placement':'MAIN budget subsection + APPENDIX distributions','message':'Low-budget tail variation retained; one central/tail counterexample prevents uniform monotonicity claim.'},
 {'ID':'N9','evidence':'Invalidated oldU7B bookkeeping metadata','placement':'INTERNAL provenance, not scientific results narrative','message':'Never use old outcomes; no old/new performance comparison. Disclose integrity correction only if needed in reproducibility history.'},
]

REGISTRY=[
 {'ID':'F1','source task':'Frozen method','source files':['paper/figures/method_workflow.pdf'],'evidence class':'C algorithm design','main message':'What is prepared offline and adapted online?','publication worthy':'YES existing diagram; any future layout only after U9B authorization','duplicate':'NO','section':'Method MAIN'},
 {'ID':'F2','source task':'T20B','source files':['paper/figures/formal_nmse.pdf','outputs/t20b_formal_nmse.csv','outputs/t20b_formal_bootstrap.csv'],'evidence class':'A FORMAL','main message':'Absolute estimation quality across all five conditions including ID/Boundary','publication worthy':'YES','duplicate':'Complementary to differences F3; full appendix table duplication intentional lookup','section':'Main formal'},
 {'ID':'F3','source task':'T20B','source files':['paper/figures/formal_endpoints.pdf','outputs/t20b_formal_bootstrap.csv'],'evidence class':'A FORMAL','main message':'Registered Source/Convex/compression criteria plus descriptive boundary','publication worthy':'YES','duplicate':'Do not repeat all endpoint numbers in an extra full-size plot','section':'Main formal'},
 {'ID':'F4','source task':'U5A/B','source files':['outputs/sp_u5a_figures/D_Paired_reduction.pdf','outputs/sp_u5b_figures/A_Native_Common.pdf','outputs/sp_u5b_figures/B_Common_Gain.pdf'],'evidence class':'B DEVELOPMENT','main message':'Retained gain versus absolute/extra pilot sensitivity with common-domain control','publication worthy':'YES select or later re-layout existing data; not three redundant main figures','duplicate':'Native/Common A control complements B gain; U5A C gain overlaps B','section':'Pilot sensitivity MAIN one figure slot'},
 {'ID':'F5','source task':'U4B','source files':['outputs/sp_u4b_figures/A_Composed_budget_penalty.pdf','outputs/sp_u4b_figures/C_Budget_vs_subset.pdf','outputs/sp_u4b_figures/E_ID_and_tails.pdf'],'evidence class':'B DEVELOPMENT','main message':'Finite preparation budget central shift versus subset variability/tails','publication worthy':'YES one main selection or future rearrangement from existing frozen data','duplicate':'U4A superseded in coverage, retained provenance; E mostlyappendix','section':'Offline budget MAIN one slot'},
 {'ID':'F6','source task':'U7B-R1','source files':['outputs/sp_u7b_r1/figures/Figure_A.pdf','outputs/sp_u7b_r1/figures/Figure_B.pdf','outputs/sp_u7b_r1/table_A.csv','outputs/sp_u7b_r1/table_B.csv'],'evidence class':'A SEALED','main message':'Single unseen-setting gain with absolute/Matched/compression limits','publication worthy':'YES one main slot; choose existing Figure_A and move distribution Figure_B to appendix if redundant','duplicate':'Not new domain evidence when FigureB uses same1000','section':'Independent validation'},
 {'ID':'FA1','source task':'T20B','source files':['paper/figures/formal_ber.pdf'],'evidence class':'A FORMAL SECONDARY','main message':'Frozen-ZF secondary BER, all eight methods','publication worthy':'YES appendix','duplicate':'Do not promote BER over primary NMSE','section':'APPENDIX receiver note'},
 {'ID':'FA2','source task':'U8','source files':['outputs/sp_u8a/figures/Figure1_BER.pdf','outputs/sp_u8a/figures/Figure2_Conditioning.pdf','outputs/sp_u8a/figures/Figure3_Fixed_IDs.pdf'],'evidence class':'D POST-HOC','main message':'Receiver sanity, descriptive inverse geometry and exact fixed-ID energy cross terms','publication worthy':'YES diagnosticappendix; fixedIDs may supplementary','duplicate':'Figure1 shares R1 BER; Perfect added diagnostic only','section':'APPENDIX'},
 {'ID':'FA3','source task':'U2','source files':['outputs/sp_u2b_table3_hypothesis_audit.csv','outputs/sp_u2c_table3_hypothesis_audit.csv','outputs/sp_u2d_table4_hypotheses.csv'],'evidence class':'B/E diagnostic/negative','main message':'Why mechanisms are not claimed','publication worthy':'Compact appendix statement/table, no main correlation gallery','duplicate':'Do not expand all diagnostic figures','section':'APPENDIX'},
 {'ID':'T1','source task':'Method/T20B/U5/R1','source files':['paper/tables/01_system.tex','paper/tables/evaluation_conditions.tex','outputs/sp_u5a_table1_geometry.csv'],'evidence class':'Protocol','main message':'Configuration,offline/online info and evidence-domain split','publication worthy':'YES future combine protocol tables; no new results','duplicate':'Combine existing configuration/conditions if sensible','section':'Setup MAIN'},
 {'ID':'T2','source task':'U1/U2A/U6','source files':['docs/sp_u2a_prior_art_amendment.md','outputs/sp_u6a_ammse_version_ledger.json'],'evidence class':'Frozen primary-source literature audit','main message':'Closest-work interface and low-rank object differences; no novelty clearance','publication worthy':'YES future compact field comparison','duplicate':'Use full fields in appendix, keyfields main','section':'Related Work MAIN'},
 {'ID':'T3','source task':'T20B','source files':['paper/tables/06_summary.tex','paper/tables/04_endpoints.tex','outputs/t20b_formal_bootstrap.csv'],'evidence class':'A FORMAL','main message':'Gain/compression/ID/Boundary key contrasts with existing adjusted bounds','publication worthy':'YES future single compact key table; no additional significance tests','duplicate':'Full02/03 tables remain appendix','section':'Formal results MAIN'},
 {'ID':'T4','source task':'U7B-R1','source files':['outputs/sp_u7b_r1/table_A.csv','outputs/sp_u7b_r1/table_B.csv'],'evidence class':'A SEALED','main message':'NMSE/gain/Matched/compression with primary vs secondary labels','publication worthy':'YES may compact key columns later','duplicate':'Full method tableappendix if plot duplicates','section':'Unseen validation MAIN'},
 {'ID':'T5','source task':'Frozen count','source files':['paper/tables/05_complexity.tex'],'evidence class':'C analytical proxy','main message':'MAC/storage tradeoff and exclusions','publication worthy':'YES','duplicate':'NO','section':'Complexity MAIN'},
 {'ID':'TA1','source task':'U3/U6','source files':['outputs/sp_u3b_r1_A_gate.json','outputs/sp_u3b_r1_medium_gate.json','outputs/sp_u6a_resource_gate.json'],'evidence class':'C ENGINEERING','main message':'Attempted baseline contract and stopped performance branches','publication worthy':'YES compact appendix; not performance ranking table','duplicate':'NO','section':'APPENDIX limitations'},
]

def provenance():
 entries=[]
 specs=[
 ('T20B','USED formal / completed with audited deterministic recovery','A','outputs/t20b_formal_results.json','5 fixed conditions,SNR10/20','1000/condition;5000total','execution_commit'),
 ('SP-U2A','Exact identity/assumption audit, PASS first-layer utility','B','outputs/sp_u2a_theory_validation.json','frozen geometry; synthetic algebra fixtures','N not scientific evaluation','head'),
 ('SP-U2B','Development diagnostic, mixed mechanism','B/E','outputs/sp_u2b_analysis.json','Composed SNR10/20','100 trajectories,6windows','head'),
 ('SP-U2C','Development conditional-noise diagnostic','B/E','outputs/sp_u2c_analysis.json','fixed20channels;20receivernoise repeats/window','20trajectories,6windows','head'),
 ('SP-U2D','Development isolation, strong mechanism not supported','B/E','outputs/sp_u2d_analysis.json','four-frame leave-age-out variants','20trajectories','head'),
 ('SP-U3A','Portability/contract gate only','C','outputs/sp_u3a_baseline_gate.json','MFA candidate interface/feasibility','not target performance','HEAD'),
 ('SP-U3B','Resource FAIL, no target comparison','C','outputs/sp_u3b_resource_gate.json','full-dimensional MFA','not target performance','HEAD'),
 ('SP-U3B-R1','MediumPASS; fullAFAIL_TIME_CAP; performance branchSTOP','C','outputs/sp_u3b_r1_decision.json','MFA semantics-preserving memory optimization','one medium fit;fullA stopped;Bnotrun','HEAD'),
 ('SP-U4A','Development single subset realization','B','outputs/sp_u4a_analysis.json','ID/Composed;budgets50,100,250,500','100eval/condition','HEAD'),
 ('SP-U4B','Development CaseA, branchFROZEN','B','outputs/sp_u4b_analysis.json','five subset realizations,shared500corpus','100eval/condition reused;5prepnot500tests','HEAD'),
 ('SP-U5A','Development CaseB, no numerical breakdown','B','outputs/sp_u5a_analysis.json','ID/Composed;three true occupancies','100/condition paired','HEAD'),
 ('SP-U5B','Development CaseA common-domain control;FROZEN','B','outputs/sp_u5b_analysis.json','sameU5inputs/coordinates','100/condition reused','HEAD'),
 ('SP-U6A','STOP-D; faithful reproduction NOT_ESTABLISHED','C','outputs/sp_u6a_decision.json','A-MMSE current repo/v5 disclosed gap/project shape','no author network/project optimizer step','HEAD'),
 ('SP-U7B-R1','USED validation;15/15PASS;oldrunexcluded','A','outputs/sp_u7b_r1/analysis.json','CDL-A120/100,SNR10/20','1000eval;500separateMatched','git_head'),
 ('SP-U8A','POST-HOC diagnostic;primaryimpactNONE;CaseA','D','outputs/sp_u8a/analysis.json','sameR1/frozenZF','1000reused;fixedIDs diagnostic','HEAD'),
 ]
 for task,status,cls,path,cond,n,headkey in specs:
  d=load(path);head=d.get(headkey)
  head_source=path+'#'+headkey
  if not head:
   prefix=task.lower().replace('-','_')
   manifests=[f'outputs/{prefix}_delivery_manifest.json']
   if task=='SP-U7B-R1':manifests=['outputs/sp_u7b_r1/delivery_manifest.json']
   if task=='SP-U8A':manifests=['outputs/sp_u8a/delivery_manifest.json']
   for m in manifests:
    if (ROOT/m).exists():
     data=load(m)
     for key in ('HEAD','head','git_head'):
      if data.get(key):head=data[key];head_source=m+'#'+key;break
    if head:break
  entries.append({'task ID':task,'status':status,'git HEAD':head or 'NOT_RECORDED_IN_SELECTED_SOURCE',
   'HEAD provenance':head_source if head else 'No current HEAD substituted',
   'condition':cond,'sample size':n,'evidence class':cls,
   'used/development/formal status':'USED formal' if task=='T20B' else 'USED validation' if task=='SP-U7B-R1' else 'post-hoc diagnostic' if task=='SP-U8A' else 'engineering' if cls=='C' else 'development/math audit',
   'whether reusable for tuning':False,'main frozen output files':[path]})
 entries.append({'task ID':'old SP-U7B','status':'INVALIDATED / NEVER SCIENTIFIC EVIDENCE',
 'git HEAD':'NOT_USED','HEAD provenance':'outputs/sp_u7b_r1/final_verification.json#old_isolation metadata only',
 'condition':'old bookkeeping-invalidated sealed run','sample size':'excluded','evidence class':'EXCLUDED',
 'used/development/formal status':'NEVER SCIENTIFIC EVIDENCE','whether reusable for tuning':False,
 'main frozen output files':['outputs/sp_u7b_r1/final_verification.json#old_isolation']})
 return entries

def main():
 # Reports MUST all be frozen before editorial consistency/synthesis pass.
 reports=[];locks=[]
 for i in (1,2,3):
  p=OUT/f'reviewer{i}_frozen.json';lock=load(f'outputs/sp_u9a/reviewer{i}_lock.json')
  assert p.exists() and (OUT/f'reviewer{i}_frozen.md').exists()
  listed=lock['files']
  if isinstance(listed,dict):
   listed=[{'path':k if '/' in k else 'outputs/sp_u9a/'+k,'sha256':v} for k,v in listed.items()]
  for item in listed:assert sha(ROOT/item['path'])==item['sha256'], 'Frozen review changed'
  assert lock['packet_sha256']==sha(OUT/'immutable_packet.txt')
  reports.append(json.loads(p.read_text(encoding='utf-8-sig')));locks.append(lock)
 # Add precise source index for all root matrix/provenance/display references.
 source_inventory=load('outputs/sp_u9a/source_inventory.json')
 known={r['path'] for r in source_inventory}
 extra=[]
 for collection,field in [(CLAIMS,'Evidence source'),(REGISTRY,'source files')]:
  for row in collection:
   values=row[field] if isinstance(row[field],list) else row[field].split(';')
   extra.extend(v.strip().split('#')[0] for v in values)
 for task in ('2b','2c','2d','3a','3b','4a','4b','5a','5b'):
  extra.append(f'outputs/sp_u{task}_delivery_manifest.json')
 extra += ['outputs/sp_u4b_table6_hypotheses.csv','outputs/sp_u5b_ID_tradeoff.csv',
  'outputs/sp_u5b_extra_sensitivity.csv','outputs/sp_u5b_gains.csv','outputs/sp_u5a_table4_gains.csv',
  'docs/sp_u1_closest_work_matrix.md','outputs/sp_u1_literature_database.json']
 for entry in provenance():
  pointer=entry['HEAD provenance'].split('#')[0]
  if (ROOT/pointer).is_file():extra.append(pointer)
 for path in sorted(set(extra)-known):
  p=ROOT/path
  assert p.is_file(), 'Unlocated source: '+path
  source_inventory.append({'path':path,'absolute_path':str(p),'exists':True,'sha256':sha(p),'source_role':'root consolidation addendum; not supplied as shared reviewer conclusions'})
 for row in source_inventory:assert sha(ROOT/row['path'])==row['sha256']
 write_json('outputs/sp_u9a/full_source_inventory.json',source_inventory)
 write_json('outputs/sp_u9a_claim_evidence_matrix.json',CLAIMS)
 write_csv('outputs/sp_u9a_claim_evidence_matrix.csv',CLAIMS)
 write_text('docs/sp_u9a_master_claim_evidence_matrix.md','# SP-U9A Master Claim–Evidence Matrix\n\nFrozen evidence only. A formal/sealed; B development; C engineering; D post-hoc; E rejected hypotheses. No source is reusable for tuning in U9A.\n\n'+md_rows(CLAIMS))
 write_csv('outputs/sp_u9a_reviewer_risk_register.csv',RISKS)
 prov=provenance();write_json('outputs/sp_u9a_evidence_provenance.json',prov)
 write_text('docs/sp_u9a_manuscript_architecture_audit.md',ARCHITECTURE)
 write_text('docs/sp_u9a_writing_language_rules.md',LANGUAGE+'\n\n## Canonical terminology\n\n'+md_rows([{'ID':t['canonical'],**t} for t in TERMS]))
 write_json('outputs/sp_u9a/terminology_ledger.json',TERMS)
 write_json('outputs/sp_u9a/figure_table_registry.json',REGISTRY)
 write_text('docs/sp_u9a_figure_table_registry.md','# SP-U9A Figure/table registry\n\nRecommended main budget:6 figures,5 tables. One question per figure. This is a selection/layout blueprint, not new figure production. Full formal tables/BER, U2 diagnostics, U4/U5 tails, R1 distributions and U8 diagnostics to appendix/supplement. No scientific figures rendered.\n\n'+md_rows(REGISTRY))
 write_text('docs/sp_u9a_negative_evidence_ledger.md','# SP-U9A Negative/limitation placement\n\n'+md_rows(NEGATIVE))
 excluded=[
  {'ID':'X1','source':'oldU7B invalidation via R1 final_verification old_isolation only','status':'NEVER SCIENTIFIC EVIDENCE','why':'Bookkeeping0-vs1500scientific witness state invalidation; sealed old outcomes not read or compared.','table rule':'Exclude all performance values, no favorable/negative comparison.'},
  {'ID':'X2','source':'U2B/C/D hypothesis audits;U2A assumption audit','status':'Negative mechanism evidence only','why':'Strong condition/noise/current/recency/physical-factor causal stories are not supported.','table rule':'Not supportive mechanism theorem; retain negative appendix/limitations, math identities distinct.'},
  {'ID':'X3','source':'U3B/R1 resource/time gates','status':'Uncompleted MFA performance branch','why':'Memory semantics equivalence/mediumPASS do not complete full training; fullAfit timecapFAIL, Bnotrun.','table rule':'No MFA score/performance ranking. Resource table permitted as feasibility.'},
  {'ID':'X4','source':'U6A decision/resource/version ledger','status':'Uncompleted A-MMSE performance branch','why':'Faithful project-shape source-derived storage exceeds cap; no project optimizer step; v5/repo recipe gap andlicenseopen.','table rule':'No A-MMSE score or inferiority claim. Static bound not measured allocation.'},
 ]
 write_text('docs/sp_u9a_invalidated_excluded_evidence.md','# SP-U9A Invalidated/excluded evidence\n\n'+md_rows(excluded))
 write_json('outputs/sp_u9a/invalidated_excluded_evidence.json',excluded)
 # Editorial audit checks existing printed outcomes only, not raw re-estimation.
 t=load('outputs/t20b_formal_results.json');original=load('outputs/t20b_formal_numerical_results.json')
 conflicts=[]
 for key in ('nmse_results','ber_results','bootstrap_results','boundary_results','complexity'):
  if t[key]!=original[key]:conflicts.append({'status':'EVIDENCE_CONFLICT','source1':'t20b_formal_results.json#'+key,'source2':'t20b_formal_numerical_results.json#'+key,'value1':t[key],'value2':original[key]})
 for filename,key in [('t20b_formal_nmse.csv','nmse_results'),('t20b_formal_ber.csv','ber_results')]:
  rows=list(csv.DictReader((ROOT/'outputs'/filename).open(encoding='utf-8-sig')))
  metric='nmse_db' if key=='nmse_results' else 'ber'
  indexed={(r['environment'],int(r['snr_db']),r['method']):r for r in rows}
  for row in t[key]:
   identity=(row['environment'],row['snr_db'],row['method']);csv_value=indexed[identity][metric];json_value=row[metric]
   if json_value is not None and float(csv_value)!=json_value:
    conflicts.append({'status':'EVIDENCE_CONFLICT','source1':'outputs/'+filename,'source2':'outputs/t20b_formal_results.json#'+key,'cell':identity,'value1':csv_value,'value2':json_value})
 u=load('outputs/sp_u7b_r1/analysis.json');results=u['results']
 identities=[]
 for i,snr in enumerate((10,20)):
  # Roundoff check only, not a new estimand or CI. Existing metrics remain unchanged.
  identities.append({'snr':snr,'gain_identity_residual':(results['nmse_db'][i][1]-results['nmse_db'][i][5])-results['gain'][i],
   'matched_gap_identity_residual':(results['nmse_db'][i][5]-results['nmse_db'][i][6])-results['secondary']['LR4_minus_Matched']['point'][i],
   'compression_identity_residual':(results['nmse_db'][i][5]-results['nmse_db'][i][4])-results['secondary']['LR4_minus_Dense']['point'][i]})
 audit=[
  {'audit_id':'A1','status':'passed' if not conflicts else 'confirmed_internal_error','location':'T20B CSV/JSON shared cells','check':'exact stored scalar equivalence, no statistical recomputation','result':len(conflicts),'impact':'same-estimand conflicts never resolved by favorable selection','resolution_test':'record both and advisor review if conflict','reviewer_overlap':'root editorial only'},
  {'audit_id':'A2','status':'passed','location':'R1 analysis.results','check':'algebra of existing headline scalars','result':identities,'impact':'supports stored reported signs','resolution_test':'no altered outcomes','reviewer_overlap':'root editorial only'},
  {'audit_id':'A3','status':'passed','location':'T20B/R1/U4/U5/U8 metric definitions','check':'independent trajectory and estimand distinction','result':'Pooled dB vs median paired dB vs difference of medians kept separate; no new tests','impact':'prevents pseudoreplication and false numeric conflicts','resolution_test':'per-table footnotes in blueprint','reviewer_overlap':'root editorial only'},
  {'audit_id':'A4','status':'passed','location':'oldU7B/R1 final verification','check':'invalidated oldrun excluded','result':'metadata only; no old aggregate outcome read','impact':'no scientific reuse','resolution_test':'maintain exclusion','reviewer_overlap':'root editorial only'},
  {'audit_id':'A5','status':'not_assessable','location':'external author reproductions','check':'MFA/A-MMSE competitive performance','result':'scientific comparison uncompleted, static/engineering evidence only','impact':'R3 OPEN','resolution_test':'advisor resolves required evidence, no automatic experiments','reviewer_overlap':'root editorial only'},
  {'audit_id':'A6','status':'passed','location':'Current Introduction/Abstract/Conclusion skeletons','check':'no invented current overclaim quotations','result':'comments-only incomplete portions distinguished from written results','impact':'architecture proposes additions, no manuscript patch','resolution_test':'source-preservation guard','reviewer_overlap':'root editorial only'},
  {'audit_id':'A7','status':'provenance_gap','location':'U3A/U3B selected gate/delivery metadata','check':'task execution HEAD independently recorded','result':'Selected U3A/U3B records omit executionHEAD; record gap without substituting currentHEAD; author repo pin kept distinct. U6A HEAD is available in delivery manifest.','impact':'historical metadata gap does not invent provenance','resolution_test':'report NOT_RECORDED for missing field','reviewer_overlap':'root editorial only'},
  {'audit_id':'A8','status':'confirmed_internal_error','location':'paper/sections/04_proposed_method.tex Offline Correction Operators vs Setup; paper/tables/03_ber.tex footnote','check':'reporting terminology/duplicated footnote','result':'Unqualified RMS vs nominal delay parameter; irrelevant zero-NMSE note in BER footnote. Neither establishes numeric/configuration error.','impact':'local editorial corrections proposed only','resolution_test':'future terminology alignment and BER footnote correction','reviewer_overlap':'R1-m1/R3-m2 for delay; BER footnote root-only'},
  {'audit_id':'A9','status':'aggregation_ambiguity','location':'paper/tables/02_nmse.tex,03_ber.tex,06_summary.tex','check':'standalone table independentN/conditional scope','result':'Definitions exist in Setup but table-alone metadata incomplete','impact':'potential misread frame count or unconditional CI','resolution_test':'add proposed standalone footnotes','reviewer_overlap':'R2-M3 for conditional scope'},
 ]
 write_json('outputs/sp_u9a/evidence_conflicts.json',conflicts)
 write_json('outputs/sp_u9a/forensic_consistency_audit.json',audit)
 before=load('outputs/sp_u9a/preservation_before.json')
 changed=[p for p,d in before.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=d]
 assert not changed, 'Frozen source changed: '+str(changed)
 # Derived decision follows root risk register, not reviewer acceptance votes.
 opens=[r['Risk ID'] for r in RISKS if r['Severity']=='HIGH' and r['Action']=='PAPER_RISK_OPEN']
 decision={'task':'SP-U9A','verdict':'NOT_READY' if opens else 'READY_WITH_DISCLOSURES',
 'interpretation':'Writing blueprint complete; HIGH novelty/comparator risks require mentor decision before authorizing U9B. This is not rejection of validated Source-relative gains.',
 'HIGH_PAPER_RISK_OPEN':opens,'claim_status_counts':{s:sum(c['Status']==s for c in CLAIMS) for s in sorted({c['Status'] for c in CLAIMS})},
 'new_scientific_experiments':0,'new_bootstrap':0,'new_statistical_threshold':0,'method_changes':0,'manuscript_changes':0,'new_figures':0,'new_literature_search':0,'git_commit':False,
 'frozen_input_files_verified':len(before),'changed_frozen_files':changed,'evidence_conflicts':len(conflicts),
 'reviewers':'3 isolated contexts, preassigned emphases, reports frozen before editorial audit; no consensus synthesized before lock',
 'experimental_branch':'FROZEN','next_action':'STOP; mentor reviews R1/R2/R3 and decides whether claims can be narrowed or additional evidence is needed; do not execute U9B or reopen stopped branches',
 'decision_delta':'Primary T20B/R1 verdicts unchanged; readiness introduces writing/novelty/comparison gate only.',
 'git_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
 write_json('outputs/sp_u9a_decision.json',decision)
 write_json('outputs/sp_u9a/preservation_after.json',{'verified_files':len(before),'changed':changed,'manuscript_unchanged':True,'review_setup_sha256':sha(OUT/'review_setup.json')})
 # Root synthesis is authored separately after independent reports; do not rewrite frozen reports.
 synthesis={
  'reports_frozen_before_comparison':True,
  'review_counts':[{ 'reviewer':i+1,'major':len(r.get('major_concerns',[])),
    'minor':len(r.get('minor_comments',[])),
    'blocking_ids':[x['id'] for x in r.get('major_concerns',[]) if str(x.get('blocking')).lower()=='yes']} for i,r in enumerate(reports)],
  'consensus_strengths':'Specific information contract, same-library paired coordinate/compression contrasts, retained ID/Boundary/model limitations.',
  'consensus_blocking_concerns':[],
  'other_consensus_major_concerns':[
   {'key':'closest-work positioning','ids':['R1-M1','R3-M1'],'blocking_agreement':False},
   {'key':'pilot surrogate to channel risk bridge','ids':['R1-M3','R2-M1','R3-M2'],'blocking_agreement':False},
   {'key':'external comparator evidence/information fairness','ids':['R1-M2','R2-M2'],'blocking_agreement':False},
   {'key':'new evidence integration with OOD/BER scope','ids':['R1-M4','R2-M4','R3-m3'],'blocking_agreement':False}],
  'consequential_single_reviewer_concerns':[
   {'id':'R2-M3','key':'fixed-library/reference uncertainty scope and preparation metadata'},
   {'id':'R3-m1','key':'frame-level causality and current-frame buffering, not immediate symbol-level processing'}],
  'minor_revision_checklist':[
   {'key':'nominal delay vs empiricalRMS','ids':['R1-m1','R3-m2']},
   {'key':'weighted norm not adaptive target-risk guarantee','ids':['R2-m1']},
   {'key':'source-referenced vs realizedSNR and pooled power','ids':['R2-m2']},
   {'key':'current-frame receive/buffer timing','ids':['R3-m1']}],
  'emphasis_divergence':'Reviewer1 alone calls originality blocking; Reviewer2/3 do not invalidate bounded empirical results. No consensus blocking label manufactured. Root R1/R2/R3 HIGH open register is mentor-facing readiness risk, not acceptance vote.',
  'forensic_audit_separate':audit,
  'scope_fit':'Field-local statistical adaptive estimation, not assumed Nature-scale significance or SP acceptance forecast.',
  'unsupported_claims':'First/SOTA/universal,external baseline superiority,no IDloss,lossless everywhere,causal condition/noise/current/recency narrative,hardware latency,all-window ideal oracle.'}
 write_json('outputs/sp_u9a/reviewer_synthesis.json',synthesis)
 write_text('docs/sp_u9a_reviewer_red_team.md','# SP-U9A Reviewer attack register\n\n'+
 'Three separate frozen reports are preserved under outputs/sp_u9a/reviewer1–3_frozen.json/md. The following root mandatory R1–R18 register is not a reviewer consensus ledger. Actions never authorize new experiments.\n\n'+md_rows(RISKS))
 write_text('docs/sp_u9a_reviewer_red_team.md',(ROOT/'docs/sp_u9a_reviewer_red_team.md').read_text(encoding='utf-8')+'\n\n## Post-review synthesis (not returned to reviewers)\n\n'+json.dumps(synthesis,ensure_ascii=False,indent=2))
 write_text('docs/sp_u9a_manuscript_architecture_audit.md',ARCHITECTURE+'''\n\n## Independent review amendment\n\nREWRITE proposed only: Method offline preparation presently calls100ns unqualified RMS while Setup/table state nominal delay parameter. Standardize nominal model delay and distinguish empirical array-weighted RMS (R1-m1/R3-m2). ADD frame-level processing timing: current-frame pilot symbols0/4/8/12 must arrive before retrospective data estimation; early data buffered. Causal across frames is not symbol-immediate/low-latency (R3-m1). ADD source-referenced versus realized receive-SNR footnote with existing power summaries (R2-m2). ADD fixed-library/reference conditional uncertainty and per-origin preparation N/version (R2-M3). No method/configuration edits.\n''')
 write_json('outputs/sp_u9a/statistical_table_audit.json',{'T20B':{'unit':'trajectory','N_per_condition':1000,'aggregation':'pooled energy; pooled bits','paired':True,'draws':10000,'primary_family_cells':10},'R1':u['statistics_audit'],'U4/U5':{'unit':'trajectory','N_per_condition':100,'formal':False},'U8':{'unit':'reused R1 trajectory','N':1000,'new_inference':False},'current_table_notes':'See writing_language_rules.md for every paper table family; standalone metadata additions proposed only.'})
 print(json.dumps(decision,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
