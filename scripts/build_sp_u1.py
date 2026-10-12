"""Build the SP-U1 literature/upgrade planning package from audited source notes.

No imports from src, no experiment results ingestion, and no manuscript writes.
"""
import json
from pathlib import Path
from textwrap import dedent

ROOT=Path(__file__).resolve().parents[1]
FIELDS=['title','year','venue','DOI/arXiv','problem','system model','pilot setting',
        'estimation object','adaptation object','offline information','online information',
        'target clean CSI required?','target labels required?','environment label required?',
        'coefficient/adaptation mechanism','low-rank object','complexity strategy',
        'distribution-shift evaluation','OOD evaluation','pilot-overhead study',
        'theoretical analysis','code available?','closest overlap with us','key difference',
        'uncertainty','source quality']

WORKS=[]
def work(key,title,year,venue,identifier,source,quality,locators,close=False,**facts):
    row={f:'UNKNOWN' for f in FIELDS}
    row.update(title=title,year=year,venue=venue)
    row.update({'DOI/arXiv':identifier,'source quality':quality})
    aliases={'coefficient_adaptation_mechanism':'coefficient/adaptation mechanism',
             'low_rank_object':'low-rank object',
             'distribution_shift_evaluation':'distribution-shift evaluation',
             'pilot_overhead_study':'pilot-overhead study'}
    for k,v in facts.items():
        field=aliases.get(k,k.replace('_',' '))
        if field not in FIELDS:
            raise ValueError('Unrecognized literature field: '+field)
        row[field]=v
    row.update(id=key,source_url=source,read_locators=locators,high_core_proximity=close,
               access_date='2026-10-04',technical_status=quality)
    WORKS.append(row)

work('oz2006','Toward real-time adaptive low-rank LMMSE channel estimation of MIMO-OFDM systems',2006,'IEEE TWC','10.1109/TWC.2006.04713',
     'https://www.researchgate.net/publication/3433671_Toward_real-time_adaptive_low-rank_LMMSE_channel_estimation_of_MIMO-OFDM_systems','ABSTRACT_ONLY',['public journal abstract'],True,
     problem='LMMSE complexity',system_model='MIMO-OFDM',pilot_setting='comb-type',estimation_object='frequency-domain channels',
     low_rank_object='generalized low-rank LMMSE',complexity_strategy='subspace tracking',
     closest_overlap_with_us='adaptive low-rank MIMO LMMSE; pilot/non-pilot subspaces',key_difference='Detailed difference unresolved',
     uncertainty='Rank construction, tracking equations, statistics assumptions and online update implementation UNKNOWN. No lawful full text obtained.')
work('ls2010','Least-squares adaptation of affine combinations of multiple adaptive filters',2010,'IEEE ISCAS','10.1109/ISCAS.2010.5538026',
     'https://www.researchgate.net/publication/221380510_Least-squares_adaptation_of_affine_combinations_of_multiple_adaptive_filters','AUTHOR_FULL_TEXT',['Sec II A/B; equations 1–9; Sec III'],True,
     problem='filter mixing',system_model='multiple adaptive filters',estimation_object='desired signal',adaptation_object='unrestricted mixing weights',
     online_information='common regressor and desired response',coefficient_adaptation_mechanism='windowed LS of output differences',
     theoretical_analysis='steady-state comparison with optimal combination',closest_overlap_with_us='Difference-output LS and dependent source weight',
     key_difference='No verified sparse cross-pilot Wiener completion contract',uncertainty='Does not establish our complete pilot/noise/MIMO interface')
work('edfors1996','OFDM channel estimation by singular value decomposition',1996,'Luleå technical report TULEA 1996:18','urn:nbn:se:ltu:diva-24711',
     'https://www.diva-portal.org/smash/get/diva2:997763/FULLTEXT01.pdf','INSTITUTIONAL_FULL_TEXT',['Sec 3.2/3.3; Appendix A'],True,
     problem='LMMSE implementation cost',system_model='OFDM',estimation_object='channel',low_rank_object='full estimator',
     complexity_strategy='SVD truncation',theoretical_analysis='rank/error construction',closest_overlap_with_us='weighted low-rank linear estimation',
     key_difference='We retain full Source and truncate corrections',uncertainty='DOI/arXiv N/A; no DOI/arXiv verified in sources checked; URN exception applies')
work('mei2021','LMMSE channel estimation for OFDM systems with channel correlation function selection',2021,'IET Communications','10.1049/cmu2.12250',
     'https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cmu2.12250','PUBLISHER_INDEXED_SECTIONS',['Sec 3–5; Algorithm 1; accepted T20D read record'],True,
     problem='correlation mismatch',system_model='OFDM',pilot_setting='comb/block variants',adaptation_object='candidate CCF selection',
     offline_information='candidate correlations',online_information='pilot observations',coefficient_adaptation_mechanism='interpolation-error selection',
     closest_overlap_with_us='pilot-based selection among statistical estimators',key_difference='selection versus unrestricted two-coordinate combination',
     uncertainty='Direct full-page request now 403; exact 2D FDM port and code license UNKNOWN')
work('gui2014','Affine Combination of Two Adaptive Sparse Filters for Estimating Large Scale MIMO Channels',2014,'arXiv author preprint','1407.6083v1',
     'https://arxiv.org/pdf/1407.6083v1','ARXIV_FULL_TEXT',['adaptive combination formulation; experimental section'],True,
     problem='sparse CE',system_model='large-scale MIMO',adaptation_object='sparse adaptive filters and mixing coefficient',
     coefficient_adaptation_mechanism='LMS-type adaptation',closest_overlap_with_us='affine CE combination',key_difference='online filter adaptation versus frozen cross-pilot operators',
     uncertainty='Separate 2015 journal DOI 10.1049/iet-com.2014.0979 is not counted as another work; version details not merged')
work('ammse','Learning MMSE Filters for OFDM Channel Estimation: Attention Transformer Gains at Linear Inference',2026,'arXiv v5 (initial 2025)','2506.00452v5',
     'https://arxiv.org/html/2506.00452v5','ARXIV_FULL_TEXT',['Sec II–VI; v1 Sec VI separately; code main/data/AMMSE'],True,
     problem='learn linear MMSE filters',system_model='single-antenna OFDM',pilot_setting='72×14 grid; DMRS',estimation_object='full-grid channel',
     adaptation_object='offline learned full filter',offline_information='true channels',online_information='pilots and fixed exported filter',
     coefficient_adaptation_mechanism='attention training; fixed population filter at deployment',low_rank_object='full filter RA extension',
     complexity_strategy='linear inference; learned rank factors',distribution_shift_evaluation='SNR mismatch; v1 TDL and v5 COST2100 distinguished',
     theoretical_analysis='linear/rank structure',closest_overlap_with_us='learned linear operators and compressed inference',key_difference='whole-filter training versus pilot-CV correction coordinates',
     uncertainty='Code default global-token mode; online plots do not verify reproducible online-update implementation')
WORKS[-1]['target clean CSI required?']='YES for inspected offline training'
WORKS[-1]['target labels required?']='YES if training for that target distribution; no labels at fixed inference'
WORKS[-1]['code available?']='author repo fa2647f7fd0411bdb6a33ad9705cadb134ab3793; license unverified'
work('channelformer','Channelformer: Attention Based Neural Solution for Wireless Channel Estimation and Effective Online Training',2023,'IEEE TWC','10.1109/TWC.2023.3244484; 2302.04368v1',
     'https://arxiv.org/html/2302.04368v1','ARXIV_FULL_TEXT',['Sec II; IV C; V; VI D'],True,
     problem='receiver-side adaptive CE',system_model='SISO OFDM',pilot_setting='custom double-symbol DMRS online',adaptation_object='neural weights',
     offline_information='clean training channels',online_information='feature/denoised or power-boosted label pilots',
     coefficient_adaptation_mechanism='gradient updates',low_rank_object='UNKNOWN; weight pruning is not correction rank',complexity_strategy='weight pruning',
     distribution_shift_evaluation='dynamic online adaptation',closest_overlap_with_us='receiver-pilot adaptation',key_difference='special label-pilot information and gradient budget',
     uncertainty='Version comparison tied to v1, not presumed identical journal final')
WORKS[-1]['target clean CSI required?']='NO for inspected online label construction; YES offline pretraining'
WORKS[-1]['target labels required?']='receiver-constructed labels'
WORKS[-1]['code available?']='author MATLAB repo 6312c79e103422178ab4cc3d698aeae0731c308f; GPL-2.0'
work('peft','Two-Tier PEFT Adapter Switching for Environment- and SNR-Adaptive Channel Estimation',2026,'IEEE OJCOMS','10.1109/OJCOMS.2026.3697928',
     'https://www.researchgate.net/publication/405423813_Two-Tier_PEFT_Adapter_Switching_for_Environment-_and_SNR-Adaptive_Channel_Estimation','AUTHOR_FULL_TEXT',['Sec III F equation 7; IV; V A; Algorithms 1–2; storage table'],True,
     problem='environment/SNR adaptation',system_model='NYUSim RMa→InF SISO',pilot_setting='first OFDM symbol comb-6',adaptation_object='neural base and adapter library',
     offline_information='source and target supervised pairs',online_information='pilots; estimated SNR; selected environment base',
     coefficient_adaptation_mechanism='environment training and nearest-SNR switching',low_rank_object='neural LoRA weight updates',
     complexity_strategy='merge adapter into base',distribution_shift_evaluation='RMa→InF',closest_overlap_with_us='frozen base and compact adaptation library',
     key_difference='supervised target preparation and switching, not analytical affine coordinates',uncertainty='label-free environment-base selection; code/runtime reproduction UNKNOWN')
WORKS[-1]['target clean CSI required?']='YES in target training objective Eq 7'
WORKS[-1]['target labels required?']='YES; 500 target samples per adapter in read version'
work('mfa','Low-Rank Structured MMSE Channel Estimation with Mixtures of Factor Analyzers',2023,'Asilomar / arXiv author version','2304.14809v2',
     'https://arxiv.org/pdf/2304.14809v2','ARXIV_FULL_TEXT',['Sec II–V; equations 9–12; code README/pyproject'],True,
     problem='compact learned channel prior',system_model='measured 64-antenna channel vectors',pilot_setting='noisy full vector in inspected experiment',
     adaptation_object='posterior component weights',offline_information='channel corpus; EM prior fitting',online_information='noisy observation',
     coefficient_adaptation_mechanism='posterior convex mixture of conditional LMMSE',low_rank_object='component covariances',complexity_strategy='matrix inversion lemma / precomputed filters',
     theoretical_analysis='conditional estimator; asymptotic prior approximation',closest_overlap_with_us='compact statistical filters and observation-driven mixture',
     key_difference='covariance factors/posterior simplex, not correction-only affine extrapolation',uncertainty='MIMO time-frequency sampling port not a reproduced paper configuration')
WORKS[-1]['code available?']='author cplx-mfa 9f9c22a7775076fbedd0fb73929c195b0dd2aa6b; BSD-3-Clause; partial CE implementation'
work('pbce','On the Asymptotic MSE-Optimality of Parametric Bayesian Channel Estimation in mmWave Systems',2025,'arXiv v3','2504.16710v3',
     'https://arxiv.org/html/2504.16710v3','ARXIV_FULL_TEXT',['parametric LMMSE construction; asymptotic analysis; simulation'],True,
     problem='parametric Bayesian CE',system_model='mmWave',adaptation_object='channel parameters / conditional LMMSE',online_information='noisy observations',
     coefficient_adaptation_mechanism='parameter inference',theoretical_analysis='high-SNR/long-coherence optimality',closest_overlap_with_us='low-dimensional statistical adaptation',
     key_difference='physical parameter recovery versus output-difference coordinates',uncertainty='joint 2×2 FDM and code availability UNKNOWN')
work('turan','Channel-Adaptive Pilot Design for FDD-MIMO Systems Utilizing Gaussian Mixture Models',2024,'arXiv v1','2403.17577v1',
     'https://arxiv.org/pdf/2403.17577v1','ARXIV_FULL_TEXT',['GMM conditional estimator; pilot design formulation'],True,
     problem='pilot design/CE',system_model='FDD MIMO',adaptation_object='pilot codebook and posterior estimates',offline_information='learned GMM prior',
     online_information='received pilots',coefficient_adaptation_mechanism='posterior weighted conditional filters',low_rank_object='UNKNOWN',
     closest_overlap_with_us='observation-driven estimator combination',key_difference='joint pilot design rather than fixed FDM geometry',uncertainty='equal-budget OOD and target-label policy UNKNOWN')
work('semiblind','Semi-Blind Strategies for MMSE Channel Estimation Utilizing Generative Priors',2025,'arXiv v2','2504.17573v2',
     'https://arxiv.org/html/2504.17573v2','ARXIV_FULL_TEXT',['subspace preprocessing; GMM/VAE formulation'],False,
     problem='semi-blind CE',system_model='massive MIMO',online_information='pilots AND payload',adaptation_object='received-signal subspace',
     low_rank_object='channel subspace',theoretical_analysis='Rayleigh MSE comparison',closest_overlap_with_us='subspace-informed MMSE',
     key_difference='payload information is unavailable to our online estimator',uncertainty='equal sparse-pilot budget not established')
work('ridge','Attention-Aided MMSE with Ridge Denoising: How to Train under Noisy Channel Samples',2026,'arXiv v1','2608.15945v1',
     'https://arxiv.org/html/2608.15945v1','ARXIV_FULL_TEXT',['Sec II–V; same-noise bias; fixed-filter stylization'],True,
     problem='biased noisy-label training',system_model='OFDM/COST2100',offline_information='noisy full-grid sounding; known noise variance',
     online_information='sparse pilots',adaptation_object='learned filter / training surrogate',coefficient_adaptation_mechanism='ridge and covariance shrinkage',
     theoretical_analysis='stylized scalar shrinkage',closest_overlap_with_us='statistical linear filtering without full clean online CSI',
     key_difference='training-corpus acquisition problem, not frozen-coordinate adaptation',uncertainty='Paper specifies input-dependent filter; do not conflate with v5 fixed export')
WORKS[-1]['target clean CSI required?']='NO in stated noisy-training setting'
work('sirius','Unsupervised Online Channel Estimation for High-Mobility OFDM via Implicit Neural Representation',2026,'arXiv v1','2605.10213v1',
     'https://arxiv.org/html/2605.10213v1','ARXIV_FULL_TEXT',['Sec II–IV; Algorithm 1'],False,
     problem='ICI-aware online CE',system_model='high-mobility OFDM',online_information='received symbols, pilots and decision-directed pseudo-pilots',
     adaptation_object='slot-specific neural representation',coefficient_adaptation_mechanism='iterative reconstruction loss',
     closest_overlap_with_us='sparse-pilot adaptation',key_difference='payload decisions, ICI model, online gradients',uncertainty='MIMO port/code/runtime fairness UNKNOWN')
WORKS[-1]['target clean CSI required?']='NO in inspected online objective'
work('lorat','LoRAT: Low Rank Adaptation and Transfer for Multi-environment Channel Estimations',2025,'IEEE GLOBECOM','10.1109/GLOBECOM59602.2025.11431847',
     'https://xubolee.github.io/files/papers/GLOBECOM25_LoRAT.pdf','AUTHOR_FULL_TEXT',['adaptation/training formulation; experiments'],False,
     problem='multi-environment neural transfer',adaptation_object='attention projection weights',low_rank_object='neural weight updates',
     closest_overlap_with_us='low-dimensional adaptation',key_difference='learned network parameters versus fixed Wiener corrections',
     uncertainty='pilot-only CV/label-free target interface and reusable code UNKNOWN')
work('li2024','Parametric Channel Estimation for RIS-Assisted mmWave MIMO-OFDM Systems with Low Pilot Overhead',2024,'Signal Processing 225','10.1016/j.sigpro.2024.109611',
     'https://eprints.bournemouth.ac.uk/40155/1/SP_Final_Version.pdf','INSTITUTIONAL_FULL_TEXT',['Sec 2–4; pilot uniqueness; Table 2'],False,
     problem='low-pilot parametric CE',system_model='RIS mmWave MIMO-OFDM',pilot_setting='two constrained RIS training schemes',
     estimation_object='angles, delays, gains',coefficient_adaptation_mechanism='constrained tensor decomposition',
     complexity_strategy='algebraic recovery',pilot_overhead_study='minimum overhead and simulation',theoretical_analysis='uniqueness/training complexity',
     closest_overlap_with_us='low-pilot evidence and complexity',key_difference='different physical observation model',uncertainty='not a directly comparable adaptive operator baseline')
work('wan2025','Sparse channel estimation for underwater acoustic OFDM systems with super-nested pilot design',2025,'Signal Processing 227, Short communication','10.1016/j.sigpro.2024.109709',
     'https://mel2.xmu.edu.cn/member/upload_paper/202409161945280617.pdf','AUTHOR_FULL_TEXT',['Sec 2–5; pilot design; simulation/pool/sea experiments'],False,
     problem='large-delay sparse CE',system_model='underwater OFDM',pilot_setting='super-nested pilots',
     adaptation_object='covariance refinement',coefficient_adaptation_mechanism='cross-correlation cancellation and SPICE',
     pilot_overhead_study='limited-pilot delay coverage',closest_overlap_with_us='covariance/pilot-information limits',
     key_difference='pilot redesign and sparse delay model',uncertainty='2025 volume, 2024 online/DOI year; no 2×2 CDL generalization evidence')
work('csgmm','Sparse Bayesian Generative Modeling for Joint Parameter and Channel Estimation',2025,'IEEE SSP / arXiv author version','2502.18369v1; 10.1109/SSP64130.2025.11073327',
     'https://arxiv.org/html/2502.18369v1','ARXIV_FULL_TEXT',['Sec II–V; equations 2–11; online complexity'],True,
     problem='joint channel/parameter estimation',system_model='spatial ULA channel',adaptation_object='posterior latent weights',
     offline_information='noisy compressed samples; physical dictionary',online_information='measurement',coefficient_adaptation_mechanism='CSGMM posterior',
     low_rank_object='dictionary covariance representation',complexity_strategy='structured models',closest_overlap_with_us='data-driven statistical combination',
     key_difference='dictionary/prior learning, not prepared two-direction Wiener library',uncertainty='wideband 2×2 CV adaptation is not evaluated here')
work('pilotcov2021','Impact of Pilot Overhead and Channel Estimation on the Performance of Massive MIMO',2021,'IEEE TCOM','10.1109/TCOMM.2021.3112213',
     'https://research.aalto.fi/en/publications/impact-of-pilot-overhead-and-channel-estimation-on-the-performanc/','ABSTRACT_ONLY',['institutional abstract'],False,
     problem='covariance-estimation pilot cost',system_model='TDD massive MIMO',pilot_overhead_study='UL/DL SE versus added covariance pilots',
     theoretical_analysis='closed-form SE with estimated covariances',closest_overlap_with_us='offline-statistics cost cannot be ignored',
     key_difference='SE/training trade-off versus affine coordinate fitting',uncertainty='Full derivations not read; pilot arrangement/details UNKNOWN')
work('nlms2022','On the behavior of a combination of adaptive filters operating with the NLMS algorithm in a nonstationary environment',2022,'Signal Processing 196','10.1016/j.sigpro.2022.108465',
     'https://www.sciencedirect.com/science/article/abs/pii/S0165168422000123','ABSTRACT_ONLY',['publisher abstract/introduction preview only'],True,
     problem='adaptive-filter combination dynamics',system_model='nonstationary adaptive filtering',adaptation_object='combiner and filters',
     theoretical_analysis='transient/steady-state stochastic model claimed by abstract',closest_overlap_with_us='coefficient/error analysis is established research',
     key_difference='specific CE cross-fold dependence remains unverified',uncertainty='Exact assumptions/theorems UNKNOWN; not borrowed as a guarantee')

def dump(path,data):
    (ROOT/path).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def md(name,body):
    (ROOT/'docs'/('sp_u1_'+name+'.md')).write_text(dedent(body).strip()+'\n',encoding='utf-8')

def main():
    dump('outputs/sp_u1_literature_database.json',{'date':'2026-10-04','fields':FIELDS,'works':WORKS,
        'manuscript_bib_changed':False,'review_type':'focused reproducible search; NOT a systematic review'})
    lines=['# Closest-work matrix','', 'Each table is one work. UNKNOWN is intentional. Source versions and locators govern every technical comparison. Multiple versions are not counted as multiple works. Historical T20D/T20G records remain frozen.', '']
    for r in WORKS:
        lines += [f"## {r['id']} — {r['title']}",'',f"[Read source]({r['source_url']}); inspected locators: {', '.join(r['read_locators'])}.",'','| Field | Audited value |','|---|---|']
        lines += [f"| {f} | {str(r[f]).replace('|','/')} |" for f in FIELDS]
        lines += ['']
    md('closest_work_matrix','\n'.join(lines))
    build_documents()
    print(json.dumps({'works':len(WORKS),'recent_2023_2026':sum(2023<=r['year']<=2026 for r in WORKS),
        'high_core_proximity':sum(r['high_core_proximity'] for r in WORKS),'scientific_runs':0}))

def build_documents():
    md('signal_processing_venue_profile',r'''
    # Signal Processing venue profile

    Access date 2026-10-04. Fit assessment is an engineering judgment, not an acceptance prediction.

    | Item | Official evidence | Status |
    |---|---|---|
    | Scope | [Elsevier journal shop](https://shop.elsevier.com/journals/signal-processing/0165-1684) includes statistical processing, estimation, filtering and communications | VERIFIED |
    | Article types | Original research, review, Fast Communications on that page | VERIFIED |
    | Length | Research ≤30 pages, reviews usually 30–60, Fast Communications ≤10; single column, double spaced, figures/tables/references included | VERIFIED on shop page; recheck full guide before submission |
    | Current full author guide | [ScienceDirect guide](https://www.sciencedirect.com/journal/signal-processing/publish/guide-for-authors) returned 403 | UNKNOWN remaining requirements |
    | Aims-and-scope page | [Official page](https://www.sciencedirect.com/journal/signal-processing/about/aims-and-scope) returned 403 | shop scope used |
    | Data/code rule | [Elsevier data-statement guidance](https://www.elsevier.com/researcher/author/tools-and-resources/research-data/data-statement) recommends checking each journal's policy | Journal-specific requirement UNKNOWN |

    Do not infer mandatory open code, a page limit in a two-column template, abstract length, highlights or submission checklist from another journal's guide. Preserve reproducibility artifacts and prepare an accurate availability statement later. No submission format migration this round.

    ## Core problem (90 words or fewer)

    A frozen LMMSE channel-estimation operator can become mismatched when propagation statistics change. We ask whether a compact space of prepared operator corrections can be adapted from sparse received pilots without reconstructing the full target covariance. The study separates representation capacity, pilot-driven coordinate inference, and correction-compression error under a fixed information budget. Its intended contribution is an adaptation-specific statistical analysis and bounded evidence of when this procedure succeeds or fails, together with explicit offline preparation cost. It is positioned as statistical signal processing and adaptive linear channel estimation for communications.

    ## Evidence needed for this venue

    Highest-value gaps are analysis of the actual stochastic pilot-CV design, strong comparable statistical/learning baselines, preparation-sample sensitivity and a real pilot-density trade-off. Additional model size or network architecture is not a venue-fit argument. Official scope does not require all recent-paper evidence patterns below.
    ''')
    queries=json.loads((ROOT/'outputs/sp_u1_search_queries.json').read_text(encoding='utf-8'))
    rows=['# Literature search log','','Focused, reproducible search; **not a systematic review**. Date 2026-10-04. Horizon 2021–2026, emphasis 2023–2026; indispensable classics retained. Discovery uses web search, then primary publisher/arXiv/author/institutional records. No institutional database subscription or academic MCP was available. Search engine totals were not exposed, so result count is UNKNOWN, not zero.','',
          '| ID | Query | Source/date | Total result count |','|---|---|---|---|']
    rows += [f"| Q{i+1:02} | {q['query'].replace('|','/')} | web discovery / {q['date']} | UNKNOWN |" for i,q in enumerate(queries)]
    rows += ['', '## Screening disposition', '', '| Record | Decision | Reason |','|---|---|---|']
    rows += [f"| {r['id']} | included | {r['closest overlap with us']} |" for r in WORKS]
    rows += [
        '| Joseph 2025, SP 235 110020 | pattern-only survey | not an original CE benchmark; [institutional full text](https://pure.tudelft.nl/ws/portalfiles/portal/242295268/1-s2.0-S0165168425001343-main.pdf) |',
        '| SP 2026 tensor completion 110637 | pattern-only | primary abstract, CSI completion application; not affine adaptation |',
        '| SP 2024 speckle covariance 109590 | pattern-only | radar covariance, not OFDM adaptation |',
        '| SP December 2026 110724 | excluded | future issue beyond audit cutoff; not used for current pattern inference |',
        '| Kalman/VB adaptive tracking papers 2023/24 | secondary screening only | model/noise adaptation, different online observations |',
        '| Quantized/Dithered TCOM 2024 CE | excluded baseline | additional quantization/dither hardware interface |',
        '| LiDAR channel prediction 2025 | excluded baseline | side information and prediction task |',
        '| CSGMM paper platform AI summaries | excluded source | primary arXiv used instead |',
        '| Reddit estimator combination | excluded source | not primary technical evidence |',
        '| General transceiver surveys | excluded core set | discovery only; not method equivalence evidence |',
        '| Dissertation referring to Özdemir | excluded technical substitute | not the journal algorithm |',
        '| Generic mmWave/OTFS/beamforming/medical affine papers | excluded core set | interfaces differ or relevance insufficient |',
        '| Other Elsevier journal guides | excluded venue source | rules for a different journal |',
        '| Gui journal/preprint and A-MMSE revisions | grouped, not duplicate counted | version differences preserved |',
        '', 'Screened catalogue = the 20 included unique works plus 14 named screening dispositions above. This is a recorded selected-work catalogue, not a claim that all engine hits were screened. Repeat exact queries to inspect engine ranking; ranking can change.',
        '', '## Retrieval and legacy closure',
        '', 'IEEE record 1705928 returned JavaScript/robot verification. DOI, author/institution, exact-title/PDF searches and the linked public ResearchGate publication were checked; the publication offers request-only full text and an inspectable abstract. No email/request was sent. A guessed ResearchGate URL failed and is not evidence. DBLP authors differ from the frozen three-author Crossref record; metadata needs publisher reconciliation before final citation correction, not a historical-record rewrite.',
        '', 'Özdemir now has ABSTRACT_ONLY evidence in this new audit. Known comb pilots, generalized low-rank LMMSE, pilot/non-pilot subspace relation and tracking are public abstract facts. No rank/update equations or statistical assumptions are filled. Historical technical_closure.json is byte-unchanged.',
        '', 'Direct ScienceDirect guide/articles and Wiley full page returned 403. Public institutional PDFs succeeded for Li 2024 and Wan 2025; publisher-indexed abstracts remain explicitly limited. GitHub API initially unavailable through web, then public read-only HTTP succeeded; three repo commits/licenses were recorded. No author code/data/model executed.',
        '', '## Theme coverage',
        '', 'Q12–15 cover LMMSE/covariance adaptation and tracking; Q20–25 and the classic records cover affine/mixture/LS theory; Q16–19 cover operator and pilot-only/self-supervised/model-based/environment/domain/PEFT themes. Exact-title retrieval covers A-MMSE, Channelformer, PEFT, MFA, low-pilot OFDM, MIMO-OFDM and recent Signal Processing. Coverage is focused, not exhaustive. Author copy and backward references supplement keyword searches.'
    ]
    md('literature_search_log','\n'.join(rows))
    md('signal_processing_recent_paper_patterns',r'''
    # Recent Signal Processing paper patterns

    Sample covers 2024–2026, with 2022/2023 used only as theory context. Five deliberately selected items do not establish journal-wide frequencies or mandatory editorial rules.

    | Work / actual access | Theory | Experiment/baseline/complexity | Model/data/pilots | Gap relevant here |
    |---|---|---|---|---|
    | Li et al. 2024, [109611 accepted full text](https://eprints.bournemouth.ac.uk/40155/1/SP_Final_Version.pdf), Sec 3–4/Table 2 | training identifiability and overhead | comparisons to TRICE-CS/TenRice; complexity table | RIS parametric MIMO-OFDM, designed training | tie pilot requirements to identifiability |
    | Wan et al. 2025, [109709 author full text](https://mel2.xmu.edu.cn/member/upload_paper/202409161945280617.pdf), Sec 2–5 | delay coverage/virtual pilots | simulations, pool and sea measurements | underwater OFDM sparse delays | show overhead affects the inference object |
    | [2026 tensor-completion 110637](https://www.sciencedirect.com/science/article/abs/pii/S0165168426001519), ABSTRACT_ONLY publisher-indexed | detailed guarantees UNKNOWN | detailed baseline/timing UNKNOWN | abstract names sub-6 GHz CSI completion and DOA applications | low-rank compression alone is not a statistical insight |
    | [2024 speckle covariance 109590](https://www.sciencedirect.com/science/article/pii/S0165168424002093), ABSTRACT_ONLY publisher-indexed | structural constrained ML stated | detailed comparisons UNKNOWN | abstract states measured clutter spectral symmetry | connect covariance simplification to an explicit structure/assumption |
    | [Joseph et al. 2025 survey 110020](https://pure.tudelft.nl/ws/portalfiles/portal/242295268/1-s2.0-S0165168425001343-main.pdf), full first-page/type verified | broad estimation tools | not a primary algorithm comparison | low-resolution sensing/communications | excluded from inference about original-paper experiment requirements |

    Two fully inspected CE articles connect observation design with recovery, rather than merely add more channels. The two abstract-only items support only the listed limited observations. We cannot assert their exact theorem count, pilot overhead or baseline strength. The survey is context, not a third primary CE success example.

    Our upgrade should add a useful actual-algorithm conditioning analysis, preparation cost and equal-information baselines. Measurement data would add value if compatible, but it is not established as a journal requirement and would need a new data/interface audit. No full ICI system is implied by a high speed in a diagonal channel model.
    ''')
    risks=[
      {'question':'Q1','status':'FOUND_PARTIAL','evidence':['ls2010','gui2014','peft'],'finding':'Affine multi-filter coordinates and frozen compact libraries exist separately; exact complete frozen LMMSE/cross-pilot combination not verified.'},
      {'question':'Q2','status':'FOUND_PARTIAL','evidence':['ls2010','mei2021','channelformer'],'finding':'LS combiner adaptation and received-pilot supervision exist; our complete statistical FDM interface remains a scoped distinction.'},
      {'question':'Q3','status':'NOT_FOUND_IN_SEARCHED_WORKS','evidence':['edfors1996','ammse','mfa'],'finding':'No direct correction-only/full-Source match identified in this verified set. It is not a priority clearance; Özdemir full text unresolved.'},
      {'question':'Q4','status':'FOUND_PARTIAL','evidence':['ls2010','peft'],'finding':'Reference-relative differences and libraries are established. Profile/mobility prepared Wiener finite differences not verified as identical.'},
      {'question':'Q5','status':'FOUND_PARTIAL','evidence':['oz2006'],'finding':'Abstract directly describes tracking for low-rank MIMO LMMSE. Whether it achieves our sparse affine adaptation objective is UNKNOWN.'},
      {'question':'Q6','status':'FOUND_PARTIAL','evidence':['peft','mfa','turan','ammse'],'finding':'Environment adaptation and conditional linear mixtures exist; discrete/library and simplex constraints differ from real affine coordinates.'},
    ]
    dump('outputs/sp_u1_prior_art_risk.json',{'questions':risks,'priority_claim_cleared':False,
        'legacy_algorithm_equivalence_unknown':True,'affine_ls_concept_risk':'HIGH','generic_lowrank_risk':'HIGH'})
    text=['# Novelty collision gate','','Within the literature examined, generic affine combination, LS mixing, low-rank filtering and compact adapter libraries are established. No direct match was identified in the verified set for the complete information contract. Potential overlap remains, particularly in unread legacy implementations.','',
          '| Question | Status | Evidence IDs | Bounded interpretation |','|---|---|---|---|']
    text += [f"| {r['question']} | {r['status']} | {', '.join(r['evidence'])} | {r['finding']} |" for r in risks]
    text += ['', '## Internal contribution assessment', '', '| Component | Concept | Combination | Analysis | Empirical | Reason |','|---|---|---|---|---|---|',
      '| A Operator-space representation | LOW | MEDIUM | MEDIUM potential | MEDIUM | finite prepared statistical differences; no physical separability |',
      '| B Pilot-driven coordinates | LOW | MEDIUM | MEDIUM potential | MEDIUM | LS old; cross-pilot endogenous design is the useful analysis target |',
      '| C Affine extrapolation | LOW | LOW | LOW alone | MEDIUM | unconstrained weights and out-of-hull geometry are established |',
      '| D Correction-only rank | LOW | MEDIUM | LOW generic / MEDIUM if target error coupling derived | MEDIUM | weighted SVD old; full base plus compressed corrections is concrete |',
      '| E Offline/online separation | LOW | MEDIUM | LOW | MEDIUM | explicit clean-preparation and pilot-only deployment contract |',
      '', 'Scores are advisor heuristics, not bibliometric or experimental measurements. None justifies a priority superlative. Candidate contribution is integrated analysis plus bounded evidence, not five independent conceptual inventions.',
      '', 'Top five collision risks are multi-filter LS (ls2010), adaptive low-rank LMMSE tracking (oz2006), full-filter compression (edfors1996/ammse), observation-dependent statistical mixtures (mfa/turan), and compact environment libraries (peft). Metadata-only comparisons cannot establish a sharp difference.' ]
    md('novelty_collision_gate','\n'.join(text))
    from sp_u1_plans import write_plans
    write_plans(ROOT,md,dump)

if __name__=='__main__':main()
