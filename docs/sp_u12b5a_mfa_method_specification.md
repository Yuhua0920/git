# Evidence-first MFA method specification

Created before the adapter implementation.

1. observation [BOTH]
Paper y=h+n after decorrelation; author supports y=A h+n with arbitrary known A. Project uses 64x4032 owner-pilot selector.
Paper: Sec II Eq (1); Code: reference/sp_u12b5a/paper_period/MFA_estimator/MFA_estimator.py:32. 

2. channel model [BOTH]
h|k = mu_k + W_k z + v_k; z proper CN(0,I), v_k proper CN(0,Psi_k).
Paper: Sec III Eqs (3)-(6); Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:202. 

3. mixture probability [BOTH]
Categorical pi_k >=0, sum pi_k=1; marginal proper-complex Gaussian mixture.
Paper: Eqs (4)-(8); Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:134. 

4. factor analyzer [BOTH]
Every component is low-rank loading plus positive residual covariance.
Paper: Eqs (3),(6); Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:202. 

5. loading shape [BOTH]
W_k is D x L; array K x D x L, complex128.
Paper: Sec III Eq (3); Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:122. 

6. latent dimension [BOTH]
L = number of loading columns; latent_dim; not rank(C_k) or empirical effective rank.
Paper: Sec III L<N; Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:54. 

7. component means [BOTH]
Learned, not constrained to zero; no Source centering supplied to MFA.
Paper: Eq (3); Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:182. 

8. residual covariance [BOTH]
Paper numerical setting: separate isotropic Psi_k = psi_k^2 I. Freeze PPCA=True, lock_psis=False. Initial author diagonal var is isotropic only after first M-step.
Paper: Sec V numerical setting; Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:188. 

9. weights [BOTH]
Positive random normalized initialization; EM updates Nk/N. No post hoc clipping, pruning or weight floor.
Paper: Eq (8); Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:190. 

10. responsibilities [BOTH]
Proper-complex log density with -q log(pi)-logdet-quadratic; log-sum-exp over components. Training rs_clip=0.0; inference no floor.
Paper: Eq (12); Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:217. 

11. parameter learning [AUTHOR_CODE]
Author sequential mean/loading/residual updates from responsibility-weighted latent moments.
Paper: Sec III refers to EM but gives no update formulas; Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:162. 

12. EM [BOTH]
EM-style algorithm exactly as original author; no substituted textbook updates.
Paper: Sec III, citation [11]; Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:137. 

13. E step [AUTHOR_CODE]
r_kn; beta=W^H C^-1; posterior z=beta(x-mu); Ezz^H=I-beta W+zz^H.
Paper: EM update details UNSPECIFIED; Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:69. 

14. M step [AUTHOR_CODE]
mu_new=sum r(x-W_old z)/Nk; W_new=sum r(x-mu_new)z^H [sum r Ezz^H]^-1; residual uses W_old z, then author floor/isotropization; pi=Nk/N.
Paper: EM update details UNSPECIFIED; Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:181. Do not replace old Wz in residual with new Wz; do not assert strict likelihood monotonicity.

15. constraints [AUTHOR_CODE]
Psi entries clipped to 1e-6 at M-step before isotropic averaging. No clipping at initialization in 2023; no responsibility floor at rs_clip=0.0.
Paper: Regularization details UNSPECIFIED; Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:187. 

16. stopping [AUTHOR_CODE]
After i>5, abs((newL-L)/newL)<1e-4; pre-update log likelihood stored each iteration.
Paper: UNSPECIFIED; Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:148. 

17. max iterations [AUTHOR_CODE]
100, original constructor default; official examples sometimes 400, not a paper mandate.
Paper: UNSPECIFIED; Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:47. 

18. initialization [AUTHOR_CODE]
KMeans on [Re(x),Im(x)] with n_init=1, k-means++; tiny proper-CN random loadings divided by sqrt(1e6); featurewise complex np.var(ddof=0); normalized random weights.
Paper: UNSPECIFIED; Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:116. 

19. restarts [AUTHOR_CODE]
One model fit; no outer restart loop or best-run selection. KMeans n_init=1 is not an EM restart.
Paper: UNSPECIFIED; Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:101. 

20. original model selection [UNSPECIFIED]
Paper displays K/L comparisons, but no target-blind automated selection/split protocol is specified. Project future 400/100 trajectory split is mentor protocol, not reproduced paper recipe.
Paper: Sec V Figs 2-5; Code: None. 

21. inference [BOTH]
Sum all K noisy-observation responsibilities times conditional LMMSE component means. Pass n_summands_or_proba=1.0 float explicitly; integer default 1 is hard selection.
Paper: Eqs (10)-(12); Code: reference/sp_u12b5a/paper_period/MFA_estimator/MFA_estimator.py:50. 

22. noise [BOTH]
Known complex noise covariance Cn=sigma2 I after division by unit-modulus owner pilots; sigma2 source referenced, no target normalization.
Paper: Secs II/IV Eq (1); Code: reference/sp_u12b5a/paper_period/MFA_estimator/MFA_estimator.py:74. 

23. online clean CSI [BOTH]
No; estimate(y,Cn,A) uses frozen learned model and received pilots only.
Paper: Sec IV; Code: reference/sp_u12b5a/paper_period/MFA_estimator/MFA_estimator.py:6. 

24. offline clean CSI [BOTH]
Yes; model trained from noiseless channel vectors, not received observations.
Paper: Sec II channel dataset H; Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:91. 

25. complex representation [BOTH]
Native proper-complex statistics; training/inference complex128. Real expansion used solely for KMeans means.
Paper: Eqs (2)-(8); Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:97. 

26. real expansion [AUTHOR_CODE]
concat(Re,Im) without sqrt(2), split half then Re+i Im; not a doubled real Gaussian training model.
Paper: Not used as model representation; Code: reference/sp_u12b5a/paper_period/cplx-mfa/utils.py:4. 

27. normalization centering [BOTH]
Paper measurements globally normalized E||h||^2=D; author fit performs no scaling/precentering. Project retains physical units and learned component means.
Paper: Sec II measured data normalization; Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:91. Fixed-unit adaptation, not a numerical replication of paper figures; fixed 1e-6 author floor retained in these units.

28. inverse normalization [AUTHOR_CODE]
None in author code or project; output remains in same physical units as clean training data.
Paper: Paper global experimental normalization convention; Code: reference/sp_u12b5a/paper_period/MFA_estimator/MFA_estimator.py:67. 

29. numerical linear algebra [AUTHOR_CODE]
Author scipy.linalg.inv for latent moment; numpy pinv for Woodbury inner and noisy observation covariance; Psi floor 1e-6. Preserve all; no added ridge/fallback.
Paper: Sec IV Woodbury, Eq (15); Code: reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:292. 

30. paper/code differences [BOTH]
Paper identity observation special case, author general A; paper isotropic numerical setting selected by PPCA flag; constructor hard-selection default explicitly overridden by documented float full-mixture API. Paper EM/init details unspecified; original code is authoritative.
Paper: Secs II-V; Code: reference/sp_u12b5a/paper_period/MFA_estimator/MFA_estimator.py:10. 

## Frozen initialization/restart policy

{
  "status": "FROZEN",
  "reference": "2023 paper-period original author commits",
  "PPCA": true,
  "lock_psis": false,
  "rs_clip": 0.0,
  "max_condition_number": 1000000.0,
  "maxiter": 100,
  "tol": 0.0001,
  "verbose": false,
  "outer_model_fits": 1,
  "outer_restarts": 0,
  "KMeans_n_init": 1,
  "KMeans_init": "k-means++",
  "KMeans_seed": "Recorded NumPy legacy MT19937 initialization seed (random_state=None follows existing seeded global state)",
  "loading_and_weight_seed": "Same recorded initialization seed, consumed sequentially after KMeans",
  "seed_control": "Temporarily save global RandomState, seed it once, execute original author fit, restore state; no unrecorded global randomness",
  "training_seed": "Same as initialization seed: EM has no additional stochastic draws",
  "restart_seed": null,
  "best_run_selection": "NOT_APPLICABLE",
  "future_formal_seeds": "Must be allocated and frozen by future B5B before training; toy seeds never select policy",
  "stopping": "i > 5 and abs((newL - previousL)/newL) < 1e-4",
  "maxiter_exhaustion": "Original author warning retained; future formal convergence gate STOP, no rescue",
  "source": "reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:47",
  "basis": "AUTHOR_CODE defaults; PPCA specialization from PAPER Sec V"
}

## Future RNG registration (design only)

{
  "namespace": "SP_U12B5A_MFA_AUTHOR2023_V1",
  "derivation": "uint32 big endian first four bytes of SHA256(namespace|role|K=K|L=L|fit=0)",
  "roles": [
    "candidate",
    "full_retraining"
  ],
  "registered": [
    {
      "K": 1,
      "loading_rank": 4,
      "candidate_training_initialization_seed": 375404078,
      "full_retraining_initialization_seed": 974208452
    },
    {
      "K": 1,
      "loading_rank": 8,
      "candidate_training_initialization_seed": 2201718927,
      "full_retraining_initialization_seed": 908565316
    },
    {
      "K": 1,
      "loading_rank": 16,
      "candidate_training_initialization_seed": 1485464306,
      "full_retraining_initialization_seed": 4143049057
    },
    {
      "K": 2,
      "loading_rank": 4,
      "candidate_training_initialization_seed": 1443235906,
      "full_retraining_initialization_seed": 2675241376
    },
    {
      "K": 2,
      "loading_rank": 8,
      "candidate_training_initialization_seed": 1625978320,
      "full_retraining_initialization_seed": 41481437
    },
    {
      "K": 2,
      "loading_rank": 16,
      "candidate_training_initialization_seed": 3827983190,
      "full_retraining_initialization_seed": 3273188115
    },
    {
      "K": 4,
      "loading_rank": 4,
      "candidate_training_initialization_seed": 1714107369,
      "full_retraining_initialization_seed": 2802962356
    },
    {
      "K": 4,
      "loading_rank": 8,
      "candidate_training_initialization_seed": 80346791,
      "full_retraining_initialization_seed": 1558626682
    },
    {
      "K": 4,
      "loading_rank": 16,
      "candidate_training_initialization_seed": 1483984871,
      "full_retraining_initialization_seed": 3252496116
    }
  ],
  "generated_without_training": true
}

These seed registrations do not select K or loading rank and do not execute training. Full retraining uses the selected configuration's preassigned independent role seed.
