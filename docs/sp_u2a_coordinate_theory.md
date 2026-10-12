# SP-U2A first-layer coordinate theory: raw derivations

Controlling inventory: `docs/sp_u2a_equation_inventory.md`, E01–E11, accepted HEAD ca83982. These are algorithm-specific explanatory identities and deterministic bounds. They neither establish a new general theorem nor change the frozen method. All vectors below are in column notation; real coefficients are essential.

## 1. Coordinate oracles and risk domains

### O01. Empirical heldout-pilot optimum

\[
\widehat c_{p,j}\in\arg\min_{c\in\mathbb R^2}\|z_j-G_jc\|^2.
\]

This is exactly the deployed fit, uniquely defined when rank_R(G)=2. It uses observed noisy heldout pilots. Comparing the deployed fit to this same oracle gives **zero fitting error in exact arithmetic**; it does not measure generalization or channel accuracy. A rank-deficient problem has a set of minimizers, while the unchanged implementation attempts solve and can fail.

### O02. Population heldout-pilot optimum

Fix a target law of the five-frame trajectory and the frozen layout/library/noise reference. Define

\[
J_p(c)=\mathbb E\|z_j-G_jc\|^2,
\quad H_p=\mathbb E(G_j^TG_j),\quad b_p=\mathbb E(G_j^Tz_j),
\quad c_p^*=H_p^{-1}b_p.\tag{O02}
\]

Finite second moments and H_p positive definite are assumptions, not consequences of 640 rows. This optimum is deterministic for a fixed target law, but unavailable online. Nonstationarity requires specifying j or the window law; a single law is not assumed to describe all mobility trajectories. It remains the optimum of the **noisy heldout loss**, not clean heldout-pilot MSE or full channel MSE.

### O03. Channel-MSE optimum, dense and deployed rank4

For a=full-dense correction or rank4, let

\[
a_s(u_j)=\mu_{d,s}+W_su_j,\qquad
X_a(u_j)=[B_{P,a}u_j,B_{M,a}u_j]\in\mathbb C^{3968\times2}.
\]

Using every data-domain true channel coefficient rather than only heldout pilots,

\[
J_{d,a}(c)=\mathbb E\|h_{d,j}-a_s(u_j)-X_a(u_j)c\|^2,
\quad H_{d,a}=\mathbb E\Re(X_a^HX_a),
\quad b_{d,a}=\mathbb E\Re[X_a^H(h_{d,j}-a_s)],
\quad c_{d,a}^*=H_{d,a}^{-1}b_{d,a}.\tag{O03}
\]

This mean-aware population MSE oracle is evaluation/theory only and requires the true joint target law. A per-realization oracle minimizes the same norm for one clean h and noisy u; it is random and truth-assisted, not O03. Neither oracle is a deployed input. Dense and rank4 oracles are different objectives; replacing the dense coordinates by a rank4-optimized oracle would change the existing compression comparison. This task does not do so.

**Full-grid wording:** the deployed API estimates 3968 data-domain entries, not all 4096. If a literal full-grid MSE is intended, define a fixed extension \(\widetilde h_a(c)\) equal to the data estimator on those entries and some c-independent estimator on the other 128. Then the full-grid squared loss is data loss plus a c-independent term, and its oracle is O03. If the extension also depends on c, its extra rows must enter H and b; equality is not assured. No full-grid extension has been implemented or presumed here. O03 is the actual output-domain channel oracle needed for propagation.

### Pilot/channel optimum alignment is a condition

When both Hessians are positive definite,

\[
c_p^*=c_{d,a}^*\iff b_{d,a}=H_{d,a}H_p^{-1}b_p.\tag{O04}
\]

Proof: substitute the two unique normal-equation solutions. A sufficient condition is H_d=tH_p and b_d=t b_p for the same positive t; matching coefficients by this condition is not a claim about the current CDL law. The fixed-origin centered filter oracle at `src/frozen_affine_validation.py:129–150` minimizes \(\|(W_t-W_s-\sum c_mB_m)L_t\|_F^2\). Its docstring explicitly excludes mean-bias optimization. It equals O03 only with appropriate centering/mean alignment; do not rename it full-channel MSE oracle.

### Same pilot law can support different data oracles

Take any pilot law with finite moments and identifiable pilot H_p. Keep that law and every frozen pilot operator fixed. Set data h in two joint laws to \(L_1 h_p+\eta\) and \(L_2 h_p+\eta\), with independent zero-mean eta and unequal cross-covariance maps L_i. Each joint covariance is PSD by construction, and received-pilot laws agree. Pilot H_p,b_p,c_p* agree, whereas channel b_d and generally c_d* differ. For Gaussian clean pilots of covariance I and observation noise sigma²I, choosing \(L_i=(1+\sigma^2)\sum c_{i,m}B_m\) with Ws=0 and source means=0 gives matched data operator \(\sum c_{i,m}B_m\); its dense MSE oracle is c_i if H_d is positive definite. This proves a general obstruction, not a claim that these artificial laws are CDL realizations. Sparse pilot observations alone cannot identify an arbitrary target data–pilot map.

## 2. Identifiability and actual geometry

For a fixed observed window let H=G^TG=[[a,b],[b,d]], where a=||g_P||², d=||g_M||² and b=g_P^Tg_M are real. Unique **empirical coordinate identifiability** is equivalent to rank_R(G)=2, equivalently

\[
a>0,\quad d>0,\quad ad-b^2>0.\tag{T01}
\]

This is not identifiability of physical profile/mobility factors, target covariance, or target oracle. For example, d_M=i d_P gives complex rank one but real rank two with b=0 and a=d>0. Imposing complex rank two would be wrong for the actual real coefficients.

The eigenvalues and condition numbers are

\[
\lambda_\pm=\frac{a+d\pm\sqrt{(a-d)^2+4b^2}}2,
\quad\lambda_{\min}(H)=\sigma_{\min}(G)^2,
\quad\kappa_2(H)=\lambda_+/\lambda_-=\kappa_2(G)^2.\tag{T02}
\]

At singular rank the condition number is infinite and no positive lower eigenvalue is asserted. The deployed solve has no new numerical rejection criterion. Exact T01 is a mathematical predicate, not a proposed deployment threshold.

Normalized column coherence \(\rho=b/\sqrt{ad}\) isolates real collinearity: normalized Gram eigenvalues are 1±|rho|. It does not replace the absolute excitation scale a,d; tiny equally scaled columns can have condition number one and still yield large perturbations.

Using the **actual fold selection maps** T_m from E10,

\[
H_{mn}=\sum_{\ell=j-4}^j\Re(u_\ell^HT_m^HT_nu_\ell).
\tag{T03}
\]

A coordinate direction q is unexcited iff \((q_PT_P+q_MT_M)u_\ell=0\) for every frame in the window. Thus geometry, origin covariance subblocks and the observed window jointly determine identifiability. Pilot count or window length by itself does not. Adding a frame with the **same frozen maps** adds a PSD Gram term, so lambda_min cannot decrease, but lambda_max/condition number can increase. Changing pilot geometry changes selection, K and DeltaK; the simple PSD argument does not compare different geometries. No alternate geometry or pilot density has been run here.

## 3. Noise enters design and residual

For fixed clean centered pilots v_l and independent proper AWGN, E11 implies

\[
\mathbb E[H_{mn}\mid v]=\sum_\ell\Re\{v_\ell^HT_m^HT_nv_\ell+\sigma^2\operatorname{tr}(T_m^HT_n)\},
\tag{T04}
\]
\[
\mathbb E[b_m\mid v]=\sum_\ell\Re\{v_\ell^HT_m^HUv_\ell+\sigma^2\operatorname{tr}(T_m^HU)\},
\tag{T05}
\]
\[
\mathbb E[(G^T\epsilon(c))_m\mid v]
=\sum_\ell\Re\{v_\ell^HT_m^HF_cv_\ell+\sigma^2\operatorname{tr}(T_m^HF_c)\}.
\tag{T06}
\]

Proof: expand (v+n)^H A(v+n), discard zero-mean cross terms, and use E nn^H=sigma²I. The trace term is present even if a clean-pilot relation is exact. Because S_H S_O^T=0 within each fold, its fold contribution is

\[
\mathbb E[D_{m,\ell f}^H\epsilon_{n,\ell f}\mid v]
=-\sigma^2\operatorname{tr}\{\Delta K_{m,f}^H(K_{s,f}+\sum c_n\Delta K_{n,f})\}.
\tag{T07}
\]

This explicitly links the endogenous noise to the cross-pilot covariance geometry. Heldout AWGN is independent of the observed AWGN **within that fold**, but the residual subtracts a noisy-input prediction; observed AWGN therefore remains shared with D. Across folds the same pilots swap roles. Noise can add apparent Gram excitation without identifying the target data operator. T06 does not imply bias of the nonlinear inverse H^-1 equals inverse-of-expected-H times expected score.

Conditioned on clean v, the one-frame noise residual has complex covariance sigma² F_c F_c^H. Its real expansion has covariance

\[
\frac{\sigma^2}{2}\begin{bmatrix}\Re F_cF_c^H&-\Im F_cF_c^H\\\Im F_cF_c^H&\Re F_cF_c^H\end{bmatrix}.
\tag{T08}
\]

Rows are generally correlated; five frames are not 640 independent observations. T08 is conditional on clean v, **not** conditional on noisy G, and cannot be plugged into an exogenous OLS formula. Cross-frame AWGN independence gives corresponding block covariance after row permutation; correlated channels still affect the unconditional law.

## 4. First-layer coefficient perturbations

For any fixed reference c* and nonsingular observed H, define epsilon=z-Gc*. The deployed solve gives the exact identity

\[
\widehat c-c^*=H^{-1}G^T\epsilon,
\qquad\|\widehat c-c^*\|\le\frac{\|\epsilon\|}{\sqrt{\lambda_{\min}(H)}}.
\tag{T09}
\]

Proof: subtract Hc* from G^Tz; the norm of G's left inverse is 1/sigma_min(G). This bound is pointwise, so it does not require independent error. It does not bound epsilon from knowledge of sigma² alone. Decompose c_hat-c_d*= (c_hat-c_p*)+(c_p*-c_d*): T09 with pilot oracle alone misses surrogate mismatch.

For fixed deterministic c*, finite conditional moments and A_G=H^-1G^T,

\[
\mathbb E[\widehat c-c^*\mid G]=A_G\mu_{\epsilon\mid G},\qquad
\operatorname{Cov}(\widehat c-c^*\mid G)=A_G\Sigma_{\epsilon\mid G}A_G^T.\tag{T10}
\]

If the reference depends on random truth, its conditional variation must be included; T10 is not a covariance assertion about a random per-realization oracle. Classical tau² H^-1 additionally requires conditional centered residual and Sigma_epsilon|G=tau²I. Both are unestablished; raw complex sigma² also corresponds to sigma²/2 per real component only for proper white noise before fold transformations.

### Actual pilot perturbation changes both design and response

Stack a window perturbation delta u in the same order. E10 induces delta G by columns T_m delta u, and delta z by U delta u, with real expansion. Let G'=G+E, z'=z+d, c'=the new solve, e=z-G c_hat (so G^T e=0). Then

\[
(G'^TG')(c'-\widehat c)=G'^T(d-E\widehat c)+E^Te,\tag{T11}
\]
\[
\|c'-\widehat c\|\le
\frac{\|G'\|\|d-E\widehat c\|+\|E\|\|e\|}{\sigma_{\min}(G')^2}.\tag{T12}
\]

Proof: residual at c_hat under the perturbed observations is e+d-E c_hat; G'^T e=E^T e. If ||E||<sigma_min(G), Weyl gives sigma_min(G')>=sigma_min(G)-||E||>0; replacing the denominator with that square produces an a priori bound. This sufficient perturbation domain is a theorem assumption, **not a selected rejection threshold**. A pure response-only bound is sharper T09, but actual received-pilot changes are not pure-response perturbations. No perturbation law is inferred from T20B.

## 5. Exact coordinate-to-channel propagation

Hold the current noisy pilot u_j and frozen factors fixed when comparing coordinates. For c_ref and c_hat,

\[
\Delta\widehat h_{d,j}=X_4(u_j)(\widehat c-c_{ref})
=\Delta\alpha A_{P,4}(Q_{P,4}u_j)+\Delta\beta A_{M,4}(Q_{M,4}u_j).\tag{T13}
\]

Thus the user's `B_m,4 y` expression is exact if y means **source-centered received Pilot-LS u_j**, not raw received grid or clean pilots. Define M_j=Re(X_4(u_j)^H X_4(u_j)), a PSD real 2×2 matrix. Then

\[
\|\Delta\widehat h_d\|^2=\Delta c^T M_j\Delta c
=\Delta\alpha^2\|B_{P,4}u_j\|^2+\Delta\beta^2\|B_{M,4}u_j\|^2
+2\Delta\alpha\Delta\beta\Re\{(B_{P,4}u_j)^H B_{M,4}u_j\}.\tag{T14}
\]

Both M_j and delta c depend on the current pilot; delta c also depends on four preceding frames. The two module outputs are not assumed orthogonal.

Combining T09 with T13 gives an exact response-residual gain and a pointwise bound:

\[
\Delta\widehat h_d=X_4(u_j)H_j^{-1}G_j^T\epsilon_j,
\quad\|\Delta\widehat h_d\|^2\le
\frac{\lambda_{\max}(M_j)}{\lambda_{\min}(H_j)}\|\epsilon_j\|^2.\tag{T15}
\]

The sharper worst-case gain is \(\|[\Re X_4;\Im X_4]H^{-1}G^T\|_2\). This connects the **actual fold/window** excitation to the **current rank4 output**; it is not a channel-NMSE bound without separately controlling residual and representation terms. Expectation must retain the random ratio jointly; replacing it by a ratio of expectations or E M times coordinate covariance is unjustified. Conditionally on the **whole observed pilot window**, delta c and M are deterministic for fixed c_ref, so a conditional covariance there is zero and provides no uncertainty certificate.

If the pilots themselves change, there is an additional direct observation effect. With W_4(c)=Ws+sum c_m B_m,4,

\[
\widehat h(u+\delta u,c+\delta c)-\widehat h(u,c)
=W_4(c)\delta u+X_4(u)\delta c+\sum_m\delta c_m B_{m,4}\delta u.\tag{T16}
\]

Omitting the first or bilinear last term would confuse coordinate propagation at fixed pilots with total received-pilot sensitivity. The validation evaluates both identities against the actual factored predictor.

## 6. Four error definitions: exact vectors and cross terms

Choose one **dense population channel-MSE oracle c*=c_d,dense*** as reference, or explicitly declare another fixed reference. Let W(c*)=Ws+sum c*_m B_m; W4(c_hat)=Ws+sum c_hat_m B_m,4. With clean v=h_p-mu_ps and noisy u=v+n, define

\[
e_{rep}=\mu_{d,s}+W(c^*)v-h_d,\qquad
e_{coord}=\sum_m(\widehat c_m-c_m^*)B_m v,
\tag{D01}
\]
\[
e_{comp}=\sum_m\widehat c_m(B_{m,4}-B_m)v,
\qquad e_{noise}=W_4(\widehat c)n.\tag{D02}
\]

Then, **pathwise and exactly**,

\[
\widehat h_d-h_d=e_{rep}+e_{coord}+e_{comp}+e_{noise}.\tag{D03}
\]

This clean-input bookkeeping isolates explicit observation noise. e_rep includes finite dense operator-space limitation, source-mean bias and any unpredictable clean data component; it is not exclusively an operator projection residual. e_coord/e_comp use clean v for this bookkeeping, while the independently defined deployed coordinate comparison T13 uses noisy u and compressed corrections. e_noise is not independent because c_hat depends on n; the noise-induced coordinate changes also affect e_coord/e_comp. None is an independently switchable measured causal contribution. No noise is added a second time.

All four scalar squared energies exist if the required moments exist. They are **not generally additive**:

\[
\|\widehat h-h\|^2=\sum_{a}\|e_a\|^2+2\sum_{a<b}\Re(e_a^He_b),
\quad \|\widehat h-h\|^2\le4\sum_a\|e_a\|^2.\tag{D04}
\]

There are six pairwise cross terms; expectation preserves them. Population optimality within a fixed linear coordinate class does not make adaptive c_hat(n) errors orthogonal.

### Alternative matched-reference convention (not another additive noise term)

Let \(h_t^*=\mu_{d,t}+W_t(p-\mu_{p,t})\), \(e_t=h_t^*-h_d\), and \(b_\mu=\mu_{d,s}-\mu_{d,t}+W_t(\mu_{p,t}-\mu_{p,s})\). Another exact four-vector convention is

\[
\widehat h_d-h_d=e_t+
\{b_\mu+[W(c^*)-W_t]u\}
+\sum_m(\widehat c_m-c_m^*)B_mu
+\sum_m\widehat c_m(B_{m,4}-B_m)u.\tag{D05}
\]

Here e_t is matched LMMSE **observation/prior residual**, not pure AWGN: it includes incomplete observation and channel randomness. Unlike D01–D03 it places noisy u in representation, coordinate and compression terms. D05 must not be mixed with the pure explicit-noise convention to obtain a five-term sum. LMMSE orthogonality to fixed affine functions of p does not remove adaptive nonlinear cross terms. If the true conditional mean using the entire window were used instead, a different orthogonality argument might be possible; that is not the frozen matched-statistics reference or a theorem claimed here.

The origin-weighted compression covariance is Rpp_m+sigma²I. Under source centering the prepared-origin input second moment also includes (mu_pm-mu_ps)(mu_pm-mu_ps)^H. At a target it changes again. Adaptive coefficients introduce module cross terms. The historical weighted SVD is unchanged; its singular tails alone are not an adaptive target-risk guarantee.

## 7. Stop-rule assessment

First-layer deliverable self-assessment: **PASS for an explanatory, actual-algorithm theory layer** if the independent sentinels pass. It contains more than textbook OLS: exact cross-pilot/noisy-design maps E10–11; fold-specific endogenous-noise score T07; distinct pilot/data oracles O01–04 and a same-pilot-law obstruction; actual pilot perturbations T11–12; and the geometry-to-rank4 propagation T13–16. These connections meet the stated scope's minimum utility.

**FAIL / NOT ESTABLISHED for a new general theoretical contribution, unbiased-coordinate guarantee, pilot-density performance law, or empirical uncertainty calibration.** No closed target-law covariance or strict risk-alignment result for CDL has been proved. If advisor judges these explanatory connections insufficient for a substantive paper theory contribution, mark `theory branch insufficient` and stop; no large U3 run is authorized by the algebra or self-assessment. This task stops at advisor review irrespective of that verdict.
