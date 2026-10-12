# SP-U2A authoritative equation/code inventory

Task authority: researcher instruction dated 2026-10-04. Accepted HEAD: `ca83982b1d0e4bdca0215a3ecbe3a7a8f8f8ec55`. This inventory describes the unchanged deployed two-coordinate algorithm. It supersedes abbreviated SP-U1 notation for this task; it does not replace historical records or manuscript equations. Code line numbers refer to the protected working bytes captured by `scripts/validate_sp_u2a.py --snapshot`.

## 1. Observation and geometry

Let \(p_\ell\in\mathbb C^{64}\) denote **received Pilot-LS channel coefficients**, rather than the raw receiver grid. `src/estimators.py:6–42` divides the received signal by the known owner pilot, while the other TX is muted. Unit-modulus pilots preserve complex noise variance. Under the frozen AWGN model:

\[
p_\ell=h_{p,\ell}+n_\ell,\qquad n_\ell\sim\mathcal{CN}(0,\sigma^2 I_{64}),\qquad
u_\ell=p_\ell-\mu_{p,s}=v_\ell+n_\ell,\quad v_\ell=h_{p,\ell}-\mu_{p,s}.
\tag{E01}
\]

The 64 coefficients use C order `[rx,tx,k,t]` in a `[2,2,64,16]` grid. The selected physical frequencies for TX0 are 0,16,32,48 and for TX1 are 8,24,40,56; times are 0,4,8,12. Each spatial link has 16 observed coefficients. The union contains 32 REs (1/32 overhead); multiplication by Nr gives 64 observed coefficients. The **data-domain** channel vector has 3968 entries: four links times 992 nonunion REs. The full grid has 4096 entries; the 128 union-RE link coefficients are not part of this deployed data-domain output.

`src/pilots.py:26–85` constructs the masks; `src/spatial_information.py` supplies `observation_geometry` used by the frozen pipeline. The validation artifact records its live definition and the exact selected indices without loading any saved scientific observations.

## 2. Every fold

`src/pilot_cv.py:12–37` unravels the 64 selected entries and assigns

\[
f(k,t)=(\lfloor k/16\rfloor+\lfloor t/4\rfloor)\bmod4.\tag{E02}
\]

For each spatial link and frequency block \(b=0,1,2,3\), heldout time is \(t=4((f-b)\bmod4)\), with physical \(k=16b+8\,tx\). All other selected entries form O. RX does not change the fold assignment.

| Fold | Heldout time at blocks b=0,1,2,3, per RX/TX | Input | Target | Predicted quantity |
|---|---|---|---|---|
| 0 | 0,12,8,4 | \(u_{\ell,O_0}\), 48 complex | \(u_{\ell,H_0}\), 16 complex | heldout centered noisy Pilot-LS |
| 1 | 4,0,12,8 | \(u_{\ell,O_1}\), 48 complex | \(u_{\ell,H_1}\), 16 complex | same |
| 2 | 8,4,0,12 | \(u_{\ell,O_2}\), 48 complex | \(u_{\ell,H_2}\), 16 complex | same |
| 3 | 12,8,4,0 | \(u_{\ell,O_3}\), 48 complex | \(u_{\ell,H_3}\), 16 complex | same |

The emitted JSON additionally lists **all 64 selected grid indices, all fold labels, and every observed/heldout local index**. H sets partition 64; H and O are disjoint within a fold. Across folds a heldout observation reappears in three input sets. These are cross-pilot folds inside one frame, not four independent samples or four independent trajectories.

Let \(S_{O_f}\in\{0,1\}^{48\times64}\), \(S_{H_f}\in\{0,1\}^{16\times64}\) select those entries. For each prepared origin \(q\in\{s,P,M\}\),

\[
K_{q,f}=R_{pp,q}[H_f,O_f]\{R_{pp,q}[O_f,O_f]+\sigma^2I_{48}\}^{-1},
\quad\Delta K_{m,f}=K_{m,f}-K_{s,f}.\tag{E03}
\]

`src/pilot_cv.py:40–59` computes K by Cholesky and adjoint solves, never explicit inversion. `scripts/run_pilot_cv.py:43–58` builds the three origin kernels and their differences. Each K and DeltaK is 16×48. All are offline frozen at the receiver-known noise variance. **Rank-4 compression does not apply to these pilot kernels.** Source centering is used even for development-origin kernels: no origin-specific mean is inserted online.

## 3. Fold objective and real expansion

For unrestricted \(c=(\alpha,\beta)^T\in\mathbb R^2\):

\[
D_{\ell f}=[\Delta K_{P,f}S_{O_f}u_\ell,\ \Delta K_{M,f}S_{O_f}u_\ell]\in\mathbb C^{16\times2},
\quad r_{\ell f}=S_{H_f}u_\ell-K_{s,f}S_{O_f}u_\ell\in\mathbb C^{16}.
\tag{E04}
\]

Fold prediction is \(\mu_{p,s}[H_f]+K_{s,f}u_{\ell,O_f}+D_{\ell f}c\); objective is \(\|r_{\ell f}-D_{\ell f}c\|^2\), predicting noisy heldout coefficients. Neither target clean CSI nor target means appear.

Stack folds 0..3 within each frame, and frames in ascending order. Five-frame complex D is 320×2 and r is 320. Define

\[
G_j=\begin{bmatrix}\Re D_j\\\Im D_j\end{bmatrix}\in\mathbb R^{640\times2},
\qquad z_j=\begin{bmatrix}\Re r_j\\\Im r_j\end{bmatrix}\in\mathbb R^{640}.
\tag{E05}
\]

**Notation firewall:** `z` at `src/pilot_cv.py:74` is \(u_\ell\), the 64-entry source-centered noisy pilot. \(z_j\) in `z=Gc+epsilon` is instead the 640-entry real response. They must never be conflated. One-frame real dimensions are 128×2 and 128; a generic accepted n-frame call has 128n rows. The deployed primary path always uses five frames.

For real c, exact arithmetic gives

\[
\|r_j-D_jc\|^2=\|z_j-G_jc\|^2,
\quad H_j=G_j^TG_j=\sum_{\ell,f}\Re(D_{\ell f}^HD_{\ell f}),
\quad b_j=G_j^Tz_j=\sum_{\ell,f}\Re(D_{\ell f}^Hr_{\ell f}).\tag{E06}
\]

`frame_objective` does not allocate G. It accumulates complex 2×2 moments per frame, `fit` sums them then takes the real part. Stacking E05 changes floating point reduction order; equivalence is mathematical, verified to a stated complex128 tolerance, rather than claimed byte identity between reductions.

## 4. Window, solve and deployment

\[
\mathcal W_j=\{j-4,\ldots,j\},\qquad H_j\widehat c_j=b_j.\tag{E07}
\]

`causal_windows` accepts a `[10,64]` trajectory and emits j=4..9 with slices `[j-4:j+1]`. Current frame is included, no future frame or previous trajectory enters. `PilotCV.fit` accepts one to five frames for older arms; that general API is not the final primary protocol. Singular `np.linalg.solve` raises; no eigenvalue rejection threshold, clipping, ridge, pseudoinverse, fallback, sample exclusion or complex coordinate is introduced. The diagnostic condition number is **cond(H)**, not cond(G).

The same fitted c is used for dense and compressed data predictors:

\[
\widehat h_{d,j}=\mu_{d,s}+W_su_j+
\widehat\alpha_j A_{P,4}(Q_{P,4}u_j)+
\widehat\beta_j A_{M,4}(Q_{M,4}u_j).\tag{E08}
\]

Ws is 3968×64 full; each A is 3968×4; Q is 4×64. `B_m,4=A_m,4 Q_m,4` is a mathematical product; inference does not materialize it. `src/lowrank_modules.py:176–179` uses an ordinary transpose in `u @ W_s.T`, equivalent to Ws u in column notation, not a conjugate transpose. Development weighting remains \(B_mL_m=U\Sigma V^H,\ Q_mL_m=V_m^H\); `lowrank_modules.py:98–116` implements the right triangular solve. No factor is rebuilt by this task.

## 5. Line-by-line critical map

| Code lines | Exact operation | Equation |
|---|---|---|
| pilot_cv:13–19 | validate 64 indices; physical unravel; assign labels | E01–02 |
| pilot_cv:21–32 | each H/O, 16/48 counts, four link coverage, freeze arrays | E02 |
| pilot_cv:42–46 | finite Hermitian frozen covariance, positive variance | E03 assumptions |
| pilot_cv:48–54 | covariance subblocks, Cholesky, adjoint solve | E03 |
| pilot_cv:55–58 | solve residual audit and frozen kernel | E03 numerical check |
| pilot_cv:72–74 | current finite pilots, source centering | E01 |
| pilot_cv:76–77 | observed input; source heldout prediction | E04 |
| pilot_cv:78 | P/M difference predictions, two columns | E04 D |
| pilot_cv:79 | withheld noisy response minus source | E04 r |
| pilot_cv:80–81 | concatenate folds; complex Gram/RHS/base loss | E05–06 |
| pilot_cv:85–87 | window size and declared real-coordinate arm | E07 |
| pilot_cv:88–91 | sum frame moments; real Gram/RHS; primary uses both columns | E06–07 |
| pilot_cv:92–93 | normal equation solve and finite-output check | E07 |
| pilot_cv:94–99 | base/fitted losses, Gram condition/residual diagnostics | E06 |
| pilot_cv:102–105 | exact five-frame causal slices | E07 |
| pilot_cv:108–114 | dense source-centered prediction | dense E08 |
| pilot_cv:123–128 | uniform two-coordinate primary fit/deploy | E07–08 |
| frozen_affine_validation:103–125 | primary5 window, single fit reused dense/LR4 | E07–08 |
| lowrank_modules:74–76,158–179 | two factor applications, real coordinates, source means | E08 |
| frozen_affine_validation:129–150 | legacy centered operator oracle, not full mean-aware MSE oracle | theory O04 |

## 6. Is z=Gc+epsilon exact?

For any declared reference c*, define \(\epsilon_j(c^*)=z_j-G_jc^*\). Then

\[
z_j=G_jc^*+\epsilon_j(c^*)\tag{E09}
\]

is **exact**. No approximation is made to the algorithm or operators. It is a residual identity, not proof of a correctly specified exogenous stochastic linear model. It need not mean a physically true constant coordinate generated these pilots. Choosing c*=the fitted empirical minimizer makes E09 tautological and cannot establish unbiasedness.

Write the one-frame stacked complex maps

\[
T_m=\operatorname{stack}_f(\Delta K_{m,f}S_{O_f}),\quad
U=\operatorname{stack}_f(S_{H_f}-K_{s,f}S_{O_f}),\quad
F_c=U-c_PT_P-c_MT_M.\tag{E10}
\]

Each is 64×64, with rows in fold order. For each frame,

\[
D_\ell=[T_P(v_\ell+n_\ell),T_M(v_\ell+n_\ell)],\quad
r_\ell=U(v_\ell+n_\ell),\quad
\epsilon_{\ell,\mathbb C}(c)=F_cv_\ell+F_cn_\ell.\tag{E11}
\]

Thus G explicitly depends on noisy observations, and the residual noise depends on the **same** noise. The assumption `epsilon = heldout AWGN only` is false: in fold f its noise term is \(n_H-(K_s+\alpha\Delta K_P+\beta\Delta K_M)n_O\). E09 with independent, centered, homoscedastic epsilon is only a further modeling assumption; it is not an exact description established by this code.

No old formal outcome was read to define these equations. Numerical fixtures use synthetic arrays and the existing fixed layout only.
