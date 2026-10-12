# SP-U12B1 — Raw proof and code audit

No scientific experiment or production-code modification.

## Proposition 1

- equations: eq:general-rank-correction;eq:weighted-optimum
- code: src/lowrank_modules.py:98–116
- assumptions: Fixed B_m; C_m positive definite; invertible lower L_m; rank r within matrix dimensions
- proof: Invertible rank-preserving Y=XL_m; standard Eckart–Young–Mirsky; right solve QL=V^H
- scope: Weighted correction approximation, not target channel-MSE optimum; no new truncated-SVD theorem
- result: PASS

## Corollary 1

- equations: eq:reference-tail;eq:true-moment-error
- code: src/cdl_benchmark.py:149–177;src/lowrank_modules.py:98–116
- assumptions: Condition on prepared corpus; uniform clean-frame draw; independent CN noise; population 1/N centered covariance
- proof: E[uu^H]=C_m+delta_mu delta_mu^H; trace identity; Proposition 1
- scope: Reference-law equality; unknown target moments use their own trace identity; no frame independence needed
- result: PASS

## Compression propagation

- equations: eq:compression-cross;eq:compression-risk
- code: src/lowrank_modules.py:158–179;src/frozen_affine_validation.py:103–125
- assumptions: Same real fitted coordinates/input in Dense and LR
- proof: Expand norm of -alpha E_P u-beta E_M u; then expand true error difference
- scope: Coordinates depend on pilots; retain two-correction and reference-error cross terms; not a spectral-tail NMSE guarantee
- result: PASS

## Proposition 2

- equations: eq:coordinate-fit;eq:normal;eq:conditioning
- code: src/pilot_cv.py:70–99
- assumptions: Fixed observed real G640x2 and response; no prior on coordinates
- proof: Hessian2G^TG; null-space alternatives; squared singular values
- scope: 640 rows not excitation guarantee; noisy design/response; no classical exogenous homoscedastic OLS covariance
- result: PASS

## Lemma S1

- equations: eq:pilot-perturbation-map;eq:normal-perturbation;eq:coordinate-perturbation-bound;eq:joint-output-perturbation
- code: src/pilot_cv.py:70–99;src/lowrank_modules.py:158–179
- assumptions: Same window/fixed kernels/statistics/noise; S positive definite; ||E_S||<lambda_min(S)
- proof: Exact normal-system subtraction; quadratic-form eigenvalue bound; triangle inequality; full output expansion
- scope: Deterministic local sufficient condition; altered truth must also be subtracted; no beneficial-noise or recency theory
- result: PASS

## Exact manuscript source

\begin{samepage}
\begin{proposition}[Weighted rank-$r$ correction approximation]\label{prop:weighted}
Let $B_m\in\mathbb C^{n_d\times n_p}$, $C_m=L_mL_m^H\succ0$, and let $B_mL_m=U_m\Sigma_mV_m^H$ have singular values $\sigma_{m,1}\ge\cdots\ge0$.
For $1\le r\le\min(n_d,n_p)$, define
\begin{equation}\label{eq:general-rank-correction}
B_{m,r}=U_{m,r}\Sigma_{m,r}V_{m,r}^HL_m^{-1}.
\end{equation}
Then
\begin{align}\label{eq:weighted-optimum}
B_{m,r}&\in\arg\min_{\operatorname{rank}(X)\le r}
           \|(B_m-X)L_m\|_F^2,\\
\min_{\operatorname{rank}(X)\le r}\|(B_m-X)L_m\|_F^2
 &=\sum_{k>r}\sigma_{m,k}^2.\nonumber
\end{align}
\end{proposition}
\begin{proof}
Set $Y=XL_m$. Since $L_m$ is invertible, this is a bijection preserving rank, and
$\|(B_m-X)L_m\|_F=\|B_mL_m-Y\|_F$.
The standard Eckart--Young--Mirsky theorem gives the truncated SVD as a best rank-$r$ approximation to $B_mL_m$, with squared residual $\sum_{k>r}\sigma_{m,k}^2$.
Transforming back gives~\eqref{eq:general-rank-correction}.
\end{proof}
\end{samepage}
Proposition~\ref{prop:weighted} applies the standard approximation result to the correction weight in~\eqref{eq:weight}; the deployed factors in~\eqref{eq:factors} implement its rank-four instance.

\begin{samepage}
\begin{corollary}[Reference-moment correction error]\label{cor:reference}
Condition on the prepared corpus for origin $m$.
Draw a clean pilot frame uniformly from that corpus and add independent $n\sim\mathcal{CN}(0,\sigma^2I_{n_p})$ to obtain $p$.
For $u=p-\mu_{p,s}$ and $E_{m,r}=B_m-B_{m,r}$,
\begin{equation}\label{eq:reference-tail}
\mathbb E_{\mathrm{ref}}\|E_{m,r}u\|_2^2
=\sum_{k>r}\sigma_{m,k}^2
 +\|E_{m,r}(\mu_{p,m}-\mu_{p,s})\|_2^2.
\end{equation}
\end{corollary}
\begin{proof}
The empirical $1/N$ convention makes the reference mean $\mu_{p,m}$ and centered pilot-observation covariance $C_m$ exactly.
Hence $\mathbb E_{\mathrm{ref}}[uu^H]=C_m+\delta\mu_m\delta\mu_m^H$, where $\delta\mu_m=\mu_{p,m}-\mu_{p,s}$.
Taking the trace after multiplication by $E_{m,r}$ gives
$\|E_{m,r}L_m\|_F^2+\|E_{m,r}\delta\mu_m\|_2^2$.
Proposition~\ref{prop:weighted} identifies the first term as the singular-value tail.
\end{proof}
\end{samepage}
This identity describes the prepared empirical reference law.
For an actual pilot-observation law with finite mean $\mu_{\mathrm{true}}$ and centered covariance $C_{\mathrm{true}}$, the corresponding expression is
\begin{equation}\label{eq:true-moment-error}\begin{aligned}
\mathbb E\|E_{m,r}u\|_2^2
&=\operatorname{tr}(E_{m,r}C_{\mathrm{true}}E_{m,r}^H)\\
&\quad+\|E_{m,r}(\mu_{\mathrm{true}}-\mu_{p,s})\|_2^2.
\end{aligned}\end{equation}
The finite preparation moments therefore specify the reference, not the unknown target population.

For the same current input and real coordinates in dense and rank-$r$ deployment, write $a=E_{P,r}u$ and $b=E_{M,r}u$.
The output difference $d_h=\widehat h^{\mathrm{LR}}-\widehat h^{\mathrm{dense}}=-\alpha a-\beta b$ obeys
\begin{equation}\label{eq:compression-cross}\begin{aligned}
\|d_h\|_2^2
&=\alpha^2\|a\|_2^2+\beta^2\|b\|_2^2\\
&\quad+2\alpha\beta\operatorname{Re}(a^Hb).
\end{aligned}\end{equation}
With $e_D=\widehat h^{\mathrm{dense}}-h_d$, the exact squared-error change is
\begin{equation}\label{eq:compression-risk}
\|e_D+d_h\|_2^2-\|e_D\|_2^2
=\|d_h\|_2^2+2\operatorname{Re}(e_D^Hd_h).
\end{equation}
The fitted coordinates and $u$ depend on the same observations, so the reference tails alone do not determine the final NMSE penalty.

\subsection{Online Channel Estimation}
% audit: METHOD_DESCRIPTION | src/lowrank_modules.py:158-179;src/frozen_affine_validation.py:103-125
The fitted coordinates $\widehat c_j=[\widehat\alpha_j,\widehat\beta_j]^T$ are shared by Dense Affine and LR4 Affine.
The latter applies the stored correction factors to the current input:
\begin{equation}\label{eq:final-estimate}\begin{aligned}
\widehat h_{d,j}&=\mu_{d,s}+W_su_j\\
&\quad+\widehat\alpha_j A_{P,4}(Q_{P,4}u_j)
 +\widehat\beta_j A_{M,4}(Q_{M,4}u_j).
\end{aligned}\end{equation}
Each correction uses two matrix-vector products, without constructing a dense approximate correction or performing a new factorization.
The online interface supplies no target clean channel state information (CSI), target covariance, profile label or mobility label.
Fig.~\ref{fig:workflow} summarizes the interface; Algorithm~\ref{alg:stream} specifies its frame-stream computation.
The first four frames fill the causal buffer; each subsequent frame produces an estimate.

\begin{algorithm}[!htbp]
\caption{Pilot-driven source-anchored LMMSE adaptation with rank-four corrections}\label{alg:stream}
\small\linespread{1}\selectfont
\begin{algorithmic}[1]
\Statex \textbf{Input:} Received owner-pilot symbols; known transmitted pilot symbols; pilot geometry/order; operating noise variance $\sigma^2$.
\Statex \textbf{Precomputed quantities:} $\mu_{p,s},\mu_{d,s},W_s$; four folds $(\mathcal O_f,\mathcal H_f)$;
\Statex \hspace{\algorithmicindent}$K_{s,f},\Delta K_{P,f},\Delta K_{M,f}$, $f=0,\ldots,3$;
\Statex \hspace{\algorithmicindent}$(A_{P,4},Q_{P,4})$ and $(A_{M,4},Q_{M,4})$ instantiated at $\sigma^2$.
\State Initialize an empty five-frame causal buffer.
\For{$j=0,1,\ldots$}
    \State Form $p_j$ by owner-pilot division and stacking~\eqref{eq:pilot-ls}.
    \State Update the buffer, retaining the latest five pilot vectors.
    \If{the buffer is not full}
        \State \textbf{continue}
    \EndIf
    \State $S_j\gets0_{2\times2}$; $b_j\gets0_2$.
    \For{$\ell=j-4,\ldots,j$}
        \State $u_\ell\gets p_\ell-\mu_{p,s}$.
        \For{$f=0,1,2,3$}
            \State $x\gets u_{\ell,\mathcal O_f}$.
            \State $D_{\ell f}\gets[\Delta K_{P,f}x,\;\Delta K_{M,f}x]$.
            \State $e_{\ell f}\gets u_{\ell,\mathcal H_f}-K_{s,f}x$.
            \State $S_j\gets S_j+\operatorname{Re}(D_{\ell f}^HD_{\ell f})$.
            \State $b_j\gets b_j+\operatorname{Re}(D_{\ell f}^He_{\ell f})$.
        \EndFor
    \EndFor
    \State Solve $S_j\widehat c_j=b_j$ over $\mathbb R^2$ for $(\widehat\alpha_j,\widehat\beta_j)$.
    \State $u_j\gets p_j-\mu_{p,s}$.
    \State $\begin{aligned}
        \widehat h_{d,j}&\gets\mu_{d,s}+W_su_j\\
        &\quad+\widehat\alpha_j A_{P,4}(Q_{P,4}u_j)+\widehat\beta_j A_{M,4}(Q_{M,4}u_j).
    \end{aligned}$
    \State Emit $\widehat h_{d,j}$.
\EndFor
\Statex \textbf{Output:} Current data-domain estimates $\widehat h_{d,j}$ for every $j\ge4$.
\end{algorithmic}
\end{algorithm}

% audit: METHOD_DESCRIPTION | docs/paper_method_figure_spec.md;docs/t20a_final_method_definition.md
\begin{figure}[!htbp]\centering
\includegraphics[width=1.00\linewidth]{figures/method_workflow.pdf}
\caption{Offline clean statistics prepare the full Source operator, two weighted rank-four correction-factor pairs and full cross-pilot predictors. Online fitting uses received pilots in a causal five-frame buffer. Only data-domain corrections are compressed.}\label{fig:workflow}
\end{figure}

\subsection{Coordinate Identifiability and Output Error}
\begin{samepage}
\begin{proposition}[Identifiability of the cross-pilot coordinates]\label{prop:identifiability}
For a fixed observed window, $J_j(c)=\|G_jc-z_j\|_2^2$, $c\in\mathbb R^2$, has a unique minimizer if and only if $\operatorname{rank}_{\mathbb R}(G_j)=2$, equivalently $S_j=G_j^TG_j\succ0$.
In that case,
\begin{equation}\label{eq:conditioning}
\lambda_{\min}(S_j)=s_{\min}(G_j)^2>0,
\qquad\kappa_2(S_j)=\kappa_2(G_j)^2.
\end{equation}
\end{proposition}
\begin{proof}
The Hessian of $J_j$ is $2G_j^TG_j$, which is positive definite precisely when $G_j$ has full column rank.
Otherwise, least squares still has a minimizer, and adding any nonzero vector in $\ker G_j$ gives another minimizer.
For full column rank, the eigenvalues of $S_j$ are the squared singular values of $G_j$, yielding~\eqref{eq:conditioning}.
\end{proof}
\end{samepage}
This is an excitation condition for the observed correction responses; $640$ rows alone do not guarantee it.
Both $G_j$ and $z_j$ arise from noisy pilots, with shared observations across folds and correlated frames.
Their construction does not establish the classical exogenous, homoscedastic OLS model, so its coefficient covariance formula is not applied here.
Conditioning describes sensitivity of the coordinate solve rather than directly determining channel NMSE.
Supplementary Section~\ref*{supp-sec:coordinate-perturbation} gives a conditional deterministic perturbation bound.

For the current input, define $X_4(u_j)=[B_{P,4}u_j,\;B_{M,4}u_j]$ and $\widehat h_{d,j}(c)=\mu_{d,s}+W_su_j+X_4(u_j)c$.
With a specified reference $c^\star$ and $\Delta c_j=\widehat c_j-c^\star$, the exact output perturbation is
\begin{equation}\label{eq:coordinate-propagation}\begin{gathered}
\Delta\widehat h_{d,j}=X_4(u_j)\Delta c_j,\\
 \|\Delta\widehat h_{d,j}\|_2^2=\Delta c_j^TM_4(u_j)\Delta c_j,\\
 M_4(u_j)=\operatorname{Re}[X_4(u_j)^HX_4(u_j)].
\end{gathered}\end{equation}
Here $M_4(u_j)$ and the coordinates depend on current noisy pilots; the coordinates also use previous frames.
For $e_{\mathrm{ref},j}=\widehat h_{d,j}(c^\star)-h_{d,j}$, the data-error change is
\begin{align}\label{eq:risk-difference}
 &\|\widehat h_{d,j}(\widehat c_j)-h_{d,j}\|_2^2
 -\|\widehat h_{d,j}(c^\star)-h_{d,j}\|_2^2\nonumber\\
 &\qquad=\Delta c_j^TM_4(u_j)\Delta c_j
 +2\operatorname{Re}\{e_{\mathrm{ref},j}^HX_4(u_j)\Delta c_j\}.
\end{align}
The quadratic term measures output perturbation energy, not NMSE degradation by itself.
The observed-window cross-pilot fit minimizes a pilot-domain surrogate; its optimum need not equal the data-risk optimum.
Supplementary Section~\ref*{supp-sec:math-note} retains the noisy-design interpretation and population objectives.

\subsection{General Arithmetic and Storage Analysis}
Let $M$ be the number of correction directions, $F$ the number of folds, $T_w$ the number of fitting-window frames, $n_p$ the full pilot dimension, $n_d$ the data-domain dimension, $n_o$ the fold-input size, $n_h$ the fold-held-out size, and $r$ the common correction rank.
Here $T_w$ counts frames, whereas $T$ in the system model counts OFDM symbols per frame.
For a complete window recomputation at each update, the dominant complex-MAC proxies are
\begin{align}\label{eq:general-complexity}
C_{\mathrm{Source}}&=n_dn_p,\\
C_{\mathrm{fit}}&=(M+1)FT_wn_on_h,\\
C_{\mathrm{dense\text{-}corr}}&=Mn_dn_p,\\
C_{\mathrm{LR\text{-}corr}}&=Mr(n_d+n_p).
\end{align}
The fitting term counts one Source and $M$ correction-kernel products per fold and buffered frame.
The two total deployment proxies add $C_{\mathrm{Source}}+C_{\mathrm{fit}}$ to the appropriate correction term.
Gram/right-hand-side accumulation costs $O(FT_wn_hM^2)$, and the small real solve costs $O(M^3)$ separately.
For the present $M=2$ these terms are small relative to the kernel products; their relative importance changes with the dimensions.
Scalar coefficient products, detection, memory traffic and device transfers also lie outside the dominant complex-MAC count.

With $M=2$, $F=4$, $T_w=5$, $n_p=64$, $n_d=3968$, $n_o=48$, $n_h=16$ and $r=4$, the Source, fit, dense-correction and rank-four-correction counts are respectively $253952$, $46080$, $507904$ and $32256$.
The dense and rank-four totals are $807936$ and $332288$, giving a $58.87199\%$ reduction in this arithmetic proxy.
Each update fully reevaluates the window; the count assumes no rolling or incremental reuse.

Dense corrections store $Mn_dn_p$ complex coefficients, compared with $Mr(n_d+n_p)$ for the low-rank factor pairs.
For a single operating noise variance, an estimator-only storage inventory additionally contains
\begin{equation}\label{eq:deployment-storage}\begin{aligned}
N_{\mathrm{common}}
&=n_dn_p+(n_d+n_p)\\
&\quad+(M+1)Fn_on_h+T_wn_p
\end{aligned}\end{equation}
complex coefficients for Source, means, full cross-pilot kernels and the causal buffer.
Small real matrices/vectors, pilot/fold metadata and working arrays are additional; optional multiple-noise caches and offline preparation statistics have separate storage.
The correction-only counts $507904$ and $32256$ therefore differ from total deployment storage.
These formulas describe arithmetic and coefficient storage, with offline factor construction separate; measured runtime, latency and energy require implementation-specific measurements.


## Supplementary proof source


\section{Deterministic Perturbation of the Coordinate Solve}\label{sec:coordinate-perturbation}
Fix the offline statistics, means, kernels, operating noise variance and a five-frame window.
Let $\eta_\ell=\widetilde p_\ell-p_\ell$ change its received pilot vectors.
Use $\mathcal E_G=\widetilde G-G$ and $\mathcal E_z=\widetilde z-z$ for the real-expanded perturbations.
The prepared differences $\Delta K_{P,f},\Delta K_{M,f}$ remain fixed.
From main-text~\eqref{main-eq:design}, the changed blocks obey
\begin{align}\label{eq:pilot-perturbation-map}
\mathcal E_{D,\ell f}
&=[\Delta K_{P,f}\eta_{\ell,\mathcal O_f},\;
   \Delta K_{M,f}\eta_{\ell,\mathcal O_f}],\\
\mathcal E_{e,\ell f}
&=\eta_{\ell,\mathcal H_f}-K_{s,f}\eta_{\ell,\mathcal O_f}.
\end{align}
Stacking real and imaginary parts constructs $\mathcal E_G,\mathcal E_z$ exactly.
Write $S=G^TG$, $b=G^Tz$, $\widehat c=S^{-1}b$ and
\begin{align}\label{eq:normal-perturbation}
\mathcal E_S&=G^T\mathcal E_G+\mathcal E_G^TG
                 +\mathcal E_G^T\mathcal E_G,\\
\mathcal E_b&=G^T\mathcal E_z+\mathcal E_G^Tz
                 +\mathcal E_G^T\mathcal E_z.
\end{align}
\begin{lemma}[Perturbation of an identifiable coordinate solve]\label{lem:coordinate-perturbation}
Suppose $S\succ0$ and $\|\mathcal E_S\|_2<\lambda_{\min}(S)$.
The perturbed solve is uniquely defined, and $d_c=\widetilde c-\widehat c$ satisfies
\begin{equation}\label{eq:coordinate-perturbation-bound}
\|d_c\|_2\le
\frac{\|\mathcal E_b\|_2+\|\mathcal E_S\|_2\|\widehat c\|_2}
{\lambda_{\min}(S)-\|\mathcal E_S\|_2}.
\end{equation}
\end{lemma}
\begin{proof}
Subtract $S\widehat c=b$ from $(S+\mathcal E_S)\widetilde c=b+\mathcal E_b$ to obtain
$(S+\mathcal E_S)d_c=\mathcal E_b-\mathcal E_S\widehat c$.
For every real $v$, $v^T(S+\mathcal E_S)v\ge
(\lambda_{\min}(S)-\|\mathcal E_S\|_2)\|v\|_2^2$.
Thus the inverse norm is at most the reciprocal of this positive lower bound.
The triangle inequality gives~\eqref{eq:coordinate-perturbation-bound}.
\end{proof}
The condition is a sufficient local bound, not an assumed property of every observed window or an OLS covariance formula.
It perturbs the observed solve; $d_c$ is distinct from the reference-coordinate difference $\Delta c_j$ in the main text.

The current pilot perturbation also changes the estimator input.
For $u_j=p_j-\mu_{p,s}$, $X_4(v)=[B_{P,4}v,\;B_{M,4}v]$ and $\widehat c=(\widehat\alpha,\widehat\beta)^T$, the exact output change is
\begin{equation}\label{eq:joint-output-perturbation}\begin{aligned}
\widetilde h_{d,j}-\widehat h_{d,j}
&=(W_s+\widehat\alpha B_{P,4}+\widehat\beta B_{M,4})\eta_j\\
&\quad+X_4(u_j+\eta_j)d_c.
\end{aligned}\end{equation}
For a fixed true channel and $e=\widehat h_{d,j}-h_{d,j}$, putting $v_h=\widetilde h_{d,j}-\widehat h_{d,j}$ gives the squared-error change $\|v_h\|_2^2+2\operatorname{Re}(e^Hv_h)$.
If the true channel also changes, its change must be subtracted from $v_h$ before expanding the errors.
