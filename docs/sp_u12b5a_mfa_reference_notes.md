# Reference provenance and relationship

Fesl, Turan and Utschick, Low-Rank Structured MMSE Channel Estimation with Mixtures of Factor Analyzers, Asilomar 2023, 375–380, DOI 10.1109/IEEECONF59524.2023.10477088. Full author manuscript arXiv 2304.14809v2 (2023-05-01).

Selected 2023 estimator 93ce8eec4770840a377581e70de14c1a447a628d and its exact submodule core 184e97c641cd0d1383d28bce102a53b10a527c89. GitHub archives obtained and hashed. Both BSD-3-Clause contributions; core original MIT notice retained. Current 2026 copies retained separately, never automatically updated.

2026 differs: default_rng with explicit random_state, initialized residual floor, safe Nk epsilon, clipped responsibilities renormalized (only when rs_clip>0), stopping denominator floor, safe-weight inference floor, sorted full mixture with renormalization, zero-mean option and shape validation. These are not silently adopted. 2023 inference float1.0 retains unsorted all-component sum and no weight floor.

A NumPy2 compatibility alias np.Inf→np.inf is needed only in private author-module globals; import namespace adaptation reconstructs original submodule layout. No original source bytes are edited. Recorded legacy global RNG state is scoped/restored by wrapper, preserving author initialization distribution and sequential draws.

Paper figures use measurement data and normalization without a public complete numerical recipe: REFERENCE_NUMERICAL_REPLICATION_NOT_FULLY_SPECIFIED. Use author demo, training-sequence checks and posterior parity, not guessed published NMSE.

MFA learns a probabilistic channel distribution with low-rank-plus-residual mixture structure and observation-dependent posterior weights. This project uses a fixed Source LMMSE anchor plus cross-pilot-fitted prepared correction directions. This is a mechanism comparison without performance, firstness or transfer claims.


Original author demo diagnostic: rs_clip=1e-6 does not renormalize training responsibilities; fitted weights sum to 1.0000140301507119, so the unit-weight diagnostic FAIL is retained (tolerance1e-10). Demo expected behavior is completion, printed sum and finite arrays, without an author assertion of exact normalization. The first validation attempt and failed diagnostic were archived before clarifying this expectation. This is not a tolerance relaxation or model repair. Baseline rs_clip=0 was frozen before tests and passes weight/responsibility checks.
