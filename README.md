# MIMO-OFDM statistical channel-estimation research — source snapshot

This repository contains the project source code, configuration, tests, selected mathematical/method documentation, and deployment scripts. The public scope deliberately excludes manuscripts, scientific results, training/evaluation data, model parameters, laboratory run records, personal Python environments and downloaded literature.

The code includes source-anchored LMMSE, cross-pilot coordinate fitting, correction and whole-operator compression, pooled-statistics controls, PPCA and prepared-origin Gaussian posterior estimation. Source-only copies of the later studies are kept under their original `outputs/revision_*/src`, `src_or_scripts`, `scripts` and `tests` paths so existing imports keep their layout. These paths contain code, not saved experiment outcomes.

## Environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Optional pinned environments are described in `requirements-cdl.txt`, `requirements-gmm.txt` and `requirements-ml.txt`. Install only the environment needed for the selected module; this upload does not upgrade or rebuild the author's original environments.

## Layout

- `src/`: channel, pilot, estimation, detector and numerical implementations.
- `scripts/`, `tests/`, `configs/`: project drivers, tests and configuration.
- `outputs/revision_*/`: selected later-study source and tests only.
- `experiments/`: source implementations and synthetic test utilities.
- `deployment/`: Windows scripts and technical execution code; no real approval or execution records.
- `reference/`: third-party reference source with its original copyright notices and BSD-3-Clause licenses.
- `docs/`: selected equations, assumptions, proofs and method notes.

Scientific execution and many fidelity/integration tests require excluded data, models and frozen manifests. They are not fully reproducible from this source-only repository. Do not interpret absent inputs as permission to regenerate, retune or automatically run an experiment. Nothing runs on clone or installation.

## Provenance

Files were copied from the current working tree, including required untracked source. `SOURCE_FILES.sha256` records exact original bytes. The local research Git history is preserved separately and has not been rewritten; its experimental outputs are not included in this new public snapshot. Third-party licenses apply to their respective folders. No new overall project license is assigned by this upload.
