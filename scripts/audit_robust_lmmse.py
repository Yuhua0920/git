"""One frozen ID sample: independent spectral-Gram/direct-solve HR audit.

This diagnoses the observed poor HR result without repeating a numerical scan.
"""
import json
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.validate_b0_pipeline import json_save
from src import robust_lmmse as hr
from src.lmmse_train import formal_trajectory,raw_array_sha256,sha256_file
from src.nearest_validation import verify_frame
from src.trajectories import prepare_trajectory


def independent_spectral_gram():
    nodes,weights=leggauss(128)
    f=hr.centered_frequency_grid_hz(64,30000)
    times=np.arange(16)*(80/1920000)
    delay=hr.TAU_BOUND*(nodes+1)/2
    doppler=hr.FD_BOUND*nodes
    a=np.exp(-2j*np.pi*f[:,None]*delay[None,:])*np.sqrt(weights/2)
    b=np.exp(2j*np.pi*times[:,None]*doppler[None,:])*np.sqrt(weights/2)
    # Positive spectral Gram construction, independent of the analytic sinc code.
    return np.kron(a@a.conj().T,b@b.conj().T)


def audit():
    out=ROOT/"outputs"
    report=json.loads((out/f"{hr.PREFIX}_results.json").read_text(encoding="utf8"))
    before=dict(report["provenance"]["CSV_sha256"])
    full,_=hr.build_prior(); gram=independent_spectral_gram()
    cov_error=float(np.max(np.abs(full-gram)))
    refs=hr.load_replay_references(out)
    trajectory=formal_trajectory(prepare_trajectory(frames_per_trajectory=10),refs["ID"]["manifest"][0])
    h=trajectory.h_frames[0]
    bits,symbols=hr.link.generate_master_qpsk(bits_seeds=tuple(hr.bits_seed(2000,0,q) for q in range(2)))
    z=hr.complex_gaussian_noise((2,64,16),1.,np.random.default_rng(hr.noise_seed(2000,0)),dtype=np.complex128)
    verify_frame(refs["ID"]["realizations"][2000]["frame_realizations"][0],0,h,bits,z)
    x=hr.link.assemble_pilot_data_frame(symbols,hr.LAYOUT)
    clean=hr.link.apply_frequency_mimo_channel(h,x)
    pilots=hr.map_pilot_symbols(hr.LAYOUT,hr.link.PILOT_SYMBOLS)
    clean_ls,_=hr.pilot_ls_estimate(clean,pilots,hr.LAYOUT.pilot_mask)
    pilot_error=max(float(np.max(np.abs(clean_ls[:,q,hr.LAYOUT.pilot_mask[q]]-h[:,q,hr.LAYOUT.pilot_mask[q]]))) for q in range(2))
    variance=.01
    y=hr.link.add_received_noise(clean,hr.scale_unit_noise(z,variance))
    observations,_=hr.pilot_ls_estimate(y,pilots,hr.LAYOUT.pilot_mask)
    objects=tuple(hr.lmmse.prepare_oracle_lmmse(full,hr.lmmse.pilot_flat_indices(hr.LAYOUT.pilot_mask[q]),variance,(64,16)) for q in range(2))
    actual=hr.lmmse.estimate_mimo_from_pilot_ls(observations,objects)
    manual=np.empty_like(actual)
    weights_by_tx=[]
    for q,obj in enumerate(objects):
        indices=obj.pilot_indices
        rpp=gram[np.ix_(indices,indices)]; rhp=gram[:,indices]
        samples=observations[:,q].reshape(2,1024)[:,indices]
        manual[:,q]=(rhp@np.linalg.solve(rpp+variance*np.eye(16),samples.T)).T.reshape(2,64,16)
        # Linear map calculated by a solve; no inverse/pinv/jitter.
        weights_by_tx.append(np.linalg.solve(rpp+variance*np.eye(16),rhp.conj().T).conj().T)
    solve_error=float(np.max(np.abs(actual-manual)))
    error,truth=hr.channel_error_energies(actual,h,hr.LAYOUT.data_mask)
    # Diagnostic expected NMSE under matched ID channel moments, not an old
    # estimator run. Separate prior mismatch and noise contributions to R6 MSE.
    true_cov=hr.lmmse.build_full_grid_covariance(hr.lmmse.build_frequency_covariance(),hr.lmmse.build_time_covariance())
    bias_sum,noise_sum,n=0.,0.,0
    for q,(obj,w) in enumerate(zip(objects,weights_by_tx)):
        indices=obj.pilot_indices
        bias=np.diag(true_cov).real-2*np.real(np.sum(w*true_cov[indices,:].T,axis=1))
        bias+=np.real(np.sum((w@true_cov[np.ix_(indices,indices)])*w.conj(),axis=1))
        noise=variance*np.sum(np.abs(w)**2,axis=1)
        mask=hr.LAYOUT.data_mask[q].reshape(-1)
        bias_sum+=float(bias[mask].sum()); noise_sum+=float(noise[mask].sum()); n+=int(mask.sum())
    expected=(bias_sum+noise_sum)/n
    recorded=next(r for r in report["id_results"] if r["snr_db"]==20)["R6_HR_nmse_db"]
    theoretical_db=float(10*np.log10(expected))
    checks={"spectral_Gram_matches_analytic_covariance":cov_error<1e-12,
        "owner_pilot_scale_and_mute_correct":pilot_error<1e-12,
        "direct_solve_matches_Cholesky_estimate":solve_error<1e-12,
        "matched_ID_moments_predict_observed_R6_NMSE":abs(theoretical_db-recorded)<.1,
        "numeric_CSV_unchanged":before=={s:sha256_file(out/f"{hr.PREFIX}{s}.csv") for s in before}}
    result={"task":"T13B supplemental correctness audit","sample":{"condition":"ID","trajectory_id":2000,"frame":0,"snr_db":20},
        "covariance_max_error":cov_error,"noiseless_owner_Pilot_LS_max_error":pilot_error,"direct_solve_max_error":solve_error,
        "sample_NMSE_db":float(10*np.log10(error/truth)),"expected_ID_R6_NMSE_db":theoretical_db,
        "recorded_ID_R6_NMSE_db":recorded,"expected_minus_observed_db":theoretical_db-recorded,
        "prior_mismatch_error_per_data_RE":bias_sum/n,"observation_noise_error_per_data_RE":noise_sum/n,
        "diagnosis":"fixed uniform support prior mismatch dominates noise; frequency pilot delay aliasing limits identifiability",
        "numerical_scan_rerun":False,"new_seed_used":False,"source_sha256":sha256_file(Path(__file__)),
        "checks":checks,"all_checks_passed":all(checks.values())}
    json_save(out/f"{hr.PREFIX}_independent_audit.json",result)
    if not result["all_checks_passed"]:
        raise ValueError("independent HR audit failed")
    report["independent_correctness_audit"]={"path":str(out/f"{hr.PREFIX}_independent_audit.json"),**result}
    json_save(out/f"{hr.PREFIX}_results.json",report)
    return result


if __name__=="__main__":
    print(json.dumps(audit(),indent=2))
