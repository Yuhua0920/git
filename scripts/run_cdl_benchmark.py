"""T15A official CDL integration/audit. No model training or formal-test access."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".pytest_cache/matplotlib-cdl"))
import numpy as np
from scipy.linalg import eigh
from src.cdl_benchmark import (OfficialCDL,StreamingStatistics,FrozenStatistics,frames,array_hash,
    observation_geometry,seeds,manifest_rows,marginal_kronecker_audit,moments,normalized_correlation,
    SYSTEM,ARRAY,SNRS,PREFIXES,T14B_COMMIT,EXPERIMENT_TYPE)
from src.link import (generate_master_qpsk,assemble_pilot_data_frame,apply_frequency_mimo_channel,
    estimate_ls_linear_channel,detect_data_zf,extract_reference_data_bits,demodulate_data_bits)
from src.numerics import complex_gaussian_noise,noise_variance_from_snr
from scripts.check_cdl_environment import check_main_unchanged
OUT=ROOT/"outputs"
SOURCES=("src/cdl_benchmark.py","scripts/run_cdl_benchmark.py","scripts/check_cdl_adapter.py",
    "scripts/check_cdl_environment.py","scripts/plot_cdl_benchmark.py","tests/test_cdl_benchmark.py",
    "docs/cdl_benchmark_protocol.md","requirements-cdl.txt")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_default(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,complex): return [x.real,x.imag]
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)


def save_json(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,default=json_default,allow_nan=False)+"\n",encoding="utf8")


def save_csv(name,rows):
    with (OUT/name).open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def old_hashes():
    paths=subprocess.check_output(["git","ls-tree","-r","--name-only",T14B_COMMIT],cwd=ROOT,text=True).splitlines()
    return {p:sha(ROOT/p) for p in paths if p.startswith(("src/","tests/","configs/","outputs/"))}


def marginal_audit(train,val,positions):
    mean=train.mean(0); variance=np.mean(np.abs(train-mean)**2,axis=0)
    rows=[]
    for j,pos in enumerate(positions):
        z=(val[:,j]-mean[j])/np.sqrt(variance[j]); mag=np.abs(z)
        for component,x in (("real",np.sqrt(2)*z.real),("imag",np.sqrt(2)*z.imag)):
            rows.append(dict(diagnostic="marginal",position="/".join(map(str,pos)),component=component,
                normalization="train centered complex variance; real/imag times sqrt(2)",samples=len(x),**moments(x),
                magnitude_mean=float(mag.mean()),magnitude_second_moment=float(np.mean(mag**2)),
                magnitude_fourth_moment=float(np.mean(mag**4)),
                magnitude_q10=float(np.quantile(mag,.1)),magnitude_q50=float(np.quantile(mag,.5)),magnitude_q90=float(np.quantile(mag,.9))))
    return rows


def projection_audit(frozen,val_pilots):
    values,vectors=eigh(frozen.rpp)
    threshold=float(values[-1]*1e-10); keep=values>threshold
    whitening=vectors[:,keep]/np.sqrt(values[keep])[None,:]
    whitened=(val_pilots-frozen.mean_p)@whitening.conj()
    rank=int(keep.sum()); rng=np.random.default_rng(20261564)
    directions=complex_gaussian_noise((64,rank),1.,rng)
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    projected=whitened@directions.conj().T
    reference=complex_gaussian_noise((10000,rank),1.,np.random.default_rng(20261565))
    reference_projections=reference@directions.conj().T
    rows=[]
    for label,p in (("CDL validation",projected),("CN reference",reference_projections)):
        for i in range(64):
            for component,x in (("real",np.sqrt(2)*p[:,i].real),("imag",np.sqrt(2)*p[:,i].imag)):
                rows.append(dict(namespace=label,projection=i,component=component,samples=len(x),**moments(x)))
    def energy(z):
        e=np.sum(abs(z)**2,axis=1)/rank
        return dict(mean=float(e.mean()),variance=float(e.var()),q10=float(np.quantile(e,.1)),
            q50=float(np.quantile(e,.5)),q90=float(np.quantile(e,.9)))
    return rows,dict(whitening="train Rpp eigenfactor; retain lambda > 1e-10 lambda_max",
        retained_rank=rank,eigenvalues=values.tolist(),threshold=threshold,directions=64,
        projection_seed=20261564,reference_seed=20261565,
        projection_directions_sha256=array_hash(directions),validation_energy=energy(whitened),
        gaussian_reference_energy=energy(reference),gaussian_theory_energy_mean=1.,gaussian_theory_energy_variance=1/rank,
        diagnostic_only=True,joint_gaussianity_claimed=False)


def scan():
    if Path(sys.prefix).resolve()!=(ROOT/".venv-cdl").resolve(): raise ValueError("use isolated .venv-cdl")
    if (OUT/"t15a_cdl_results.json").exists(): raise ValueError("existing numeric run protected; use --finalize")
    started=time.perf_counter(); before=old_hashes()
    env=json.loads((OUT/"t15a_cdl_environment.json").read_text(encoding="utf8"))
    small=json.loads((OUT/"t15a_cdl_adapter_calibration.json").read_text(encoding="utf8"))
    if not env["all_checks_passed"] or not small["all_checks_passed"]: raise ValueError("environment and small calibration must pass first")
    g=observation_geometry(); adapter=OfficialCDL(); prov=adapter.provenance()
    save_csv("t15a_cdl_manifest.csv",manifest_rows())
    # At most 10 trajectories/100 frames in the batching scratch.
    stream=StreamingStatistics(g.selected,g.data,full_audit=True)
    frozen={}; snapshots={}; train_coordinates=[]; val_coordinates=[]; val_pilots=[]
    positions=[(r,q,k,t) for r in range(2) for q in range(2) for k in (0,32,63) for t in (0,8,15)]
    paths_power=None; delay=None; channel_energy=0.; channel_elements=0
    correlations={lag:[0j,0.,0.] for lag in (1,4,16,32,80)}
    starts=[]; ends=[]; continuity_pairs=[]; selected_hashes={}; batch=[]; batch_ids=[]
    train_start=time.perf_counter()
    for i in range(500):
        h,a,tau=adapter.generate(i,details=True); f=frames(h); x=f.reshape(10,-1)
        if i in (0,99,249,499): selected_hashes[str(i)]=array_hash(h)
        train_coordinates.append(np.stack([f[:,r,q,k,t] for r,q,k,t in positions],axis=1))
        path=np.sum(abs(a)**2,axis=(0,1,3))
        paths_power=path if paths_power is None else paths_power+path; delay=tau
        channel_energy+=float(np.sum(abs(h)**2)); channel_elements+=h.size
        starts.append(h[:,:,32,0]); ends.append(h[:,:,32,-1])
        continuity_pairs.append((h[:,:,32,15],h[:,:,32,16]))
        for lag,v in correlations.items():
            left=h[...,:-lag]; right=h[...,lag:]
            v[0]+=np.vdot(left,right); v[1]+=float(np.sum(abs(left)**2)); v[2]+=float(np.sum(abs(right)**2))
        batch.append(x); batch_ids.append(i)
        if len(batch)==10:
            stream.update(np.concatenate(batch),batch_ids); batch=[]; batch_ids=[]
        if i+1 in PREFIXES:
            fstat=stream.freeze(); frozen[i+1]=fstat
            path=OUT/f"t15a_cdl_statistics_{i+1}.npz"
            np.savez_compressed(path,mean_p=fstat.mean_p,mean_d=fstat.mean_d,rpp=fstat.rpp,rdp=fstat.rdp,
                train_trajectory_ids=fstat.train_trajectory_ids,samples=fstat.samples)
            snapshots[str(i+1)]=dict(path=str(path.relative_to(ROOT)).replace("\\","/"),sha256=sha(path),
                train_trajectory_count=i+1,correlated_frame_count=fstat.samples,bytes=path.stat().st_size,
                rpp_hermitian_residual=float(np.linalg.norm(fstat.rpp-fstat.rpp.conj().T)/np.linalg.norm(fstat.rpp)))
        if (i+1)%50==0: print(f"train statistics: {i+1}/500 trajectories",flush=True)
    training_seconds=time.perf_counter()-train_start
    separability=marginal_kronecker_audit(stream.mfull/stream.n)
    marginal_variance=float(np.mean(abs(stream.mean)**2))
    np.savez_compressed(OUT/"t15a_cdl_marginal_covariances.npz",spatial=separability["spatial"],frequency=separability["frequency"],time=separability["time"])
    # Persist only small marginal matrices, not full covariance.
    separability.pop("frequency"); separability.pop("time")
    separability["marginal_covariance_path"]="outputs/t15a_cdl_marginal_covariances.npz"
    separability["full_covariance_shape"]=[4096,4096]
    del stream
    pp=paths_power/paths_power.sum(); mean_delay=float(pp@delay)
    empirical_rms=float(np.sqrt(pp@(delay-mean_delay)**2))
    nominal_p=np.array(prov["normalized_PDP_powers"]); nominal_tau=np.array(prov["scaled_cluster_delays_seconds"])
    nominal_mean=float(nominal_p@nominal_tau); nominal_rms=float(np.sqrt(nominal_p@(nominal_tau-nominal_mean)**2))
    first,second=zip(*continuity_pairs)
    calibration=dict(mean_channel_power=channel_energy/channel_elements,mean_component_energy=marginal_variance,
        nominal_RMS_delay_seconds=nominal_rms,configured_RMS_delay_seconds=100e-9,
        empirical_array_weighted_RMS_delay_seconds=empirical_rms,
        empirical_cluster_powers=pp,cluster_delays_seconds=delay,max_Doppler_hz=(30/3.6)*3.5e9/299792458,
        time_correlations={str(lag):v[0]/np.sqrt(v[1]*v[2]) for lag,v in correlations.items()},
        continuity_correlation=normalized_correlation(first,second),
        independent_trajectory_boundary_correlation=normalized_correlation(ends[:-1],starts[1:]),
        time_sampling_seconds=SYSTEM["symbol_duration_seconds"],one_call_states=160,frame_boundary_indices=[15,16],
        direct_continuous_slicing_passed=small["slicing_exact"],calibration_trajectories=500,
        interpretation="nominal PDP versus directional-array-weighted empirical PDP; no gain normalization or unity-power requirement",
        spatial_link_order=["rx0tx0","rx0tx1","rx1tx0","rx1tx1"],spatial_link_covariance=separability["spatial"])
    # Marginal frequency/time correlation matrices are derived train-only.
    with np.load(OUT/"t15a_cdl_marginal_covariances.npz") as cov:
        rf,rt,rs=cov["frequency"],cov["time"],cov["spatial"]
        calibration["frequency_correlation_selected"]={str(k):rf[32,k]/np.sqrt(rf[32,32].real*rf[k,k].real) for k in (0,16,31,33,48,63)}
        calibration["spatial_correlation_matrix"]=rs/np.sqrt(rs.diagonal().real[:,None]*rs.diagonal().real[None,:])
    estimators={(n,snr):frozen[n].prepare(noise_variance_from_snr(snr)) for n in PREFIXES for snr in SNRS}
    rows=[]; validation_start=time.perf_counter()
    for i in range(500,600):
        h=adapter.generate(i); f=frames(h)
        if i in (500,599): selected_hashes[str(i)]=array_hash(h)
        val_coordinates.append(np.stack([f[:,r,q,k,t] for r,q,k,t in positions],axis=1)); val_pilots.append(f.reshape(10,-1)[:,g.selected])
        accum={(snr,method):dict(error_energy=0.,truth_energy=0.,bit_errors=0,total_bits=0) for snr in SNRS for method in ("R0-Perfect-CSI","R3-LS-LIN","R10-100","R10-250","R10-500")}
        for j,truth in enumerate(f):
            seed=seeds(i,j); bits,symbols=generate_master_qpsk(bits_seeds=(seed["bits_seed_tx0"],seed["bits_seed_tx1"]))
            transmitted=assemble_pilot_data_frame(symbols,g.layout); noiseless=apply_frequency_mimo_channel(truth,transmitted)
            unit_noise=complex_gaussian_noise(noiseless.shape,1.,np.random.default_rng(seed["noise_seed"]))
            reference=extract_reference_data_bits(bits,g.layout.data_mask); truth_energy=float(np.sum(abs(truth.ravel()[g.data])**2))
            for snr in SNRS:
                sigma=noise_variance_from_snr(snr); y=noiseless+np.sqrt(sigma)*unit_noise
                sparse,linear=estimate_ls_linear_channel(y,g.layout)
                z=sparse.ravel()[g.selected]
                estimates={"R0-Perfect-CSI":truth,"R3-LS-LIN":linear}
                for n in PREFIXES:
                    estimate=np.full(4096,np.nan+1j*np.nan,np.complex128)
                    estimate[g.data]=estimators[n,snr].estimate(z)
                    estimates[f"R10-{n}"]=estimate.reshape(2,2,64,16)
                for method,estimate in estimates.items():
                    detected=detect_data_zf(y,estimate,g.layout.data_mask)*np.sqrt(2)
                    predicted=demodulate_data_bits(detected,g.layout.data_mask)
                    errors=sum(int(np.count_nonzero(a!=b)) for a,b in zip(predicted,reference))
                    entry=accum[snr,method]; entry["bit_errors"]+=errors; entry["total_bits"]+=sum(b.size for b in reference)
                    entry["truth_energy"]+=truth_energy
                    if method!="R0-Perfect-CSI": entry["error_energy"]+=float(np.sum(abs(estimate.ravel()[g.data]-truth.ravel()[g.data])**2))
        for (snr,method),entry in accum.items():
            rows.append(dict(trajectory_id=i,split="validation_dev",frames=10,dF=16,snr_db=snr,method=method,**entry,
                nmse_linear=entry["error_energy"]/entry["truth_energy"] if method!="R0-Perfect-CSI" else "",
                ber=entry["bit_errors"]/entry["total_bits"]))
        if (i-499)%10==0: print(f"validation: {i-499}/100 trajectories",flush=True)
    validation_seconds=time.perf_counter()-validation_start
    save_csv("t15a_cdl_trajectory_metrics.csv",rows)
    aggregate=[]
    for snr in SNRS:
        for method in ("R0-Perfect-CSI","R3-LS-LIN","R10-100","R10-250","R10-500"):
            subset=[r for r in rows if r["snr_db"]==snr and r["method"]==method]
            e=sum(r["error_energy"] for r in subset); den=sum(r["truth_energy"] for r in subset)
            err=sum(r["bit_errors"] for r in subset); bitcount=sum(r["total_bits"] for r in subset)
            nmse=e/den if method!="R0-Perfect-CSI" else None
            aggregate.append(dict(method=method,dF=16,pilot_ratio=float(g.layout.pilot_union.mean()),snr_db=snr,
                trajectories=100,frames_per_trajectory=10,nmse_linear=nmse,nmse_db=float(10*np.log10(nmse)) if nmse else None,
                trajectory_nmse_mean=float(np.mean([r["nmse_linear"] for r in subset])) if nmse else None,
                trajectory_nmse_std=float(np.std([r["nmse_linear"] for r in subset],ddof=1)) if nmse else None,
                error_energy=e,truth_energy=den,bit_errors=err,total_bits=bitcount,ber=err/bitcount,
                trajectory_ber_std=float(np.std([r["ber"] for r in subset],ddof=1))))
    save_csv("t15a_cdl_lmmse.csv",aggregate)
    distribution=marginal_audit(np.concatenate(train_coordinates),np.concatenate(val_coordinates),positions)
    save_csv("t15a_cdl_distribution_audit.csv",distribution)
    projection_rows,projection=projection_audit(frozen[500],np.concatenate(val_pilots))
    save_csv("t15a_cdl_projection_audit.csv",projection_rows)
    repeat={i:array_hash(adapter.generate(int(i)))==value for i,value in selected_hashes.items()}
    if not all(repeat.values()): raise ValueError("selected trajectory repeat hashes changed")
    audit=[]
    for snr in SNRS:
        values={n:next(r for r in aggregate if r["snr_db"]==snr and r["method"]==f"R10-{n}") for n in PREFIXES}
        audit.append(dict(snr_db=snr,nmse_linear={str(n):values[n]["nmse_linear"] for n in PREFIXES},
            nmse_db={str(n):values[n]["nmse_db"] for n in PREFIXES},
            delta_100_to_250_linear=values[250]["nmse_linear"]-values[100]["nmse_linear"],
            delta_250_to_500_linear=values[500]["nmse_linear"]-values[250]["nmse_linear"],
            delta_100_to_250_db=values[250]["nmse_db"]-values[100]["nmse_db"],
            delta_250_to_500_db=values[500]["nmse_db"]-values[250]["nmse_db"]))
    covariance_changes={f"{a}_to_{b}":{key:float(np.linalg.norm(getattr(frozen[b],key)-getattr(frozen[a],key))/np.linalg.norm(getattr(frozen[b],key))) for key in ("rpp","rdp")} for a,b in ((100,250),(250,500))}
    frozen_context=[]
    with (OUT/"t10d_lmmse_train_validation_scan.csv").open(encoding="utf8",newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["dF"])==16 and int(row["snr_db"]) in SNRS:
                frozen_context.append({key:row[key] for key in ("dF","snr_db","ls_nmse_db","train_nmse_db","oracle_nmse_db")})
    after=old_hashes(); unchanged=before==after
    calibration["selected_trajectory_hashes"]=selected_hashes; calibration["repeat_hash_checks"]=repeat
    save_json("t15a_cdl_calibration.json",calibration)
    results=dict(task="T15A",experiment_type=EXPERIMENT_TYPE,environment="outputs/t15a_cdl_environment.json",
        sionna_provenance=prov,system=SYSTEM,array_geometry=ARRAY,cdl_profile="CDL-C",
        manifest=dict(path="outputs/t15a_cdl_manifest.csv",namespace="T15A_CDL_DEV_v1",train_trajectories=500,
            validation_trajectories=100,frames_per_trajectory=10,split_before_frames=True,
            seed_policy="SeedSequence([20261530,15001,trajectory,frame,role]); channel roles frame=0 once/trajectory"),
        calibration=calibration,distribution_audit=dict(path="outputs/t15a_cdl_distribution_audit.csv",positions=positions,
            samples_per_position=1000,normalization_fit="train_dev only",joint_gaussianity_claimed=False),
        projection_audit=projection,separability_audit=separability,
        empirical_lmmse_definition=dict(method="R10-CDL-LMMSE",formula="mu_d + Rdp solve(Rpp + sigma2 I, z_p-mu_p)",
            solver="Cholesky / cho_solve; no inverse, jitter or clipping",rpp_shape=[64,64],rdp_shape=[3968,64],
            centered=True,covariance_divisor="N population",training_only=True,snapshots=snapshots),
        covariance_sample_size_audit=audit,covariance_prefix_relative_changes=covariance_changes,
        validation_results=aggregate,validation_rows=len(rows),
        cholesky_solve_relative_residuals=[v.solve_relative_residual for v in estimators.values()],
        cdl_pipeline_valid=True,empirical_lmmse_stable=None,empirical_lmmse_stability_assessment="descriptive_only",
        gaussian_marginal_diagnostics="descriptive_only; marginal/projection moments do not prove joint Gaussianity",
        kronecker_separability_residual=separability["relative_Frobenius_residual"],
        frozen_ID_TDL_context=dict(path="outputs/t10d_lmmse_train_validation_scan.csv",sha256=sha(OUT/"t10d_lmmse_train_validation_scan.csv"),
            rows=frozen_context,interpretation="different channel distributions; absolute NMSE does not rank difficulty or realism"),
        new_model_trained=False,proposed_method=False,formal_test_used=False,formal_test_sealed=True,
        historical_files_unchanged=unchanged,provenance=dict(T14B_commit=T14B_COMMIT,historical_before=before,historical_after=after,
            source_sha256={p:sha(ROOT/p) for p in SOURCES}),
        runtime=dict(training_statistics_seconds=training_seconds,validation_seconds=validation_seconds,
            scan_wall_seconds=time.perf_counter()-started,full_covariance_scratch_bytes=4096**2*16,
            batch_channel_scratch_bytes=100*4096*16,statistics_disk_bytes=sum(x["bytes"] for x in snapshots.values()),
            complete_raw_H_saved=False,neural_training_seconds=0),
        warnings=["Matplotlib default cache was denied during initial environment smoke; subsequent scripts use workspace cache.",
            "Triton not installed on Windows; optional Triton flop counting unavailable, CDL and torch CUDA smoke passed.",
            "Directional element patterns alter gain and effective PDP; fixed nominal transmit SNR is retained.",
            "Ten frames per trajectory are correlated; statistics unit is trajectory.",
            "Whitening numerical rank tolerance is predefined, not a Gaussianity decision threshold.",
            "Separability metric uses marginal factors, not a global best Kronecker fit.",
            "Validation development results, not final formal-test E1 results."],
        all_checks_passed=False,completion_state="numeric scan complete; full regression and figure QA pending")
    save_json("t15a_cdl_results.json",results)
    if not unchanged: raise ValueError("historical tracked numerical/source files changed")
    print(json.dumps(dict(scan_seconds=results["runtime"]["scan_wall_seconds"],validation_rows=len(rows),
        mean_channel_power=calibration["mean_channel_power"],sample_size_audit=audit),indent=2),flush=True)


def finalize():
    path=OUT/"t15a_cdl_results.json"; result=json.loads(path.read_text(encoding="utf8"))
    if old_hashes()!=result["provenance"]["historical_before"]: raise ValueError("historical files changed")
    current={p:sha(ROOT/p) for p in SOURCES}
    if current!=result["provenance"]["source_sha256"]: raise ValueError("run source changed; retain audit evidence before accepting")
    p=subprocess.run([str(ROOT/".venv/Scripts/python.exe"),"-m","pytest","-q"],cwd=ROOT,capture_output=True,text=True,encoding="utf8",errors="replace")
    (OUT/"t15a_pytest_full.txt").write_text(p.stdout+p.stderr,encoding="utf8")
    result["pytest"]=dict(command=".venv/Scripts/python.exe -m pytest -q",returncode=p.returncode,output=p.stdout+p.stderr)
    result["main_environment_final"]=check_main_unchanged()
    diff=subprocess.run(["git","diff","--check"],cwd=ROOT,capture_output=True,text=True)
    result["git_diff_check"]=dict(returncode=diff.returncode,output=diff.stdout+diff.stderr)
    if p.returncode or not result["main_environment_final"]["all_checks_passed"] or diff.returncode:
        save_json(path.name,result); raise ValueError("final checks failed; original numeric results preserved")
    result["figure_audit"]=json.loads((OUT/"t15a_cdl_figure_audit.json").read_text(encoding="utf8"))
    if not result["figure_audit"]["all_checks_passed"]: raise ValueError("figure audit must pass")
    result["all_checks_passed"]=True; result["completion_state"]="completed"
    result["artifact_sha256"]={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in OUT.glob("t15a_cdl_*") if p.is_file() and p.name!=path.name}
    save_json(path.name,result); print("T15A full regression, isolation, history, CSV figure audit PASS",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--finalize",action="store_true")
    if parser.parse_args().finalize: finalize()
    else: scan()
