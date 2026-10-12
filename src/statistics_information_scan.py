"""T13D correctness audits and deterministic information summaries only."""
from time import perf_counter

import numpy as np
from scipy.linalg import cholesky,solve_triangular

from src import statistics_information as info
from src.numerics import complex_gaussian_noise

PREFIX="t13d_statistics_information"
T13C_COMMIT="13e3853ceed7bc9dd211c862871cccc4095ccbc4"
SPEED_GRID=tuple(range(10,151,5)); DS_GRID=tuple(range(50,351,10))
MC_DRAWS=10000; AUDIT_SEED=20261304; FD_STEPS=(1e-3,3e-4,1e-4)
TIMING_KEYS=("covariance_construction_seconds","derivatives_seconds","FIM_seconds","KL_grid_seconds","Monte_Carlo_audit_seconds")


def derivative_validation():
    records=[]; convergence=[]
    for point,speed,ds in info.POINTS:
        theta=info.theta_from_physical(speed,ds)
        for window in info.WINDOWS:
            rf,rt,drf,drt=info.factors(*theta,window)
            for parameter,analytic,position in (("fD",drt,0),("tau_rms",drf,1)):
                errors=[]
                for step in FD_STEPS:
                    high=theta.copy(); low=theta.copy(); high[position]*=1+step; low[position]*=1-step
                    a=info.factors(*high,window,derivatives=False)[1-position]
                    b=info.factors(*low,window,derivatives=False)[1-position]
                    finite=(a-b)/(2*theta[position]*step)
                    error=float(np.max(np.abs(finite-analytic)))
                    relative=float(np.linalg.norm(finite-analytic)/np.linalg.norm(analytic)); errors.append(relative)
                    records.append({"point":point,"window_frames":window,"parameter":parameter,"relative_step":step,
                        "max_absolute_error":error,"relative_Frobenius_error":relative})
                passed=errors[0]>errors[1]>errors[2] and errors[-1]<2e-6
                convergence.append({"point":point,"window_frames":window,"parameter":parameter,
                    "relative_errors":errors,"all_checks_passed":bool(passed)})
    if not all(r["all_checks_passed"] for r in convergence):
        raise ValueError("analytic derivative finite-difference convergence failed; STOP")
    return {"steps":list(FD_STEPS),"records":records,"convergence":convergence,"production_derivatives":"analytic only","all_checks_passed":True}


def score_monte_carlo_audit(draws=MC_DRAWS,seed=AUDIT_SEED):
    theta=info.theta_from_physical(30,100); matrices=[]; j=np.zeros((2,2)); traces=[]; link_models=[]
    for nr in range(2):
        for tx in range(2):
            g=info.geometry(16,5,tx); c,dc=info.observation_covariance(theta,g,.1)
            model=info.prepare_covariance(c); a=info.whiten_derivatives(model,dc*theta[:,None,None])
            matrices.append(a); traces.append(np.trace(a,axis1=1,axis2=2).real)
            j+=info.fisher_link(model,dc*theta[:,None,None]); link_models.append(model)
    # Rows are independent joint four-link samples. Whitened expression is
    # exactly the requested physical score after division by theta.
    scores=np.zeros((draws,2)); rng=np.random.default_rng(seed)
    for start in range(0,draws,250):
        n=min(250,draws-start)
        for a,trace in zip(matrices,traces):
            w=complex_gaussian_noise((80,n),1.,rng,dtype=np.complex128)
            for i in range(2):
                quadratic=np.sum(w.conj()*(a[i]@w),axis=0)
                if np.max(np.abs(quadratic.imag))>1e-10:
                    raise ValueError("proper-complex synthetic score is not real")
                scores[start:start+n,i]+=quadratic.real-trace[i]
    mean=scores.mean(0); covariance=np.cov(scores,rowvar=False,ddof=1)
    fj=cholesky(j,lower=True); normalized=solve_triangular(fj,scores.T,lower=True).T
    whitened_cov=np.cov(normalized,rowvar=False,ddof=1)
    error=float(np.linalg.norm(whitened_cov-np.eye(2))/np.sqrt(2))
    standardized_mean=mean/np.sqrt(np.diag(j)/draws)
    if np.max(np.abs(standardized_mean))>5 or error>.08:
        raise ValueError("Gaussian-score MC audit failed predeclared five-SE/8percent gates; no seed retry")
    return {"point":"P0","dF":16,"window_frames":5,"snr_db":10,"draws":draws,"independent_links_per_draw":4,
        "audit_seed":seed,"seed_role":"synthetic score audit only; no channel/bits/noise/training experiment seed consumed",
        "sampling":"w~CN(0,I), z=Lw; score_log=w^H(L^-1 C_log L^-H)w-tr(L^-1 C_log L^-H)",
        "score_mean_log_parameters":mean.tolist(),"score_mean_physical_parameters":(mean/theta).tolist(),
        "score_mean_in_analytic_standard_errors":standardized_mean.tolist(),"analytic_FIM_log":j.tolist(),
        "empirical_score_covariance_log":covariance.tolist(),"analytic_FIM_physical":(j/theta[:,None]/theta[None,:]).tolist(),
        "empirical_score_covariance_physical":(covariance/theta[:,None]/theta[None,:]).tolist(),
        "FIM_whitened_covariance":whitened_cov.tolist(),"FIM_whitened_covariance_relative_error":error,
        "maximum_mean_SE_allowed":5,"maximum_covariance_relative_error_allowed":.08,"all_checks_passed":True}


def comparisons(rows):
    expected={(p,d,w,s) for p,_,_ in info.POINTS for d in info.DFS for w in info.WINDOWS for s in info.SNRS}
    index={(r["point"],r["dF"],r["window_frames"],r["snr_db"]):r for r in rows}
    if len(rows)!=48 or set(index)!=expected:
        raise ValueError("all48 theoretical cases required")
    density=[]; window=[]
    for p,_,_ in info.POINTS:
        for s in info.SNRS:
            for w in info.WINDOWS:
                a,b=index[p,16,w,s],index[p,8,w,s]
                density.append({"point":p,"snr_db":s,"window_frames":w,
                    "speed_std_ratio_dF16_over_dF8":a["speed_std_kmh"]/b["speed_std_kmh"],
                    "delay_std_ratio_dF16_over_dF8":a["delay_std_ns"]/b["delay_std_ns"]})
            for d in info.DFS:
                a,b=index[p,d,1,s],index[p,d,5,s]
                window.append({"point":p,"snr_db":s,"dF":d,
                    "speed_std_ratio_1frame_over_5frames":a["speed_std_kmh"]/b["speed_std_kmh"],
                    "delay_std_ratio_1frame_over_5frames":a["delay_std_ns"]/b["delay_std_ns"]})
    return density,window


def kl_grid(timing,progress=None):
    rows=[]; references={}; geometries={w:tuple(info.geometry(16,w,q) for q in range(2)) for w in info.WINDOWS}
    for w in info.WINDOWS:
        for s in info.SNRS:
            for p,v,ds in info.POINTS:
                pair=tuple(info.prepare_covariance(info.observation_covariance(info.theta_from_physical(v,ds),g,10**(-s/10),derivatives=False,timing=timing)) for g in geometries[w])
                references[w,s,p]=pair
    for w in info.WINDOWS:
        for s in info.SNRS:
            stamp=perf_counter()
            for v in SPEED_GRID:
                for ds in DS_GRID:
                    pair=tuple(info.prepare_covariance(info.observation_covariance(info.theta_from_physical(v,ds),g,10**(-s/10),derivatives=False,timing=timing)) for g in geometries[w])
                    for p,v0,ds0 in info.POINTS:
                        a=references[w,s,p]
                        # Compute both TX terms from their own mask before summing
                        # their two independent RX copies. No spatial covariance.
                        value,per_link=info.kl_four_links((a[0],a[1],a[0],a[1]),(pair[0],pair[1],pair[0],pair[1]))
                        rows.append({"reference_point":p,"window_frames":w,"snr_db":s,"candidate_speed":v,"candidate_ds":ds,
                            "KL_total":value,"nonlocal_flag":abs(v-v0)>=20 or abs(ds-ds0)>=50,
                            **{f"KL_H{nr}{tx}":per_link[2*nr+tx] for nr in range(2) for tx in range(2)}})
            timing["KL_grid_seconds"]+=perf_counter()-stamp
            if progress:
                progress(f"KL grid window{w}, SNR{s}: 899 candidates x4 references complete")
    # KL_grid_seconds includes its covariance construction; report this overlap
    # explicitly rather than presenting the timing buckets as disjoint.
    return rows


def global_summaries(rows):
    summaries=[]; minima=[]
    speeds=np.array(SPEED_GRID); delays=np.array(DS_GRID)
    for p,v0,ds0 in info.POINTS:
        for w in info.WINDOWS:
            for s in info.SNRS:
                part=[r for r in rows if (r["reference_point"],r["window_frames"],r["snr_db"])==(p,w,s)]
                expected={(v,d) for v in SPEED_GRID for d in DS_GRID}
                if len(part)!=899 or {(r["candidate_speed"],r["candidate_ds"]) for r in part}!=expected:
                    raise ValueError("complete899-cell KL grid required")
                matrix=np.empty((29,31)); nonlocal_rows=[]
                for r in part:
                    matrix[list(SPEED_GRID).index(r["candidate_speed"]),list(DS_GRID).index(r["candidate_ds"])]=r["KL_total"]
                    expected_nonlocal=abs(r["candidate_speed"]-v0)>=20 or abs(r["candidate_ds"]-ds0)>=50
                    if r["nonlocal_flag"]!=expected_nonlocal or r["KL_total"] < -4e-9:
                        raise ValueError("KL nonlocal definition or numerical nonnegativity failed")
                    if r["nonlocal_flag"]:
                        nonlocal_rows.append(r)
                same=next(r for r in part if r["candidate_speed"]==v0 and r["candidate_ds"]==ds0)
                if abs(same["KL_total"])>4e-9:
                    raise ValueError("KL identity failed")
                for i,v in enumerate(speeds):
                    for j,d in enumerate(delays):
                        neighbours=matrix[max(0,i-1):min(29,i+2),max(0,j-1):min(31,j+2)].copy().ravel()
                        # True8-neighbour local minima, including boundary cells.
                        if matrix[i,j]<=neighbours.min() and np.any(matrix[i,j]<neighbours):
                            minima.append({"reference_point":p,"window_frames":w,"snr_db":s,"candidate_speed":int(v),"candidate_ds":int(d),
                                "KL_total":float(matrix[i,j]),"nonlocal_flag":abs(v-v0)>=20 or abs(d-ds0)>=50,
                                "boundary_cell":bool(i in (0,28) or j in (0,30))})
                worst=min(nonlocal_rows,key=lambda r:r["KL_total"])
                basins=[r for r in minima if (r["reference_point"],r["window_frames"],r["snr_db"])==(p,w,s)]
                summaries.append({"reference_point":p,"window_frames":w,"snr_db":s,"minimum_nonlocal_KL":worst["KL_total"],
                    "argmin_speed":worst["candidate_speed"],"argmin_tau_rms_ns":worst["candidate_ds"],"KL_at_reference":same["KL_total"],
                    "grid_minimum_KL":float(matrix.min()),"local_minima":basins,
                    "additional_nonlocal_local_minima":sum(r["nonlocal_flag"] for r in basins),
                    "arbitrary_low_KL_threshold_used":False,"scope":"finite grid only; a secondary minimum alone does not establish low-KL confusability"})
    return summaries,minima


def scientific_flags(rows,global_ambiguity=None):
    key=[r for r in rows if r["dF"]==16 and r["window_frames"]==5 and r["snr_db"] in (10,20)]
    if len(key)!=8:
        raise ValueError("eight prescribed key cases required")
    local_strong=all(r["relative_speed_std_percent"]<=10 and r["relative_delay_std_percent"]<=10 for r in key)
    limited={parameter:sorted({r["point"] for r in key if r[f"relative_{parameter}_std_percent"]>=25}) for parameter in ("speed","delay")}
    multiple=any(len(points)>=2 for points in limited.values())
    if global_ambiguity is None:
        return {"continuous_statistics_locally_well_informed":None,"continuous_statistics_information_limited":None,
            "partial_information_limit":None,"local_information_strong_but_global_ambiguity_present":None,
            "local_eight_case_threshold_all_satisfied":local_strong,"points_with_relative_std_ge25_percent":limited,
            "scientific_decision_pending_global_review":True}
    well=local_strong and not global_ambiguity; weak=multiple or global_ambiguity
    return {"continuous_statistics_locally_well_informed":bool(well),"continuous_statistics_information_limited":bool(weak),
        "partial_information_limit":bool(not well and not weak),"local_information_strong_but_global_ambiguity_present":bool(local_strong and global_ambiguity),
        "local_eight_case_threshold_all_satisfied":local_strong,"points_with_relative_std_ge25_percent":limited,
        "scientific_decision_pending_global_review":False}
