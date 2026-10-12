"""T15B author package gate; run only in .venv-gmm, without generating CDL."""
import hashlib
import importlib.metadata as md
import inspect
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/"outputs"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_formula_gate():
    import numpy as np
    from gmm_estimator import GmmEstimator
    weights=np.array([.4,.6])
    means=np.array([[.2+.1j,-.3+.4j],[-.4+.2j,.7-.1j]])
    cov=np.array([[[2,.3+.2j],[.3-.2j,1]],[[1,.1-.3j],[.1+.3j,1.5]]],dtype=np.complex128)
    model=GmmEstimator(n_components=2,covariance_type="full")
    model.weights_=weights; model.means_=means; model.covariances_=cov;model.n_features_in_=2
    y=np.array([[.1+.15j],[.6-.25j]],dtype=np.complex128)
    sigma=.7;A=np.array([[1,0]],dtype=np.complex128)
    manual=[];posteriors=[]
    for sample in y[:,0]:
        variance=cov[:,0,0].real+sigma
        density=weights*np.exp(-abs(sample-means[:,0])**2/variance)/(np.pi*variance)
        posterior=density/density.sum();posteriors.append(posterior)
        conditional=means+cov[:,:,0]*(sample-means[:,0])[:,None]/variance[:,None]
        manual.append((posterior[:,None]*conditional).sum(axis=0))
    actual=model.estimate(y,np.array([[sigma]],complex),A,n_components_or_probability=1.0)
    error=float(np.max(abs(actual-np.asarray(manual))))
    hard=model.estimate(y,np.array([[sigma]],complex),A,n_components_or_probability=1)
    if error>1e-12 or np.allclose(actual,hard,atol=1e-6):
        raise ValueError("STOP: package mixture conditional-mean formula gate failed")
    # Required production dimensions: rectangular 64x128, genuinely complex full covariance.
    rng=np.random.default_rng(20261600)
    v=(rng.normal(size=(128,5))+1j*rng.normal(size=(128,5)))/10
    full=np.eye(128)+v@v.conj().T
    model=GmmEstimator(n_components=1,covariance_type="full")
    model.n_features_in_=128;model.weights_=np.ones(1)
    model.means_=np.full((1,128),.1+.2j);model.covariances_=full[None]
    z=np.full((2,64),.3-.1j);selection=np.eye(128,dtype=complex)[:64]
    prediction=model.estimate(z,.7*np.eye(64),selection,n_components_or_probability=1.0)
    if prediction.shape!=(2,128) or not np.isfinite(prediction).all() or not np.iscomplexobj(prediction):
        raise ValueError("STOP: rectangular complex observation API unavailable")
    return dict(manual_two_component_max_abs_error=error,tolerance=1e-12,
                posterior_probabilities=np.asarray(posteriors).tolist(),soft_mixture_verified=True,
                rectangular_observation_shape=[64,128],output_shape=list(prediction.shape),all_checks_passed=True)


def main():
    report=dict(task="T15B",python=sys.version,executable=sys.executable,prefix=sys.prefix,
                isolated_environment=False,all_checks_passed=False,errors=[])
    try:
        if Path(sys.prefix).resolve()!=(ROOT/".venv-gmm").resolve() or sys.version_info[:2]!=(3,13):
            raise ValueError("isolated Python3.13 .venv-gmm required")
        if 'include-system-site-packages = false' not in (ROOT/".venv-gmm/pyvenv.cfg").read_text(encoding="utf8"):
            raise ValueError("isolation disabled")
        from gmm_estimator import GmmEstimator
        from cplx_gmm import GaussianMixtureCplx
        report["isolated_environment"]=True
        packages={}
        for name,version in (("cplx-gmm","0.2.0"),("gmm-estimator","0.1.0")):
            d=md.distribution(name)
            if d.version!=version: raise ValueError("package version mismatch")
            files={str(p):sha(d.locate_file(p)) for p in d.files if str(p).endswith((".py","METADATA","LICENSE","RECORD"))}
            packages[name]=dict(version=d.version,license=d.metadata.get("License-Expression"),
                                project_urls=d.metadata.get_all("Project-URL"),dependencies=d.requires,
                                source_and_metadata_sha256=files)
        report["package_provenance"]=packages
        report["installed_versions"]={d.metadata["Name"]:d.version for d in md.distributions()}
        report["constructor_signature"]=str(inspect.signature(GmmEstimator))
        report["complex_parent_class"]=f"{GaussianMixtureCplx.__module__}.{GaussianMixtureCplx.__name__}"
        report["package_formula_gate"]=package_formula_gate()
        freeze=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
        (OUT/"t15b_gmm_packages_freeze.txt").write_text(freeze,encoding="utf8")
        report["freeze_sha256"]=sha(OUT/"t15b_gmm_packages_freeze.txt")
        report["all_checks_passed"]=True
    except Exception as exc:
        report["errors"].append(dict(type=type(exc).__name__,message=str(exc)))
        (OUT/"t15b_gmm_environment.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
        raise
    (OUT/"t15b_gmm_environment.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps({k:v for k,v in report.items() if k in ("python","isolated_environment","package_formula_gate","all_checks_passed")},indent=2),flush=True)


if __name__=="__main__": main()
