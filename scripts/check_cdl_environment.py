"""T15A isolated dependency/provenance checks; never installs into the main venv."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs"
T14B_COMMIT = "f3c1d058fed3592cdb408038e0539a14b521489d"


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024), b""): h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf8")


def environment_fingerprint(directory):
    directory = Path(directory)
    files = {}
    for p in sorted(directory.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc",".pyo"):
            files[str(p.relative_to(directory)).replace("\\","/")] = file_hash(p)
    canonical = json.dumps(files, sort_keys=True, separators=(",",":")).encode()
    return {"root":str(directory), "files":files, "sha256":hashlib.sha256(canonical).hexdigest(),
            "excluded":"runtime __pycache__, pyc/pyo only; all package payload/metadata/interpreters included"}


def snapshot_main():
    if Path(sys.prefix).resolve() != (ROOT/".venv").resolve(): raise ValueError("snapshot must use original main .venv")
    path = OUT/"t15a_main_environment_before.json"
    if path.exists(): raise ValueError("existing main snapshot protected")
    r = {"python":sys.version,"executable":sys.executable,"base_prefix":sys.base_prefix,
        "freeze":subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True),
        "fingerprint":environment_fingerprint(ROOT/".venv")}
    save_json(path,r); print("Main environment payload snapshot:",r["fingerprint"]["sha256"],flush=True)


def check_main_unchanged():
    old = json.loads((OUT/"t15a_main_environment_before.json").read_text(encoding="utf8"))
    current = environment_fingerprint(ROOT/".venv")
    freeze = subprocess.check_output([str(ROOT/".venv/Scripts/python.exe"),"-m","pip","freeze"],text=True)
    return {"before_sha256":old["fingerprint"]["sha256"],"after_sha256":current["sha256"],
            "payload_unchanged":old["fingerprint"]["files"]==current["files"],"freeze_unchanged":freeze==old["freeze"],
            "all_checks_passed":old["fingerprint"]["files"]==current["files"] and freeze==old["freeze"]}


def inspect_cdl():
    path = OUT/"t15a_cdl_environment.json"
    r = {"task":"T15A","status":"checking", "python":sys.version,"platform":platform.platform(),
        "executable":sys.executable,"prefix":sys.prefix,"base_prefix":sys.base_prefix,
        "expected_sionna_version":"2.2.0", "expected_distribution":"sionna-no-rt",
        "T14B_commit":T14B_COMMIT,"formal_test_used":False,"formal_test_sealed":True,
        "all_checks_passed":False,"errors":[]}
    try:
        if Path(sys.prefix).resolve()!=(ROOT/".venv-cdl").resolve(): raise ValueError("isolated .venv-cdl required")
        if sys.version_info[:2]!=(3,13): raise ValueError("same Python3.13 required")
        cfg = (ROOT/".venv-cdl/pyvenv.cfg").read_text(encoding="utf8")
        if "include-system-site-packages = false" not in cfg: raise ValueError("system site packages must not be shared")
        versions = {d.metadata["Name"]:d.version for d in metadata.distributions()}
        forbidden = [n for n in versions if n.lower().startswith(("tensorflow","sionna-rt","mitsuba","drjit"))]
        if forbidden: raise ValueError("forbidden dependencies: "+str(forbidden))
        if metadata.version("sionna-no-rt")!="2.2.0": raise ValueError("Sionna version mismatch")
        import torch
        import sionna
        from sionna.phy import config
        from sionna.phy.channel.tr38901 import CDL, PanelArray
        import inspect
        r.update({"torch":torch.__version__,"cuda_version":torch.version.cuda,"cuda_available":torch.cuda.is_available(),
            "device":"cuda:0" if torch.cuda.is_available() else "cpu", "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "sionna_version":metadata.version("sionna-no-rt"), "sionna_module":str(Path(sionna.__file__).resolve()),
            "CDL_signature":str(inspect.signature(CDL)),"PanelArray_signature":str(inspect.signature(PanelArray)),
            "installed_versions":dict(sorted(versions.items())),"forbidden_dependencies":forbidden})
        device = r["device"]
        config.seed = 20261501; torch.manual_seed(20261502)
        array = PanelArray(num_rows_per_panel=1,num_cols_per_panel=2,polarization="single",polarization_type="V",
            antenna_pattern="38.901",carrier_frequency=3.5e9,element_horizontal_spacing=.5,element_vertical_spacing=.5,
            precision="double",device=device)
        cdl = CDL(model="C",delay_spread=100e-9,carrier_frequency=3.5e9,ut_array=array,bs_array=array,
            direction="downlink",ut_orientation=[torch.pi,0.,0.],bs_orientation=[0.,0.,0.],ut_velocity=[30/3.6,0.,0.],
            min_speed=30/3.6,max_speed=30/3.6,normalize_delays=True,precision="double",device=device,spec_version="19.2")
        with torch.no_grad(): a,tau = cdl(batch_size=1,num_time_steps=16,sampling_frequency=1920000/80)
        if a.shape[:5]!=(1,1,2,1,2) or a.shape[-1]!=16 or not torch.isfinite(a).all(): raise ValueError("CDL-C smoke output invalid")
        r["smoke"]={"a_shape":list(a.shape),"tau_shape":list(tau.shape),"a_dtype":str(a.dtype),"tau_dtype":str(tau.dtype),
            "sampling_frequency_hz":24000.,"spec_version":cdl.spec_version,"finite":True,"passed":True}
        sources = [Path(inspect.getfile(CDL)),Path(inspect.getfile(PanelArray))]
        r["sionna_source_hashes"] = {str(p):file_hash(p) for p in sources}
        r["package_metadata_hashes"]={}
        for d in metadata.distributions():
            records = [p for p in (d.files or []) if str(p).endswith(("/RECORD","/METADATA","/WHEEL"))]
            r["package_metadata_hashes"][d.metadata["Name"]] = {str(p):file_hash(d.locate_file(p)) for p in records}
        freeze = subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
        (OUT/"t15a_cdl_packages_freeze.txt").write_text(freeze,encoding="utf8")
        r["freeze_sha256"] = file_hash(OUT/"t15a_cdl_packages_freeze.txt")
        r["isolated_payload_fingerprint"] = environment_fingerprint(ROOT/".venv-cdl")
        r["main_environment"] = check_main_unchanged()
        if not r["main_environment"]["all_checks_passed"]: raise ValueError("main environment changed")
        r["status"]="passed"; r["all_checks_passed"]=True
    except Exception as exc:
        r["status"]="failed"; r["errors"].append({"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
        save_json(path,r); raise
    save_json(path,r); print("Sionna2.2 CDL-C smoke and environment isolation PASS",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--snapshot-main",action="store_true")
    args=parser.parse_args()
    if args.snapshot_main: snapshot_main()
    else: inspect_cdl()


if __name__=="__main__": main()
