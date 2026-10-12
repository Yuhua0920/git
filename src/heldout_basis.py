"""T16C dedicated CDL namespaces and load-only historical correction bases."""
from dataclasses import dataclass
import copy
import json
from pathlib import Path
import numpy as np
from src.cdl_benchmark import OfficialCDL, ARRAY, SYSTEM, SYMBOL_SECONDS, array_hash
from src.lowrank_filter import readonly
from src.cdl_mismatch import SNRS

T16B_COMMIT="8f0c3a5764af3a9a0329aaa21d62ce788bd419ba"
HELDOUT={"CDL-C-75":dict(profile="C",speed_kmh=75.,namespace_code=16101),
         "CDL-B-120":dict(profile="B",speed_kmh=120.,namespace_code=16102)}
RANKS=(4,8,16,24)
EXPERIMENT_TYPE="held-out representation-generalization gate for a frozen shared low-rank LMMSE mismatch-correction subspace"


def definition(name):
    d=HELDOUT[name]; system=copy.deepcopy(SYSTEM); array=copy.deepcopy(ARRAY)
    system["speed_kmh"]=d["speed_kmh"]; array["ut_velocity_mps"]=[d["speed_kmh"]/3.6,0.,0.]
    return dict(profile=d["profile"],system=system,array=array,dF=16,spacing_time=4,
        pilot_ratio=1/32,ofdm_channel_normalize=False,namespace=f"T16C_{name}_HELDOUT_v1",
        diagnostic="mobility interpolation" if name=="CDL-C-75" else "composed profile/mobility shift")


def seeds(name,split,trajectory,frame=0):
    limits={"train_dev":range(500),"validation_dev":range(500,600)}
    if name not in HELDOUT or split not in limits or trajectory not in limits[split] or frame not in range(10):
        raise ValueError("dedicated T16C train/validation only; formal test sealed")
    roles=("sionna_seed","torch_seed","numpy_seed","bits_seed_tx0","bits_seed_tx1","noise_seed")
    code=1 if split=="train_dev" else 2
    return {r:int(np.random.SeedSequence([20261630,HELDOUT[name]["namespace_code"],code,
        trajectory,0 if j<=3 else frame,j]).generate_state(1)[0]) for j,r in enumerate(roles,1)}


def manifest_rows():
    return [dict(shift=name,namespace=definition(name)["namespace"],split=split,trajectory_id=i,
        frame_id=j,start_symbol=16*j,end_symbol=16*j+15,channel_generation_scope="trajectory_frame0",
        communication_seeds_consumed=split=="validation_dev",**seeds(name,split,i,j))
        for name in HELDOUT for split,ids in (("train_dev",range(500)),("validation_dev",range(500,600)))
        for i in ids for j in range(10)]


class HeldoutCDL(OfficialCDL):
    def __init__(self,name,device=None):
        super().__init__(device)
        from sionna.phy.channel.tr38901 import CDL
        self.name=name; d=definition(name)
        self.model=CDL(model=d["profile"],delay_spread=100e-9,carrier_frequency=3.5e9,
            ut_array=self.array,bs_array=self.array,direction="downlink",
            ut_orientation=ARRAY["ut_orientation_radians"],bs_orientation=ARRAY["bs_orientation_radians"],
            ut_velocity=d["array"]["ut_velocity_mps"],min_speed=d["system"]["speed_kmh"]/3.6,
            max_speed=d["system"]["speed_kmh"]/3.6,normalize_delays=True,
            precision="double",device=self.device,spec_version="19.2")

    def generate(self,trajectory,*,split):
        import torch
        from sionna.phy import config
        from sionna.phy.channel import cir_to_ofdm_channel
        s=seeds(self.name,split,trajectory)
        config.seed=s["sionna_seed"]; torch.manual_seed(s["torch_seed"])
        self.numpy_rng=np.random.default_rng(s["numpy_seed"])
        with torch.no_grad():
            a,tau=self.model(batch_size=1,num_time_steps=160,sampling_frequency=1/SYMBOL_SECONDS)
            hf=cir_to_ofdm_channel(self.frequencies,a,tau,normalize=False)
        h=hf[0,0,:,0,:,:,:].permute(0,1,3,2).cpu().numpy().astype(np.complex128,copy=True)
        if h.shape!=(2,2,64,160) or not np.isfinite(h).all(): raise ValueError("invalid heldout CDL export")
        return h

    def provenance(self):
        import hashlib
        from sionna.phy.channel.tr38901 import models
        p=super().provenance()
        p["source_hashes"]={k:v for k,v in p["source_hashes"].items() if not k.endswith("CDL-C.json")}
        f=Path(models.parameter_file(f"CDL-{HELDOUT[self.name]['profile']}.json","19.2"))
        p["source_hashes"][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
        p["profile"]=f"CDL-{HELDOUT[self.name]['profile']}"; p["heldout_definition"]=definition(self.name)
        return p


@dataclass(frozen=True)
class FrozenOutputBasis:
    u32: np.ndarray
    scope: str

    def output(self,q):
        if q not in RANKS: raise ValueError("declared frozen ranks only")
        return self.u32[:,:q]

    def coefficient_map(self,delta,q): return readonly(self.output(q).conj().T@delta)

    def estimate(self,pilot,ws,mean_p,mean_d,q,g):
        x=np.asarray(pilot,np.complex128)-mean_p
        return mean_d+x@ws.T+(x@g.T)@self.output(q).T

    def energy(self,m,q):
        u=self.output(q); p=u.conj().T@m; total=float(np.vdot(m,m).real)
        residual=m-u@p
        return dict(capture=float(np.vdot(p,p).real/total),weighted_excess_MSE_per_frame=float(np.vdot(residual,residual).real),
            correction_energy_per_frame=total)


def load_bases(root):
    """No historical construction import/path or SVD; verify file AND array bytes."""
    import hashlib
    root=Path(root); proof=json.loads((root/"outputs/t16b_basis_provenance_recovery.json").read_text(encoding="utf8"))
    old=json.loads((root/"outputs/t16b_shared_basis_results.json").read_text(encoding="utf8"))
    if not proof["historical_u32_hash_exact_match"] or proof["historical_T16B_commit"]!=T16B_COMMIT:
        raise ValueError("historical exact recovery gate absent")
    if hashlib.sha256((root/"outputs/t16b_shared_basis_results.json").read_bytes()).hexdigest()!=proof["historical_results_sha256"]:
        raise ValueError("historical T16B JSON changed")
    bases={}
    for scope,info in proof["basis_artifacts"].items():
        p=root/info["path"]
        if hashlib.sha256(p.read_bytes()).hexdigest()!=info["file_sha256"]: raise ValueError("frozen artifact file hash drift")
        with np.load(p,allow_pickle=False) as f:
            u=f["U32"]
            if u.shape!=(3968,32) or u.dtype!=np.complex128 or array_hash(u)!=info["U32_sha256"]:
                raise ValueError("frozen U32 shape/dtype/hash drift")
            if str(f["source_commit"])!=T16B_COMMIT or str(f["construction_protocol_hash"])!=proof["construction_protocol_hash"]:
                raise ValueError("frozen basis snapshot/protocol drift")
            if json.loads(str(f["construction_condition_ids"]))!=old["shared_basis_definition"][scope]["basis_conditions"]:
                raise ValueError("basis condition IDs drift")
            if array_hash(u)!=old["shared_basis_definition"][scope]["U32_sha256"]: raise ValueError("historical U32 hash drift")
        u.flags.writeable=False; bases[scope]=FrozenOutputBasis(u,scope)
        if any(array_hash(bases[scope].output(q))!=info["Uq_sha256"][str(q)] for q in RANKS): raise ValueError("Uq hash drift")
    return bases,{**proof,"t16c_basis_loaded_from_frozen_artifact":True,"formal_basis_construction_SVD_calls":0}


def gate(shared,oracle):
    focus=[r for r in shared if r["snr_db"] in (10,20)]
    controls=[r for r in oracle if r["snr_db"] in (10,20) and r["rank"]==4]
    if len(focus)!=16 or len(controls)!=4: raise ValueError("complete prespecified heldout controls required")
    def passes(q): return all(r["closure"] is not None and r["closure"]>=.8 for r in focus if r["rank"]==q)
    strong=passes(8); moderate=not strong and passes(16); weak=not passes(24)
    return dict(heldout_shared_basis_strong=strong,heldout_shared_basis_moderate=moderate,
        heldout_shared_basis_weak=weak,heldout_correction_intrinsically_lowrank=all(r["closure"] is not None and r["closure"]>=.9 for r in controls))


def qthreshold(rows,threshold):
    if any(r["rank"] not in RANKS or r.get("metric_type")!="actual_validation" for r in rows):
        raise ValueError("actual measured shared ranks required")
    candidates=[r["rank"] for r in rows if r["closure"] is not None and r["closure"]>=threshold]
    return min(candidates) if candidates else None


def gap_closure(source_db,matched_db,estimate_db):
    gap=float(source_db-matched_db)
    # Keep small/nonpositive gaps and conditions; do not replace experimental data.
    return dict(full_mismatch_gap_db=gap,recovery_db=float(source_db-estimate_db),
        closure=float((source_db-estimate_db)/gap) if gap!=0 else None,
        closure_defined=gap!=0)
