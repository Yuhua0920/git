"""Small official CDL adapter calibration, without the development scan."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".pytest_cache/matplotlib-cdl"))
import json
import numpy as np
from src.cdl_benchmark import OfficialCDL, frames, array_hash, SYSTEM


def probe():
    import torch
    from sionna.phy.channel import cir_to_ofdm_channel
    adapter=OfficialCDL(); h=adapter.generate(0); repeated=adapter.generate(0)
    a=torch.empty((1,1,2,1,2,1,2),dtype=torch.complex128,device=adapter.device)
    for rx in range(2):
        for tx in range(2):
            for t in range(2): a[0,0,rx,0,tx,0,t]=100*rx+10*tx+t+1+1j*(rx-tx)
    tau=torch.tensor([[[[125e-9]]]],dtype=torch.float64,device=adapter.device)
    mapped=cir_to_ofdm_channel(adapter.frequencies,a,tau,normalize=False)[0,0,:,0].permute(0,1,3,2).cpu().numpy()
    frequency=np.arange(-32,32)*30000
    expected=a[0,0,:,0,:,0].cpu().numpy()[:,:,None,:]*np.exp(-2j*np.pi*frequency*125e-9)[None,None,:,None]
    mapping_error=float(np.max(np.abs(mapped-expected)))
    framed=frames(h)
    positions=adapter.array.ant_pos.cpu().numpy()
    separation=np.linalg.norm(positions[1]-positions[0])
    result=dict(task="T15A adapter small calibration",profile="CDL-C",provenance=adapter.provenance(),
        shape=list(h.shape),dtype=str(h.dtype),hash=array_hash(h),repeat_hash=array_hash(repeated),
        mapping_max_abs_error=mapping_error,frequency_exact=bool(np.array_equal(adapter.frequencies.cpu(),frequency)),
        slicing_exact=bool(np.array_equal(framed[0,...,15],h[...,15]) and np.array_equal(framed[1,...,0],h[...,16])),
        frames_shape=list(framed.shape),array_separation_wavelengths=float(separation/(299792458/3.5e9)),
        symbol_duration_seconds=SYSTEM["symbol_duration_seconds"],
        all_checks_passed=bool(array_hash(h)==array_hash(repeated) and mapping_error<1e-12 and abs(separation/(299792458/3.5e9)-.5)<1e-12))
    return result


if __name__=="__main__":
    r=probe()
    path=ROOT/"outputs/t15a_cdl_adapter_calibration.json"
    path.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps({k:v for k,v in r.items() if k!="provenance"},indent=2))
    if not r["all_checks_passed"]: raise SystemExit(1)
