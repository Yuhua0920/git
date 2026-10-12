"""Prepared next-host SAFE synthetic smoke. Not executed during R4.

No real preparation/validation input, no full-N training, no cap change.
The script uses fresh host-specific output directories and the archived monitor.
"""
from pathlib import Path
import sys,argparse,copy
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u12b5b_r4_common import firewall,check_frozen,read,write,LIMITS
from scripts import run_sp_u12b5b_r4 as runner

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args();firewall();check_frozen()
    out=Path(args.out).resolve();assert out.is_relative_to(ROOT/'outputs/sp_u12b5b_r4')and not out.exists()
    sys.path.insert(0,str(ROOT/'experiments/mfa_low_memory_prototype/portable'))
    from host_inventory import inventory
    host=inventory(False);assert args.threads<=host['topology']['physical_cores']
    assert host['memory']['available_physical_bytes']>4*2**30
    out.mkdir(parents=True);write(out/'host.json',host)
    cfg=copy.deepcopy(read(ROOT/'configs/sp_u12b5b_r4_synthetic.json'))
    cfg['cases']=[{'id':'next_host_safe_D4032_N256','K':4,'r':16,'D':4032,'N':256,
        'mode':'short','steps':1,'data_seed':42493000,'initialization_seed':42493001}]
    cfg['resource_limits']=LIMITS;cfg['worker_timeout_seconds']=1200
    path=out/'safe_smoke_config.json';write(path,cfg);runner.CONFIG=path
    # Single-step source and low-memory paths; no fit on formal N=12000/15000.
    runner.monitored(out/'reference_1','parity',1)
    runner.monitored(out/f'low_{args.threads}','parity',args.threads,True,out/'reference_1')
    write(out/'STOP.json',{'next_host_safe_smoke_complete':True,'resource_certification_full_N':False,
        'formal_training_authorized':False,'caps_unchanged':LIMITS,'STOP':'MENTOR_REVIEW'})

if __name__=='__main__':main()
