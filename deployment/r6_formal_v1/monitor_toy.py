"""Safe background-launch exercise through the same PS and R5 Job plumbing."""
from pathlib import Path
import sys,os,time
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import LOCAL,write,scope
def main():
    scope();os.environ['R6_PERSONAL_SELFTEST']='1'
    job=LOCAL/('background_probe_'+time.strftime('%Y%m%d_%H%M%S'));job.mkdir(parents=True,exist_ok=False)
    write(LOCAL/'background_probe_latest.json',{'job':str(job)})
    from supervisor import run_worker
    limits={'RSS_bytes':256*2**20,'private_commit_bytes':512*2**20,'min_available_physical_bytes':1,'min_remaining_commit_bytes':1}
    state,result=run_worker('unit-probe',job,local=True,local_limits=limits,extra=['--probe-mode','tree'],timeout=10)
    write(LOCAL/'background_probe_result.json',{'PASS':state['PASS'],'job':str(job),'state':state,'synthetic_only':True,'formal_science_started':False})
if __name__=='__main__':main()
