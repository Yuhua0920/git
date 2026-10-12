"""Disjoint call-boundary timing, orthogonal to the frozen EM trace observer."""
import sys,time
import r7_common as q

class TimingObserver:
    def __init__(self,phase_writer=None):
        self.phase_writer=phase_writer;self.totals={v:0. for v in ('initialization','training','validation','serialization','waiting_monitoring')}
        self.stack=[];self.active='waiting_monitoring';self.events=[];self.previous=None
        self.files={str(q.ROOT/'src/baselines/mfa_reference_adapter.py'):{'fit_reference':'training'},
                    str(q.R6/'worker.py'):{'validate':'validation'},
                    str(q.R6/'model_ops.py'):{'save_model':'serialization','load_model':'serialization'},
                    str(q.ROOT/'reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py'):{'_initialize':'initialization'}}
    def profile(self,frame,event,arg):
        tag=self.files.get(frame.f_code.co_filename,{}).get(frame.f_code.co_name)
        if tag is None or event not in ('call','return'):return
        now=time.perf_counter();self.totals[self.active]+=now-self.last;self.last=now
        if event=='call':self.stack.append((frame,self.active));self.active=tag
        elif self.stack and self.stack[-1][0] is frame:_,self.active=self.stack.pop()
        self.events.append({'event':event,'phase':tag,'elapsed_seconds':now-self.began})
        if self.phase_writer:self.phase_writer({'phase':self.active,'last_event':self.events[-1],'exclusive_seconds_so_far':self.totals,'UTC':q.utc()})
    def __enter__(self):
        self.previous=sys.getprofile();self.began=self.last=time.perf_counter();sys.setprofile(self.profile);return self
    def __exit__(self,kind,error,tb):
        sys.setprofile(self.previous);now=time.perf_counter();self.totals[self.active]+=now-self.last
        self.result={'wall_seconds_in_bridge':now-self.began,'exclusive_seconds':self.totals,'events':self.events,'timing_changes_model':False,'error':repr(error) if error else None,'UTC':q.utc()}
