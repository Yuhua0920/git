"""R2 entry point: source audit, safe synthetic timing, and report closure."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sp_u12b5b_r2_common import *
import argparse
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','bench','reports','close'));args=ap.parse_args();firewall()
 if args.stage=='prepare':
  assert (OUT/'preservation_before.json').exists()
  from scripts.sp_u12b5b_r2_model import static
  from scripts.sp_u12b5b_r2_bench import prepare
  static();prepare()
 elif args.stage=='bench':
  from scripts.sp_u12b5b_r2_bench import monitor
  monitor()
 elif args.stage=='reports':
  from scripts.sp_u12b5b_r2_report import reports
  reports()
 else:
  from scripts.sp_u12b5b_r2_report import close
  close()
if __name__=='__main__':main()
