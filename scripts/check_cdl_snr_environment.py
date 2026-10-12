"""T15A-FIX package-payload regression without importing/generating CDL."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.check_cdl_environment import environment_fingerprint,check_main_unchanged


def verify(phase):
    old=json.loads((ROOT/"outputs/t15a_attempt01_nominal_tx_snr/outputs/t15a_cdl_environment.json").read_text(encoding="utf8"))
    current=environment_fingerprint(ROOT/".venv-cdl")
    original=old["isolated_payload_fingerprint"]
    freeze=subprocess.check_output([str(ROOT/".venv-cdl/Scripts/python.exe"),"-m","pip","freeze"],text=True)
    original_freeze=(ROOT/"outputs/t15a_attempt01_nominal_tx_snr/outputs/t15a_cdl_packages_freeze.txt").read_text(encoding="utf8")
    main=check_main_unchanged()
    result=dict(task="T15A-FIX",phase=phase,channel_generator_called=False,
        main_environment=main,isolated_environment=dict(before_sha256=original["sha256"],after_sha256=current["sha256"],
            payload_unchanged=original["files"]==current["files"],freeze_unchanged=freeze==original_freeze),
        all_checks_passed=bool(main["all_checks_passed"] and original["files"]==current["files"] and freeze==original_freeze))
    path=ROOT/f"outputs/t15a_fix_environment_{phase}.json"
    path.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps(result,indent=2),flush=True)
    if not result["all_checks_passed"]: raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--phase",choices=("before","after"),default="before")
    verify(p.parse_args().phase)
