"""Keep the first failed demo diagnostic before correcting expectation metadata."""
import shutil
from scripts.sp_u12b5a_common import ROOT,OUT,firewall,write
def main():
    firewall();dst=OUT/'validation_attempt01'
    assert not dst.exists()
    dst.mkdir()
    files=list((ROOT/'outputs').glob('sp_u12b5a_mfa_*.json'))+list((ROOT/'outputs').glob('sp_u12b5a_mfa_*.csv'))+[ROOT/'scripts/sp_u12b5a_cases.py',ROOT/'scripts/validate_mfa_fidelity.py',OUT/'validation_summary.json',OUT/'pytest_result.json']
    for f in files:
        shutil.copyfile(f,dst/f.name)
    write(dst/'reason.json',dict(reason='Unchanged author rs_clip=1e-6 demo has weights_sum=1.0000140301507119. Original demo only prints the sum, not an assertion. Retain failed unit-weight diagnostic; distinguish qualitative demo completion from default-zero-clipping model fidelity. No tolerance change, author patch or policy change.',scientific_data=False))
if __name__=='__main__':main()
