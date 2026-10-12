"""Path-only plot recovery: atomic QA temporaries stay inside owned figure folder."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2b_diagnostic as d
d.install_guard()
from scripts.analyze_sp_u2b_recovery import verify_or_create_csv


def main():
    original=ROOT/'scripts/plot_sp_u2b.py'
    pre=d.load(d.OUT/'sp_u2b_preregistration.json')
    assert d.sha(original)==pre['execution_source_sha256']['scripts/plot_sp_u2b.py']
    archive=d.OUT/'sp_u2b_plot_failure01'
    archive.mkdir(exist_ok=False)
    (archive/'plot_sp_u2b.py.txt').write_bytes(original.read_bytes())
    source=original.read_text(encoding='utf-8')
    needle="path=d.OUT/f'sp_u2b_fig_{letter}'"
    assert source.count(needle)==1
    derived=source.replace(needle,"path=d.OUT/'sp_u2b_figures'/f'fig_{letter}'")
    (archive/'executed_path_only_variant.py.txt').write_text(derived,encoding='utf-8')
    (d.OUT/'sp_u2b_figures').mkdir(exist_ok=False)
    before=d.sha(d.OUT/'sp_u2b_figure_source.csv')
    d.csvsave=verify_or_create_csv
    namespace={'__file__':str(Path(__file__).resolve()),'__name__':'sp_u2b_derived_plot'}
    exec(compile(derived,str(Path(__file__)),'exec'),namespace)
    namespace['main']()
    assert before==d.sha(d.OUT/'sp_u2b_figure_source.csv')
    d.save(d.OUT/'sp_u2b_plot_recovery.json',dict(error='output guard correctly blocked dot-prefixed atomic QA tempfile at outputs root',
          correction='put figure/QA outputs and temporary files inside owned sp_u2b_figures directory',
          original_source_sha256=d.sha(original),recovery_source_sha256=d.sha(Path(__file__)),
          derived_source_sha256=d.sha(archive/'executed_path_only_variant.py.txt'),
          firewall_unchanged=True,raw_figure_CSV_sha256=before,numerical_or_plot_geometry_changes=False,access=d.ACCESS))
    print('Plot path-only recovery PASS; guard unchanged; figure source byte-identical',flush=True)


if __name__=='__main__':main()
