from pathlib import Path
import sys
import pytest
sys.dont_write_bytecode=True
def pytest_addoption(parser):parser.addoption('--project-root',required=True)
def pytest_configure(config):
    root=Path(config.getoption('--project-root')).resolve()
    sys.path.insert(0,str(root));sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'src_or_scripts'))
@pytest.fixture(scope='session')
def project_root(request):return Path(request.config.getoption('--project-root')).resolve()
@pytest.fixture(scope='session')
def apis(project_root):
    from receiver_sanity import load_apis
    return load_apis(project_root)
