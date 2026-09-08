"""
Shared fixtures, and the rules for what can run without data.

Two separate datasets are involved:

* mwa_qa/data/, which the older unit tests read directly. It is not in the
  repository, so those modules fail at import when it is absent. They are
  skipped rather than left to error, so that a run on a fresh checkout still
  reports something useful.
* the demo observation the tests in test_manual_testing.py drive the command
  line tools with. The full observation is 18 GB, so a trimmed fixture is
  published alongside it and downloaded on demand.
"""
from pathlib import Path
import os
import shutil
import subprocess
import urllib.request
import zipfile

import pytest

from mwa_qa.data import DATA_PATH

FIXTURE_OBSID = '1226233968'
FIXTURE_URL = ('https://projects.pawsey.org.au/mwa-qa/'
               'mwa_qa_fixture_%s.zip' % FIXTURE_OBSID)

# the older tests open their data at import time, so a missing file is a
# collection error rather than a skip unless the module is ignored outright
_LEGACY_DATA = {
    'test_cal_metrics.py': ['hyp_soln_1062784992.fits', '1062784992.metafits'],
    'test_read_calfits.py': ['hyp_soln_1062784992.fits', '1062784992.metafits'],
    'test_read_uvfits.py': ['1062784992.uvfits'],
    'test_vis_metrics.py': ['1062784992.uvfits'],
    'test_prepvis_metrics.py': ['1062784992.uvfits', '1062784992.metafits'],
    'test_read_metafits.py': ['1062784992.metafits'],
    'test_image_metrics.py': ['wsclean_1088806248_XX.fits'],
    'test_read_image.py': ['wsclean_1088806248_XX.fits'],
}

collect_ignore = [
    module for module, needed in _LEGACY_DATA.items()
    if not all(os.path.exists(os.path.join(DATA_PATH, f)) for f in needed)
]


def pytest_configure(config):
    config.addinivalue_line(
        'markers', 'demo_data: needs the demo observation fixture')


@pytest.fixture(scope='session')
def demo_data(tmp_path_factory):
    """
    Directory holding the demo observation, laid out as manual_testing.md
    packages it: prep/, raw/, cal/, img/.

    Point MWA_QA_DEMO_DATA at an existing copy to skip the download, which is
    what a run against the full observation would do.
    """
    local = os.environ.get('MWA_QA_DEMO_DATA')
    if local:
        path = Path(local) / FIXTURE_OBSID
        if not path.is_dir():
            path = Path(local)
        if not (path / 'raw').is_dir():
            pytest.skip('MWA_QA_DEMO_DATA=%s has no raw/ directory' % local)
        return path

    if os.environ.get('MWA_QA_NO_DOWNLOAD'):
        pytest.skip('demo data not present and downloads are disabled')

    cache = Path(os.environ.get('MWA_QA_CACHE',
                                tmp_path_factory.mktemp('demo')))
    cache.mkdir(parents=True, exist_ok=True)
    obs = cache / FIXTURE_OBSID
    if not (obs / 'raw').is_dir():
        archive = cache / Path(FIXTURE_URL).name
        if not archive.exists():
            try:
                urllib.request.urlretrieve(FIXTURE_URL, archive)
            except Exception as exc:            # offline, proxied, whatever
                pytest.skip('could not fetch %s: %s' % (FIXTURE_URL, exc))
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(cache)
    if not (obs / 'raw').is_dir():
        pytest.skip('demo fixture did not unpack as expected')
    return obs


@pytest.fixture(scope='session')
def obsid():
    return FIXTURE_OBSID


@pytest.fixture
def results_dir(tmp_path):
    return tmp_path


def run(*argv):
    """
    Run one of the console scripts the way manual_testing.md does, and fail
    with its output attached rather than a bare exit code.
    """
    script = shutil.which(argv[0])
    if script is None:
        pytest.skip('%s is not installed' % argv[0])
    proc = subprocess.run([script] + [str(a) for a in argv[1:]],
                          capture_output=True, text=True)
    assert proc.returncode == 0, (
        '%s exited %d\nstdout:\n%s\nstderr:\n%s'
        % (' '.join(str(a) for a in argv), proc.returncode,
           proc.stdout, proc.stderr))
    return proc
