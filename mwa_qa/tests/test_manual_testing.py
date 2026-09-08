"""
The command line flows documented in manual_testing.md, run end to end.

These drive the installed console scripts rather than the library, because
that is what the pipeline calls and what breaks when an argument is renamed
or an output filename changes. They run against the trimmed demo fixture by
default; set MWA_QA_DEMO_DATA to point at the full observation instead.
"""
import json

import numpy as np
import pytest

from mwa_qa.tests.conftest import run

pytestmark = pytest.mark.demo_data

CAL_NAME = '30l_src4k'
VIS_NAME = '30l_src4k_8s_80kHz'


def prep_uvfits(demo_data, obsid):
    return demo_data / 'prep' / ('birli_%s_2s_40kHz.uvfits' % obsid)


def metafits(demo_data, obsid):
    return demo_data / 'raw' / ('%s.metafits' % obsid)


# ---------------------------------------------------------------- prepvis qa

def test_prepvisqa(demo_data, obsid, results_dir):
    out = results_dir / ('birli_%s_prepvis_metrics.json' % obsid)
    run('run_prepvisqa.py', prep_uvfits(demo_data, obsid),
        metafits(demo_data, obsid), '--out', out)
    assert out.exists()

    metrics = json.loads(out.read_text())
    for key in ('NANTS', 'NCHAN', 'NTIMES', 'ANNUMBERS', 'BAD_ANTS',
                'BAD_ANTS_PERCENT', 'STATUS', 'THRESHOLD'):
        assert key in metrics, key
    assert metrics['STATUS'] in ('GOOD', 'BAD')
    assert len(metrics['ANNUMBERS']) == metrics['NANTS']
    for pol in ('XX', 'YY'):
        assert len(metrics[pol]['RMS']) == metrics['NANTS']
    # every flagged antenna has to be one that exists
    assert set(metrics['BAD_ANTS']) <= set(metrics['ANNUMBERS'])


def test_prepvisqa_edge_flagging_follows_oversampling(demo_data, obsid,
                                                      results_dir):
    """A legacy observation is edge flagged; an oversampled one is not."""
    out = results_dir / 'auto.json'
    run('run_prepvisqa.py', prep_uvfits(demo_data, obsid),
        metafits(demo_data, obsid), '--out', out)
    metrics = json.loads(out.read_text())
    assert metrics['EDGE_FLAGGING'] is not metrics['OVERSAMPLED']

    forced = results_dir / 'forced.json'
    run('run_prepvisqa.py', prep_uvfits(demo_data, obsid),
        metafits(demo_data, obsid), '--no_edge_flagging', '--out', forced)
    assert json.loads(forced.read_text())['EDGE_FLAGGING'] is False


def test_prepvisqa_metafits_flags_land_on_the_right_tiles(demo_data, obsid,
                                                          results_dir):
    """
    TILEDATA is ordered by input, so a flagged row's position is not its
    antenna number. Tiles flagged in the metafits must be the ones blanked.
    """
    from astropy.io import fits

    out = results_dir / 'flags.json'
    run('run_prepvisqa.py', prep_uvfits(demo_data, obsid),
        metafits(demo_data, obsid), '--out', out)
    metrics = json.loads(out.read_text())

    with fits.open(metafits(demo_data, obsid)) as hdus:
        tiledata = hdus['TILEDATA'].data[::2]
        flagged = set(int(a) for a, f
                      in zip(tiledata['Antenna'], tiledata['Flag']) if f == 1)

    annumbers = np.array(metrics['ANNUMBERS'])
    rms = np.array(metrics['XX']['RMS'], dtype=float)
    blanked = set(int(a) for a in annumbers[rms == 0])
    assert flagged <= blanked, (
        'metafits-flagged tiles %s were not blanked' % sorted(flagged - blanked))


@pytest.mark.parametrize('group_by', ['none', 'flavour', 'family',
                                      'flavour+family', 'receiver', 'auto'])
def test_prepvisqa_group_by(demo_data, obsid, results_dir, group_by):
    out = results_dir / ('%s.json' % group_by.replace('+', '_'))
    run('run_prepvisqa.py', prep_uvfits(demo_data, obsid),
        metafits(demo_data, obsid), '--group_by', group_by, '--out', out)
    metrics = json.loads(out.read_text())

    if group_by == 'none':
        assert metrics['GROUP_BY'] == 'none'
        return
    if group_by == 'auto':
        assert metrics['GROUP_BY'] in ('none', 'flavour', 'family',
                                       'flavour+family', 'receiver')
        assert metrics['GROUP_SELECTION']
    else:
        assert metrics['GROUP_BY'] == group_by

    if metrics['GROUP_BY'] == 'none':
        return
    groups = np.array(metrics['GROUPS'])
    assert len(groups) == metrics['NANTS']
    # a group is either big enough to score against, or marked unscored
    for name in set(groups):
        entry = metrics['XX'].get(name)
        if entry is None:
            continue
        size = int((groups == name).sum())
        if size < 8:
            assert entry.get('UNSCORED') is True, name


def test_prepvisqa_small_groups_are_unscored_but_dead_tiles_still_flagged(
        demo_data, obsid, results_dir):
    """
    A handful of tiles cannot supply a median and a MAD, so they get no
    z-score, but a tile with almost no power is still caught.
    """
    out = results_dir / 'tiny_groups.json'
    # every group is undersized at this threshold, so nothing gets a z-score
    run('run_prepvisqa.py', prep_uvfits(demo_data, obsid),
        metafits(demo_data, obsid), '--group_by', 'flavour+family',
        '--min_group_size', '200', '--dead_rms_threshold', '0.1', '--out', out)
    metrics = json.loads(out.read_text())
    groups = np.array(metrics['GROUPS'])
    for name in set(groups):
        entry = metrics['XX'].get(name)
        if entry is not None:
            assert entry.get('UNSCORED') is True, name

    # raising the floor above the whole array flags everything that has data
    greedy = results_dir / 'greedy.json'
    run('run_prepvisqa.py', prep_uvfits(demo_data, obsid),
        metafits(demo_data, obsid), '--group_by', 'flavour+family',
        '--min_group_size', '200', '--dead_rms_threshold', '1e6',
        '--out', greedy)
    assert len(json.loads(greedy.read_text())['BAD_ANTS']) > \
        len(metrics['BAD_ANTS'])


def test_plot_prepvisqa(demo_data, obsid, results_dir):
    metrics = results_dir / 'prep.json'
    run('run_prepvisqa.py', prep_uvfits(demo_data, obsid),
        metafits(demo_data, obsid), '--out', metrics)
    run('plot_prepvisqa.py', metrics, '--save',
        '--out', results_dir / 'prep.png')
    for suffix in ('_rms', '_modz'):
        assert (results_dir / ('prep%s.png' % suffix)).exists()


def test_plot_prepvis_groups(demo_data, obsid, results_dir):
    metrics = results_dir / 'prep.json'
    run('run_prepvisqa.py', prep_uvfits(demo_data, obsid),
        metafits(demo_data, obsid), '--group_by', 'auto', '--save_spectra',
        '--out', metrics)
    spectra = results_dir / 'prep_spectra.npz'
    assert spectra.exists()
    with np.load(spectra) as npz:
        assert npz['spectra'].shape[0] == json.loads(
            metrics.read_text())['NANTS']

    run('plot_prepvis_groups.py', metrics, '--save',
        '--out', results_dir / 'groups.png')
    assert (results_dir / 'groups_xxyy.png').exists()
    assert (results_dir / 'groups_spectra.png').exists()


def test_plot_prepvis_groups_without_spectra(demo_data, obsid, results_dir):
    """The XX/YY plot still works when no spectra npz was written."""
    metrics = results_dir / 'nospec.json'
    run('run_prepvisqa.py', prep_uvfits(demo_data, obsid),
        metafits(demo_data, obsid), '--out', metrics)
    run('plot_prepvis_groups.py', metrics, '--save',
        '--out', results_dir / 'nospec.png')
    assert (results_dir / 'nospec_xxyy.png').exists()
    assert not (results_dir / 'nospec_spectra.png').exists()


# -------------------------------------------------------------------- cal qa

@pytest.mark.parametrize('pol', ['X', 'Y'])
def test_calqa(demo_data, obsid, results_dir, pol):
    soln = demo_data / 'cal' / ('hyp_soln_%s_%s.fits' % (obsid, CAL_NAME))
    if not soln.exists():
        pytest.skip('no calibration solutions in this dataset')
    out = results_dir / ('hyp_soln_%s_%s_%s.json' % (obsid, CAL_NAME, pol))
    run('run_calqa.py', soln, metafits(demo_data, obsid),
        '--pol', pol, '--out', out)
    metrics = json.loads(out.read_text())
    assert 'PERCENT_NONCONVERGED_CHS' in metrics
    assert 0 <= metrics['PERCENT_NONCONVERGED_CHS'] <= 100


def test_plot_calqa(demo_data, obsid, results_dir):
    soln = demo_data / 'cal' / ('hyp_soln_%s_%s.fits' % (obsid, CAL_NAME))
    if not soln.exists():
        pytest.skip('no calibration solutions in this dataset')
    out = results_dir / 'cal_X.json'
    run('run_calqa.py', soln, metafits(demo_data, obsid),
        '--pol', 'X', '--out', out)
    run('plot_calqa.py', out, '--save', '--out', results_dir / 'cal_X.png')
    run('plot_caljson.py', out, '--save',
        '--out', results_dir / 'cal_X_json.png')
    assert list(results_dir.glob('cal_X*.png'))


# -------------------------------------------------------------------- vis qa

def test_visqa(demo_data, obsid, results_dir):
    uvfits = demo_data / 'cal' / ('hyp_%s_%s.uvfits' % (obsid, VIS_NAME))
    if not uvfits.exists():
        pytest.skip('no calibrated visibilities in this dataset')
    out = results_dir / 'vis.json'
    run('run_visqa.py', uvfits, '--out', out)
    assert json.loads(out.read_text())
    run('plot_visqa.py', out, '--save', '--out', results_dir / 'vis.png')
    assert list(results_dir.glob('vis*.png'))


# -------------------------------------------------------------------- img qa

def test_imgqa(demo_data, obsid, results_dir):
    img = demo_data / 'img'
    images = [img / ('wsclean_hyp_%s_%s-MFS-%s-dirty.fits'
                     % (obsid, VIS_NAME, pol)) for pol in ('V', 'XX', 'YY')]
    if not all(i.exists() for i in images):
        pytest.skip('no images in this dataset')
    out = results_dir / 'img.json'
    run('run_imgqa.py', *images, '--out', out)
    assert json.loads(out.read_text())
    run('plot_imgqa.py', out, '--save', '--out', results_dir / 'img')
    assert list(results_dir.glob('img*.png'))


# ----------------------------------------------------- calqa on a dead solution

def test_calqa_on_dead_solutions(results_dir):
    """
    manual_testing.md keeps a solution file that never converged; its
    non-converged percentage has to come out at 100.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    soln = root / 'hyp_soln_1324134624.fits'
    meta = root / '1324134624.metafits'
    if not (soln.exists() and meta.exists()):
        pytest.skip('the dead solution files are not in this checkout')
    out = results_dir / 'dead.json'
    run('run_calqa.py', soln, meta, '--pol', 'X', '--out', out)
    assert json.loads(out.read_text())['PERCENT_NONCONVERGED_CHS'] == 100
