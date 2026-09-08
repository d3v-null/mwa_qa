#!/usr/bin/env python
"""
Cut a full observation down to a fixture small enough for CI.

The real demo observation is 18 GB unpacked, which no hosted runner can
download, so the tests run against a few timesteps and a couple of coarse
channels instead. Whole coarse channels are kept so that the edge and centre
channel flagging still has something to act on, and every antenna is kept so
the per-tile grouping and outlier scoring behave as they do on real data.

    make_demo_fixture.py <in_dir> <out_dir> --obsid 1226233968

Reads the layout that manual_testing.md packages (prep/, raw/, cal/, img/)
and writes the same layout, trimmed.
"""
from argparse import ArgumentParser
from pathlib import Path
from astropy.io import fits
import numpy as np
import shutil
import warnings

warnings.filterwarnings('ignore')

CHANS_PER_COARSE = 32


def trim_uvfits(src, dst, ntimes, ncoarse, chans_per_coarse=CHANS_PER_COARSE):
    with fits.open(src, memmap=True) as hdus:
        vis = hdus['PRIMARY']
        dates = np.asarray(vis.data.par('DATE'))
        keep_times = np.unique(dates)[:ntimes]
        rows = np.isin(dates, keep_times)
        nchan = ncoarse * chans_per_coarse
        if nchan > vis.header['NAXIS4']:
            raise ValueError('asked for more channels than the input has')
        # take the coarse channels from the middle of the band, which are the
        # least likely to be entirely flagged
        start = ((vis.header['NAXIS4'] // chans_per_coarse) // 2
                 - ncoarse // 2) * chans_per_coarse
        chans = slice(start, start + nchan)

        data = vis.data.data[rows][:, :, :, chans, :, :]
        pardata = [vis.data.par(name)[rows] for name in vis.data.parnames]
        groups = fits.GroupData(
            data.astype(np.float32), parnames=list(vis.data.parnames),
            pardata=pardata, bitpix=-32)
        out = fits.GroupsHDU(groups)
        for key in vis.header:
            if key in ('SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'GROUPS',
                       'PCOUNT', 'GCOUNT') or key.startswith('NAXIS'):
                continue
            if key.startswith('PTYPE') or key.startswith('PSCAL') \
                    or key.startswith('PZERO'):
                continue
            try:
                out.header[key] = (vis.header[key], vis.header.comments[key])
            except ValueError:
                pass
        # the reference pixel has to follow the channels that were kept
        out.header['CRPIX4'] = vis.header['CRPIX4'] - start
        fits.HDUList([out, hdus['AIPS AN'].copy()]).writeto(dst, overwrite=True)
    return dst


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('in_dir', type=Path, help='directory holding the full obs')
    parser.add_argument('out_dir', type=Path, help='where to write the fixture')
    parser.add_argument('--obsid', required=True)
    parser.add_argument('--ntimes', type=int, default=4,
                        help='timesteps to keep. Default %(default)s')
    parser.add_argument('--ncoarse', type=int, default=2,
                        help='coarse channels to keep. Default %(default)s')
    args = parser.parse_args()

    o = args.obsid
    prep = args.in_dir / 'prep' / ('birli_%s_2s_40kHz.uvfits' % o)
    cal = args.in_dir / 'cal' / ('hyp_%s_30l_src4k_8s_80kHz.uvfits' % o)
    copies = [
        (args.in_dir / 'raw' / ('%s.metafits' % o), 'raw'),
        (args.in_dir / 'cal' / ('hyp_soln_%s_30l_src4k.fits' % o), 'cal'),
    ]
    for pol in ('V', 'XX', 'YY'):
        copies.append((args.in_dir / 'img' /
                       ('wsclean_hyp_%s_30l_src4k_8s_80kHz-MFS-%s-dirty.fits'
                        % (o, pol)), 'img'))

    for sub in ('prep', 'raw', 'cal', 'img'):
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)

    for src, sub in copies:
        dst = args.out_dir / sub / src.name
        shutil.copyfile(src, dst)
        print('copied  %8.1f MB  %s' % (dst.stat().st_size / 1e6, dst.name))

    for src, sub in ((prep, 'prep'), (cal, 'cal')):
        if not src.exists():
            print('missing, skipped:', src)
            continue
        dst = args.out_dir / sub / src.name
        trim_uvfits(src, dst, args.ntimes, args.ncoarse)
        print('trimmed %8.1f MB  %s  (%d times, %d coarse chans)'
              % (dst.stat().st_size / 1e6, dst.name, args.ntimes, args.ncoarse))


if __name__ == '__main__':
    main()
