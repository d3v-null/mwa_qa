from argparse import ArgumentParser
from pathlib import Path
from astropy.io import fits
from mwa_qa.image_metrics import ImgMetrics
from mwa_qa.cal_metrics import CalMetrics
from mwa_qa.vis_metrics import VisMetrics


def img_qa():
    """installed as a console script in setup.py"""
    parser = ArgumentParser(description="QA for MWA images")
    parser.add_argument('fits', type=Path, nargs='+', help='FITS image file(s)')
    parser.add_argument(
        '--out', help='json output path', type=Path, default=None, required=False
    )

    args = parser.parse_args()

    met = ImgMetrics([*map(str, args.fits)])
    met.run_metrics()
    met.write_to(str(args.out) if args.out else None)


def cal_qa():
    """installed as a console script in setup.py"""
    parser = ArgumentParser(description="QA for hyperdrive calibration solutions")
    parser.add_argument('soln', type=Path, help='Hyperdrive fits file')
    parser.add_argument('metafits', type=Path, help='MWA metadata fits file')
    parser.add_argument('pol', default=None, help='Polarization, can be either X or Y')
    parser.add_argument(
        '--out', help='json output path', type=Path, default=None, required=False
    )

    args = parser.parse_args()
    met = CalMetrics(str(args.soln), str(args.metafits), args.pol)
    met.run_metrics()
    met.write_to(str(args.out) if args.out else None)


def vis_qa():
    """installed as a console script in setup.py"""
    parser = ArgumentParser(description="QA for MWA UVFITS visibility data")
    parser.add_argument('uvfits', type=Path, help='UVFITS visibility file')
    parser.add_argument(
        '--out', help='json output path', type=Path, default=None, required=False
    )

    args = parser.parse_args()
    met = VisMetrics(str(args.uvfits))
    met.run_metrics()
    met.write_to(str(args.out) if args.out else None)
