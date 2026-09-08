#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
from mwa_qa.prepvis_metrics import (PrepvisMetrics, GROUPING_STRATEGIES,
                                    MIN_GROUP_SIZE, DEAD_RMS_THRESHOLD)

parser = ArgumentParser(description="QA for MWA UVFITS visibility data")
parser.add_argument('uvfits', type=Path, help='UVFITS visibility file')
parser.add_argument('metafits', type=Path, help='Metafits file')
parser.add_argument('--ex_annumbers', type=list, default=[],
                    help='List of antennas to be excluded. Default is empty')
parser.add_argument('--edge_flagging', dest='edge_flagging', action='store_true', default='auto',
                    help='Force flagging of the coarse channel band edges and centre channel. By default '
                         'this is decided per observation: applied unless the metafits says the '
                         'observation is oversampled, which has neither artefact.')
parser.add_argument('--no_edge_flagging', dest='edge_flagging', action='store_false',
                    help='Force edge and centre-channel flagging off.')
parser.add_argument('--antenna_flags', action='store_true',
                    help='Apply flags to antenna array. Default is False')
parser.add_argument('--cutoff_threshold', type=float, default=3,
                    help='Cutoff threshold value for modified z-score to identify outliers.')
parser.add_argument('--niter', type=int, default=10,
                    help='Number of iteration for outlier identification using modified z-score.')
parser.add_argument('--out', help='json output path', type=str,
                    default=None, required=False)
parser.add_argument('--split', action='store_true', help='Allow for splitting of autocorrelations based on antennas naming convention',
                    default=None, required=False)
parser.add_argument('--cable_flavor', action='store_true',
                    help='Shorthand for --group_by flavour. Ignored if --split is given.',
                    default=None, required=False)
parser.add_argument('--group_by', default=None, required=False,
                    choices=GROUPING_STRATEGIES + ['auto'],
                    help='Score each tile against others in its own group rather than against the whole '
                         'array, so a group whose signal-path gain differs from the majority is not flagged '
                         'wholesale. "auto" measures which factor best explains this observation and picks it. '
                         'Ignored if --split is given.')
parser.add_argument('--min_group_size', type=int, default=MIN_GROUP_SIZE,
                    help='Groups smaller than this back off to a coarser grouping. A median and a MAD over '
                         'a handful of tiles is not a usable reference. Default %(default)s.')
parser.add_argument('--dead_rms_threshold', type=float, default=DEAD_RMS_THRESHOLD,
                    help='Tiles in a group too small to score are still flagged below this RMS. '
                         'Default %(default)s.')
parser.add_argument('--save_spectra', action='store_true', default=None, required=False,
                    help='Also write per-channel spectra (.npz) for plot_prepvis_groups.py')

args = parser.parse_args()
m = PrepvisMetrics(str(args.uvfits), str(args.metafits),
                   ex_annumbers=args.ex_annumbers, edge_flagging=args.edge_flagging,
                   antenna_flags=args.antenna_flags, cutoff_threshold=args.cutoff_threshold,
                   niter=args.niter)
m.run_metrics(split_autos=args.split, cable_flavor=args.cable_flavor,
              group_by=args.group_by, min_group_size=args.min_group_size,
              dead_rms_threshold=args.dead_rms_threshold)
m.write_to(args.out)
if args.save_spectra:
    spectra_out = None if args.out is None else args.out.replace('.json', '_spectra.npz')
    print('wrote', m.write_spectra(spectra_out))
