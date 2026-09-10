#!/usr/bin/env python
"""
Two views of a prepvis QA run that was grouped with --group_by:

  <name>_xxyy.png     XX against YY auto RMS, one point per tile. Tiles on the
                      diagonal behave the same in both polarisations, so
                      distance off it reads directly as a single-pol fault.
  <name>_spectra.png  per-channel spectrum of every tile, coloured by group,
                      which is where a group-wide gain offset shows up as a
                      band rather than as a set of independent bad tiles.

The spectra panel needs the .npz written by run_prepvisqa.py --save_spectra;
without it only the XX/YY plot is produced.
"""
from argparse import ArgumentParser
from mwa_qa import json_utils as ut
import matplotlib as mpl
import numpy as np
import os
import pylab
import re

mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['figure.titleweight'] = 'bold'

# assigned in a fixed order so a group keeps its colour between observations
COLORS = ['#eb6834', '#2a78d6', '#1baf7a', '#eda100',
          '#e87ba4', '#008300', '#4a3aa7', '#e34948']


def obsid_from(metrics, path):
    """
    The gps obsid. metrics['OBSID'] comes from the uvfits OBJECT header, which
    on some observations holds a field name rather than an obsid, so a 10 digit
    run in the filename is preferred when that happens.
    """
    obsid = str(metrics.get('OBSID', '')).strip()
    if obsid.isdigit():
        return obsid
    found = re.search(r'\d{10}', os.path.basename(path))
    return found.group(0) if found else (obsid or 'unknown')


def group_colors(groups):
    return {g: COLORS[i % len(COLORS)] for i, g in enumerate(sorted(set(groups)))}


def outname(base, suffix):
    if base is None:
        return None
    if base.endswith('.png'):
        return base.replace('.png', '_%s.png' % suffix)
    return '%s_%s.png' % (base, suffix)


def finish(fig, name, args):
    if args.save:
        fig.savefig(name, dpi=args.dpi, bbox_inches='tight')
        pylab.close(fig)
        print('wrote', name)
    else:
        pylab.show()


def plot_xxyy(metrics, groups, colors, title, args):
    xx = np.array(metrics['XX']['RMS'], dtype=float)
    yy = np.array(metrics['YY']['RMS'], dtype=float)
    annumbers = np.array(metrics['ANNUMBERS'])
    bad = set(np.atleast_1d(metrics['BAD_ANTS']).tolist())
    # tiles with no unflagged data plot at zero, which a log axis cannot show
    live = np.where((xx > 0) & (yy > 0))[0]

    fig = pylab.figure(figsize=(8, 7.5))
    fig.suptitle(title, size=13)
    ax = pylab.subplot(111)
    lo = max(min(xx[live].min(), yy[live].min()) * 0.7, 1e-3)
    hi = max(xx[live].max(), yy[live].max()) * 1.4
    ax.fill_between([lo, hi], [lo * 0.75, hi * 0.75], [lo * 1.25, hi * 1.25],
                    color='0.85', zorder=0, lw=0)
    ax.plot([lo, hi], [lo, hi], ls='dashed', color='0.4', lw=1.5, zorder=1)

    for g in sorted(set(groups[live])):
        inds = [i for i in live if groups[i] == g]
        keep = [i for i in inds if annumbers[i] not in bad]
        flag = [i for i in inds if annumbers[i] in bad]
        ax.scatter(xx[keep], yy[keep], s=55, facecolor='none',
                   edgecolor=colors[g], linewidths=2, zorder=3, label=g)
        ax.scatter(xx[flag], yy[flag], s=95, color=colors[g],
                   edgecolor='k', linewidths=1.4, zorder=4)
        for i in flag:
            # alternate the offset so neighbouring labels do not collide
            dy = 10 if annumbers[i] % 2 == 0 else -16
            ax.annotate(str(annumbers[i]), (xx[i], yy[i]),
                        textcoords='offset points', xytext=(0, dy),
                        ha='center', size=8, zorder=5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ticks = [t for t in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0] if lo <= t <= hi]
    for axis, setter in ((ax.xaxis, ax.set_xticks), (ax.yaxis, ax.set_yticks)):
        setter(ticks)
        axis.set_major_formatter(mpl.ticker.FormatStrFormatter('%g'))
        axis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel('XX auto RMS')
    ax.set_ylabel('YY auto RMS')
    ax.set_title('each tile\'s auto RMS in one polarisation against the other, coloured by group.\n'
                 'distance off the dashed line is polarisation asymmetry; filled = flagged',
                 size=8.5)
    ax.grid(ls='dotted')
    ax.legend(fontsize=8, loc='upper left')
    ax.tick_params(labelsize=10, direction='in', length=4, width=2)
    finish(fig, outname(args.figname, 'xxyy') or
           args.json.replace('.json', '_xxyy.png'), args)


def coarse_boundaries(metrics, nchan):
    """Fine channel indices where a coarse channel starts."""
    per_coarse = int(metrics.get('NCHANS_PER_COARSE') or 0)
    if per_coarse <= 0:
        per_coarse = max(nchan // 24, 1)   # 24 coarse channels is the MWA norm
    return np.arange(0, nchan + 1, per_coarse), per_coarse


def plot_spectra(metrics, groups, colors, spectra, title, args):
    annumbers = np.array(metrics['ANNUMBERS'])
    bad = set(np.atleast_1d(metrics['BAD_ANTS']).tolist())
    # ALL is the leftover pool of tiles with no group big enough to score
    # against, so it goes last and is labelled for what it is
    present = sorted(set(groups))
    ordered = [g for g in present if g != 'ALL']
    if 'ALL' in present:
        ordered.append('ALL')

    # y is not shared: each group sits at its own level, and the point of the
    # per-group rows is the spread within a group rather than between them
    bounds, per_coarse = coarse_boundaries(metrics, spectra.shape[1])
    step = max(int(np.ceil(len(bounds) / 8.0)), 1)
    fig, axes = pylab.subplots(len(ordered), 2, figsize=(13, 2.1 * len(ordered)),
                               sharex=True, sharey=False, squeeze=False)
    fig.suptitle(title, size=13, y=0.995)
    fig.text(0.5, 0.963, "per-channel RMS over time of each tile's autocorrelation, divided by "
             'the array median, one row per group. bold = flagged',
             ha='center', size=8.5)
    for row, group in enumerate(ordered):
        inds = np.where(groups == group)[0]
        for col, (pol, ip) in enumerate([('XX', 0), ('YY', 1)]):
            ax = axes[row][col]
            for i in inds:
                spec = spectra[i, :, ip]
                if np.all(np.isnan(spec)):
                    continue
                flagged = annumbers[i] in bad
                ax.plot(spec, color=colors[group],
                        lw=1.8 if flagged else 0.9,
                        alpha=0.95 if flagged else 0.45,
                        zorder=4 if flagged else 3)
            ax.grid(ls='dotted')
            ax.set_xlim(0, spectra.shape[1])
            ax.tick_params(labelsize=9, direction='in', length=4, width=2)
            if row == 0:
                ax.set_title(pol, size=11)
            if col == 0:
                label = 'ungrouped' if group == 'ALL' else group
                nflag = sum(1 for i in inds if annumbers[i] in bad)
                ax.set_ylabel('%s\nn=%d, %d flagged' % (label, len(inds), nflag),
                              size=9)
            ax.set_xticks(bounds)
            # every boundary gets a tick and a gridline, but labelling all of
            # them collides, so only every step'th carries a label
            ax.set_xticklabels([str(b) if i % step == 0 else ''
                                for i, b in enumerate(bounds)])
            if row == len(ordered) - 1:
                ax.set_xlabel('fine channel')
        # XX and YY of the same group stay on a common scale so a
        # single-polarisation fault is visible as an asymmetry
        lo = min(axes[row][c].get_ylim()[0] for c in (0, 1))
        hi = max(axes[row][c].get_ylim()[1] for c in (0, 1))
        for c in (0, 1):
            axes[row][c].set_ylim(lo, hi)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    finish(fig, outname(args.figname, 'spectra') or
           args.json.replace('.json', '_spectra.png'), args)


parser = ArgumentParser(
    description='Plot a grouped prepvis QA run: XX vs YY, and per-tile spectra')
parser.add_argument('json', type=str, help='prepvis metrics json file')
parser.add_argument('--spectra', default=None,
                    help='spectra npz from run_prepvisqa.py --save_spectra. '
                         'Defaults to the json path with _spectra.npz')
parser.add_argument('--save', action='store_true', default=None,
                    help='Save the figures instead of showing them')
parser.add_argument('--out', dest='figname', default=None,
                    help='Output figure basename')
parser.add_argument('--dpi', type=int, default=100,
                    help='Dots per inch for saved figures')
parser.add_argument('--title', default=None, help='Plot title')

args = parser.parse_args()
metrics = ut.load_json(args.json)
annumbers = np.array(metrics['ANNUMBERS'])
groups = np.array([str(g) for g in metrics.get(
    'GROUPS', ['ALL'] * len(annumbers))])
colors = group_colors(groups)

obsid = obsid_from(metrics, args.json)
title = args.title or '%s   group_by=%s   %d/%d flagged' % (
    obsid, metrics.get('GROUP_BY', 'none'),
    len(np.atleast_1d(metrics['BAD_ANTS'])), metrics['NANTS'])

plot_xxyy(metrics, groups, colors, title, args)

spectra_path = args.spectra or args.json.replace('.json', '_spectra.npz')
if os.path.exists(spectra_path):
    with np.load(spectra_path) as data:
        plot_spectra(metrics, groups, colors, data['spectra'], title, args)
else:
    print('no spectra npz at %s, skipping the spectra plot '
          '(rerun run_prepvisqa.py with --save_spectra)' % spectra_path)
