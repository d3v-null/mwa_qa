from mwa_qa.read_uvfits import UVfits
from mwa_qa.read_metafits import Metafits
from mwa_qa import json_utils as ju
from collections import OrderedDict
import numpy as np
import pylab
import itertools
import warnings

pol_dict = {'XX': 0, 'YY': 1, 'XY': 2, 'YX': 3}

# grouping strategies 'auto' chooses between, and the smallest group it will
# score against: a median and a MAD taken over a handful of tiles is not a
# usable reference, so anything smaller backs off to a coarser grouping.
GROUPING_STRATEGIES = ['none', 'flavour', 'family', 'flavour+family', 'receiver']
MIN_GROUP_SIZE = 8
MIN_VARIANCE_EXPLAINED = 0.15
# tiles in a group too small to score are still flagged below this RMS: no
# reference group is needed to tell that a tile has almost no power.
DEAD_RMS_THRESHOLD = 0.1


def converter(input_list, output_list):
    for elements in input_list:
        if type(elements) == list:
            converter(elements, output_list)
        else:
            output_list.append(elements)
    return output_list


class PrepvisMetrics(object):
    def __init__(self, uvfits_path, metafits_path, ex_annumbers=[], edge_flagging='auto', antenna_flags=False, cutoff_threshold=3, niter=10):
        """
        - edge_flagging : 'auto' (default) flags the coarse channel band edges
          and centre channel unless the observation is oversampled, which does
          not have those artefacts. True or False force the choice.
        """
        warnings.filterwarnings("ignore")
        self.uvfits_path = uvfits_path
        self.metafits_path = metafits_path
        self.uvf = UVfits(self.uvfits_path)
        self.meta = Metafits(self.metafits_path)
        self.ex_annumbers = ex_annumbers
        if edge_flagging == 'auto':
            edge_flagging = not getattr(self.meta, 'oversampled', False)
        self.edge_flagging = edge_flagging
        self.antenna_flags = antenna_flags
        self.cutoff_threshold = cutoff_threshold
        self.niter = niter
        self.antenna_numbers = np.unique(self.uvf.auto_antpairs())
        self._autos = None

    def autos(self):
        """
        Masked autocorrelations, cached because reading and masking them is by
        far the most expensive step and several callers need the same array.
        The cached array is shared, not copied, so callers must not modify it.
        """
        if self._autos is not None:
            return self._autos
        self._autos = self._read_autos()
        return self._autos

    def _read_autos(self):
        auto_antpairs = self.uvf.auto_antpairs()
        autos = self.uvf.data_for_antpairs(
            auto_antpairs)  # (only xx and yy pols)
        _sh = autos.shape

        # flags are properly propages for the autocorrelations, therefore
        # we are mually flagging the band edges and centre frequency only
        # for obsids below 13*
        if self.edge_flagging:
            flags = self._evaluate_edge_flags()
            flags = flags[np.newaxis, np.newaxis, :, np.newaxis]
            flags = flags.repeat(_sh[0], axis=0).repeat(
                _sh[1], axis=1).repeat(_sh[3], axis=3)
        else:
            flags = np.zeros((_sh), dtype=bool)
        autos[flags] = np.nan
        if len(self.ex_annumbers) > 0:
            autos[:, self.ex_annumbers, :, :] = np.nan
        # applying flags from metafits
        ind = self.flags_from_metafits()
        autos[:, ind, :, :] = np.nan
        return autos

    def _indices_for_annumbers(self, annumbers):
        """
        Positions into self.antenna_numbers for the given antenna numbers.
        The metafits TILEDATA is ordered by input rather than by antenna
        number, so metadata must be matched on the antenna number itself and
        never on its row position.
        """
        inds = []
        for annumber in annumbers:
            found = np.where(self.antenna_numbers == annumber)[0]
            if len(found) > 0:
                inds.append(int(found[0]))
        return inds

    def flags_from_metafits(self):
        meta_annumbers = np.array(self.meta.antenna_numbers)
        flags = np.array(self.meta.flag_array)
        flagged = np.unique(meta_annumbers[np.where(flags == 1)[0]])
        return self._indices_for_annumbers(flagged)

    def _evaluate_edge_flags(self):
        cchan_bandwidth = 1_280_000
        edge_bandwidth = 80_000
        freq_res = self.uvf.channel_width
        num_chans = self.uvf.Nchan
        edge_chans = int(edge_bandwidth // freq_res)
        num_fchans = int(cchan_bandwidth // freq_res)
        num_cchans = int(num_chans // num_fchans)
        center_fine_chan = num_fchans // 2
        # start with all flags true
        flags = np.full((num_cchans, num_fchans), True)
        # unflag within edge chans
        flags[:, edge_chans:-edge_chans] = False
        # flag center fine chan
        flags[:, center_fine_chan] = True

        return np.array(flags).flatten()

    def _plot_mode(self, data, mode):
        if mode == 'amp':
            return np.abs(data)
        elif mode == 'phs':
            return np.angle(data)
        elif mode == 'real':
            return np.real(data)
        elif mode == 'imag':
            return np.imag(data)
        elif mode == 'log':
            return np.log10(np.abs(data))
        else:
            raise ValueError("mode {} is not recognized".format(mode))

    def plot_spectra_across_chan(self, freq_chan, mode='log', save=None, figname=None):
        autos = self.autos()
        _sh = autos.shape
        plot_data = self._plot_mode(
            autos[:, :, freq_chan, :], mode=mode)
        fig = pylab.figure(figsize=(9, 7))
        pylab.suptitle('Channel {}'.format(freq_chan))
        for i in range(4):
            ax = pylab.subplot(2, 2, i + 1)
            for j in range(_sh[1]):
                ax.plot(plot_data[:, j, i], '.-', alpha=0.5)
            ax.set_title(list(pol_dict.keys())[i])
            ax.grid(ls='dotted')
            if i % 2 == 0:
                pylab.ylabel(mode, fontsize=12)
            if i // 2 > 0:
                pylab.xlabel('Timestamp', fontsize=12)
        pylab.subplots_adjust(hspace=0.2, wspace=0.3)
        if save:
            if figname is None:
                outfile = self.uvfits_path.replace(
                    '.uvfits', '_f{}_{}_{}.png'.format(freq_chan, mode, pol))
            else:
                outfile = figname
            pylab.savefig(outfile)
        else:
            pylab.show()

    def plot_spectra_across_time(self, timestamp, mode='log', save=None, figname=None):
        autos = self.autos()
        _sh = autos.shape
        plot_data = self._plot_mode(
            autos[timestamp, :, :, :], mode=mode)
        fig = pylab.figure(figsize=(9, 7))
        pylab.suptitle('Timetamp {}'.format(timestamp))
        for i in range(4):
            ax = pylab.subplot(2, 2, i + 1)
            for j in range(_sh[1]):
                ax.plot(plot_data[j, :, i], '.-', alpha=0.5)
            ax.set_title(list(pol_dict.keys())[i])
            ax.grid(ls='dotted')
            if i % 2 == 0:
                pylab.ylabel(mode, fontsize=12)
            if i // 2 > 0:
                pylab.xlabel('Frequency channel', fontsize=12)
        pylab.subplots_adjust(hspace=0.2, wspace=0.3)
        if save:
            if figname is None:
                outfile = self.uvfits_path.replace(
                    '.uvfits', '_t{}_{}_{}.png'.format(timestamp, mode, pol))
            else:
                outfile = figname
            pylab.savefig(outfile)
        else:
            pylab.show()

    def plot_spectra_2D(self, annumber, mode='log',
                        save=None, figname=None):
        autos = self.autos()
        plot_data = self._plot_mode(
            autos[:, annumber, :, :], mode=mode)
        fig = pylab.figure(figsize=(8, 7))
        pylab.suptitle('Ant {} -- {}'.format(annumber, mode))
        for i in range(4):
            ax = pylab.subplot(2, 2, i + 1)
            vmin = np.nanmin(plot_data[:, :, i])
            vmax = np.nanmax(plot_data[:, :, i])
            im = ax.imshow(plot_data[:, :, i],
                           aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_title(list(pol_dict.keys())[i])
            pylab.colorbar(im)
            if i % 2 == 0:
                pylab.ylabel('Timestamp', fontsize=12)
            if i // 2 > 0:
                pylab.xlabel('Frequency Channel', fontsize=12)
        pylab.subplots_adjust(hspace=0.2, wspace=0.3)
        if save:
            if figname is None:
                outfile = self.uvfits_path.replace(
                    '.uvfits', '_{}_{}.png'.format(mode, pol))
            else:
                outfile = figname
            pylab.savefig(outfile)
        else:
            pylab.show()

    def flag_occupancy(self, data):
        """
        Checking if the flag occupancy of any antenna/tile is greater than 50 percent.
        If so, the tile is discarded and included in the list of bad tiles.
        """
        _sh = data.shape
        count = len(np.where(~np.isnan(data))[0]) / _sh[0]
        percent_gddata = np.count_nonzero(data > 0., axis=1) / count * 100
        # discarding antenna with less than 50% of data
        inds = np.where(percent_gddata < 50.)
        return percent_gddata, inds

    def cable_flavors(self):
        """
        Cable flavour of each antenna, in the same order as self.antenna_numbers.
        The metafits TILEDATA is ordered by input rather than by antenna number,
        so it is matched on the antenna number rather than on row position.
        """
        return np.array([str(f) for f in
                         self._meta_column_for_antennas(self.meta.cable_flavors)])

    def _meta_column_for_antennas(self, column):
        """
        A metafits TILEDATA column reordered onto self.antenna_numbers.
        See _indices_for_annumbers for why row position cannot be used.
        """
        meta_annumbers = np.array(self.meta.antenna_numbers)
        column = np.array(column)
        values = []
        for annumber in self.antenna_numbers:
            inds = np.where(meta_annumbers == annumber)[0]
            if len(inds) == 0:
                raise ValueError(
                    'Antenna {} is not present in the metafits'.format(annumber))
            values.append(column[inds[0]])
        return np.array(values)

    def antenna_families(self):
        """
        'Tile' or 'Hex' for each antenna, from the metafits tile names. The two
        families sit at noticeably different auto-correlation levels even
        within one cable flavour.
        """
        names = self._meta_column_for_antennas(self.meta.antenna_names)
        families = []
        for name in names:
            name = str(name).strip()
            families.append(
                'Tile' if name.startswith('Tile')
                else ('Hex' if name.startswith('Hex') else 'Other'))
        return np.array(families)

    def receiver_numbers(self):
        """Receiver number of each antenna, in self.antenna_numbers order."""
        return np.array(['Rx%s' % r for r in
                         self._meta_column_for_antennas(self.meta.receiver_ids)])

    def _grouping_levels(self, strategy):
        """
        The label arrays a strategy is built from, finest first. A tile whose
        finest group is too small backs off to the next level, so that no tile
        is ever scored against a handful of others.
        """
        flavour, family = self.cable_flavors(), self.antenna_families()
        receiver = self.receiver_numbers()
        levels = {
            'none': [],
            'flavour': [flavour],
            'family': [family],
            'receiver': [receiver],
            'flavour+family': [np.char.add(np.char.add(flavour, '_'), family),
                               flavour],
        }
        if strategy not in levels:
            raise ValueError(
                'unknown grouping strategy {!r}, expected one of {}'.format(
                    strategy, sorted(levels) + ['auto']))
        return levels[strategy]

    def grouping_labels(self, strategy, min_group_size=MIN_GROUP_SIZE):
        """
        Group label for each antenna under the named strategy. Groups smaller
        than min_group_size back off to a coarser level, ending at ALL (the
        whole array) so that every tile always has a usable reference.
        """
        labels = np.array(['ALL'] * len(self.antenna_numbers), dtype=object)
        assigned = np.zeros(len(labels), dtype=bool)
        for level in self._grouping_levels(strategy):
            level = np.array([str(v) for v in level])
            for value in np.unique(level):
                inds = np.where((level == value) & ~assigned)[0]
                if len(inds) >= min_group_size:
                    labels[inds] = value
                    assigned[inds] = True
        return np.array([str(v) for v in labels])

    @staticmethod
    def variance_explained(values, labels):
        """
        Fraction of the variance in values accounted for by the grouping,
        adjusted for the number of groups (omega squared) so that a strategy
        is not rewarded merely for splitting the array more finely.
        """
        values = np.asarray(values, dtype=float)
        labels = np.asarray([str(v) for v in labels])
        good = np.isfinite(values)
        values, labels = values[good], labels[good]
        groups = np.unique(labels)
        n, k = len(values), len(groups)
        if n <= k or k < 2:
            return 0.0
        grand = values.mean()
        ssb = sum(len(values[labels == g]) * (values[labels == g].mean() - grand) ** 2
                  for g in groups)
        sst = ((values - grand) ** 2).sum()
        if sst == 0:
            return 0.0
        msw = (sst - ssb) / (n - k)
        return float((ssb - (k - 1) * msw) / (sst + msw))

    def select_grouping(self, strategies=GROUPING_STRATEGIES,
                        min_group_size=MIN_GROUP_SIZE,
                        min_variance_explained=MIN_VARIANCE_EXPLAINED):
        """
        Pick the grouping that best explains the tile-to-tile spread in
        auto-correlation RMS. Returns (strategy, {strategy: omega squared}).

        Which factor structures an array is not fixed: on some observations
        the cable flavour dominates, on others the receiver does, so the
        choice is measured per observation rather than assumed.
        """
        autos = self.autos()
        amps = np.abs(np.nanmean(autos, axis=0))
        normed = amps / np.nanmedian(amps, axis=0)
        scores = {}
        for strategy in strategies:
            labels = self.grouping_labels(strategy, min_group_size)
            if len(np.unique(labels)) < 2:
                scores[strategy] = 0.0
                continue
            scores[strategy] = float(np.mean([
                self.variance_explained(
                    self.calculate_rms(normed[:, :, pol_dict[p]]), labels)
                for p in ['XX', 'YY']]))
        best = max(scores, key=scores.get)
        if scores[best] < min_variance_explained:
            best = 'none'
        return best, scores

    def split_annames(self):
        """
        Splitting the antenna indeices as per antenna name conventions ('Tile', 'Hex')
        """
        annames = self.uvf.ant_names[self.antenna_numbers]
        first = [i for i in range(len(annames))
                 if annames[i].startswith('Tile')]
        second = [i for i in range(len(annames))
                  if annames[i].startswith('Hex')]
        # print(first, second)
        return first, second

    def calculate_rms(self, data):
        """
        Calculating root mean square(rms) across freq
        """
        # calculating rms across frequency
        _sh = data.shape
        rms = np.sqrt(np.nansum(data ** 2, axis=1) / _sh[1])
        return rms

    def calculate_mod_zscore(self, data):
        median_data = np.nanmedian(data, axis=0)
        diff_data = data - median_data
        mad = np.nanmedian(np.abs(diff_data), axis=0)
        mod_zscore = diff_data / mad / 1.4826
        return np.nanmean(mod_zscore, axis=1)

    def iterative_mod_zscore(self, data, threshold, niter):
        bad_inds = []
        modz_dict = {}
        mod_zscore = self.calculate_mod_zscore(data)
        inds = np.where((mod_zscore < -threshold)
                        | (mod_zscore > threshold))
        if len(inds[0]) > 0:
            bad_inds.append(inds[0].tolist())
        count = 1
        modz_dict[count - 1] = mod_zscore
        while (count <= niter and len(inds[0]) > 0):
            data[inds[0]] = np.nan
            mod_zscore = self.calculate_mod_zscore(data)
            inds = np.where(np.abs(mod_zscore) > threshold)
            modz_dict[count] = mod_zscore
            if len(inds[0]) > 0:
                bad_inds.append(inds[0].tolist())
                count += 1
        return modz_dict, converter(bad_inds, [])

    def _initialize_metrics_dict(self):
        self.metrics = OrderedDict()
        self.metrics['NANTS'] = self.uvf.Nants
        self.metrics['NTIMES'] = self.uvf.Ntimes
        self.metrics['NCHAN'] = self.uvf.Nchan
        self.metrics['NPOLS'] = self.uvf.Npols
        self.metrics['OBSID'] = self.uvf.obsid
        # fine channels per 1.28 MHz coarse channel, so plots can mark the
        # coarse channel boundaries without re-reading the metafits
        self.metrics['NCHANS_PER_COARSE'] = int(
            1_280_000 // self.uvf.channel_width)
        self.metrics['OVERSAMPLED'] = bool(
            getattr(self.meta, 'oversampled', False))
        self.metrics['EDGE_FLAGGING'] = bool(self.edge_flagging)
        if self.antenna_flags:
            self.metrics['ANNUMBERS'] = np.delete(
                self.antenna_numbers, self.flags_from_metafits())
        else:
            self.metrics['ANNUMBERS'] = self.antenna_numbers
        self.metrics['NANTS'] = len(self.metrics['ANNUMBERS'])
        self.metrics['XX'] = OrderedDict()
        self.metrics['YY'] = OrderedDict()

    def run_metrics(self, split_autos=False, cable_flavor=False,
                    group_by=None, min_group_size=MIN_GROUP_SIZE,
                    dead_rms_threshold=DEAD_RMS_THRESHOLD):
        """
        - split_autos : score Tile* and Hex* separately (legacy behaviour)
        - cable_flavor : shorthand for group_by='flavour'
        - group_by : 'none', 'flavour', 'family', 'flavour+family', 'receiver',
          or 'auto' to pick whichever best explains this observation's spread
        - dead_rms_threshold : tiles in a group too small to score are still
          flagged below this RMS
        """
        if group_by is None and cable_flavor:
            group_by = 'flavour'
        self._initialize_metrics_dict()
        group_labels = None
        if group_by is not None and group_by != 'none':
            if group_by == 'auto':
                group_by, scores = self.select_grouping(
                    min_group_size=min_group_size)
                self.metrics['GROUP_SELECTION'] = scores
            if group_by != 'none':
                group_labels = self.grouping_labels(group_by, min_group_size)
        self.metrics['GROUP_BY'] = group_by or 'none'
        if group_labels is not None:
            self.metrics['GROUPS'] = group_labels
        # auto correlations
        autos = self.autos()
        # amplitude averaged over time
        autos_amps = np.abs(np.nanmean(autos, axis=0))
        # normalizing by the median
        autos_amps_norm = autos_amps / np.nanmedian(autos_amps, axis=0)
        # finding misbehaving antennas
        for p in ['XX', 'YY']:
            bad_ants = []
            _, bd_inds = self.flag_occupancy(
                autos_amps_norm[:, :, pol_dict[p]])
            if len(bd_inds) > 0:
                bad_ants.append(self.metrics['ANNUMBERS'][bd_inds])
            autos_amps_norm[bd_inds, :, pol_dict[p]] = np.nan
            # calculating root mean square
            rms = self.calculate_rms(
                autos_amps_norm[:, :, pol_dict[p]])
            if split_autos:
                # splitting autos by antenna naming conventions
                first, second = self.split_annames()
                modz_dict_first, bd_inds_first = self.iterative_mod_zscore(
                    autos_amps_norm[first, :, pol_dict[p]], threshold=self.cutoff_threshold, niter=self.niter)
                modz_dict_second, bd_inds_second = self.iterative_mod_zscore(
                    autos_amps_norm[second, :, pol_dict[p]], threshold=self.cutoff_threshold, niter=self.niter)
                bd_inds = np.append(np.array(first)[bd_inds_first],
                                    np.array(second)[bd_inds_second])
                # bd_inds_flatten = np.array(bd_inds)
                self.metrics[p]['MODZ_SCORE'] = {}
                self.metrics[p]['MODZ_SCORE']['FIRST'] = modz_dict_first
                self.metrics[p]['MODZ_SCORE']['SECOND'] = modz_dict_second
                self.metrics['ANNUMBERS_FIRST'] = first
                self.metrics['ANNUMBERS_SECOND'] = second
            elif group_labels is not None:
                # score each tile against the others in its own group. comparing
                # against the whole-array median instead flags an entire group
                # whenever its signal-path gain differs from the majority's.
                combined_modz = np.full(len(group_labels), np.nan)
                bd_inds = np.array([], dtype=int)
                for group in np.unique(group_labels):
                    inds = np.where(group_labels == group)[0]
                    if len(inds) < min_group_size:
                        # nothing comparable to score these against, so no
                        # z-score is computed. a tile with almost no power
                        # needs no reference group to be recognised though, so
                        # those are still flagged.
                        dead_inds = inds[np.asarray(rms)[inds]
                                         < dead_rms_threshold]
                        self.metrics[p][group] = OrderedDict()
                        self.metrics[p][group]['ANNUMBERS'] = \
                            self.metrics['ANNUMBERS'][inds]
                        self.metrics[p][group]['MODZ_SCORE'] = {}
                        self.metrics[p][group]['BAD_ANTS'] = \
                            self.metrics['ANNUMBERS'][dead_inds]
                        self.metrics[p][group]['UNSCORED'] = True
                        bd_inds = np.append(bd_inds, dead_inds)
                        continue
                    modz_dict, bd_inds_group = self.iterative_mod_zscore(
                        autos_amps_norm[inds, :, pol_dict[p]],
                        threshold=self.cutoff_threshold, niter=self.niter)
                    bd_inds_group = np.array(bd_inds_group, dtype=int)
                    self.metrics[p][group] = OrderedDict()
                    self.metrics[p][group]['ANNUMBERS'] = \
                        self.metrics['ANNUMBERS'][inds]
                    self.metrics[p][group]['MODZ_SCORE'] = modz_dict
                    self.metrics[p][group]['BAD_ANTS'] = \
                        self.metrics['ANNUMBERS'][inds[bd_inds_group]]
                    combined_modz[inds] = modz_dict[0]
                    bd_inds = np.append(bd_inds, inds[bd_inds_group])
                bd_inds = np.unique(bd_inds).astype(int)
                # iteration 0 of each group, stitched back into array order,
                # so downstream plotting sees the usual MODZ_SCORE layout
                self.metrics[p]['MODZ_SCORE'] = {0: combined_modz}
            else:
                # calculating modifed z-score
                modz_dict, bd_inds = self.iterative_mod_zscore(
                    autos_amps_norm[:, :, pol_dict[p]], threshold=self.cutoff_threshold, niter=self.niter)
                bd_inds = np.array(bd_inds)
                self.metrics[p]['MODZ_SCORE'] = modz_dict
            if len(bd_inds) > 0:
                bad_ants.append(self.metrics['ANNUMBERS'][bd_inds])

            # writing stars to metrics instanc
            self.metrics[p]['RMS'] = rms
            self.metrics[p]['BAD_ANTS'] = list(
                itertools.chain.from_iterable(bad_ants))

        # combining bad antennas from both pols to determine if the observation
        # should be considered for processing or not.
        # If %bad_ants > 50, obs is discarded
        self.metrics['BAD_ANTS'] = np.unique(
            self.metrics['XX']['BAD_ANTS'] + self.metrics['YY']['BAD_ANTS'])
        nants = self.metrics['NANTS'] - len(self.ex_annumbers)
        percent_bdants = len(self.metrics['BAD_ANTS']) / nants * 100
        self.metrics['BAD_ANTS_PERCENT'] = percent_bdants
        self.metrics['STATUS'] = 'GOOD' if percent_bdants < 50 else 'BAD'
        self.metrics['THRESHOLD'] = self.cutoff_threshold

    def write_to(self, outfile=None):
        if outfile is None:
            outfile = self.uvfits_path.replace(
                '.uvfits', '_prepvis_metrics.json')
        ju.write_metrics(self.metrics, outfile)

    def write_spectra(self, outfile=None):
        """
        Per-antenna, per-channel RMS across time, normalised by the per-channel
        array median. Too large for the metrics json, so it is written
        alongside it for plot_prepvis_groups.py to pick up.
        """
        if outfile is None:
            outfile = self.uvfits_path.replace(
                '.uvfits', '_prepvis_spectra.npz')
        amps = np.abs(self.autos())
        rms_tf = np.sqrt(np.nanmean(amps ** 2, axis=0))
        normed = rms_tf / np.nanmedian(rms_tf, axis=0)[np.newaxis, :, :]
        np.savez_compressed(
            outfile, spectra=normed[:, :, :2].astype(np.float32),
            annumbers=np.asarray(self.antenna_numbers))
        return outfile
