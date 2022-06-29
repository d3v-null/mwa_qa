from mwa_qa import read_metafits as rm
from collections import OrderedDict
from astropy.io import fits
import numpy as np
import copy

class Csoln(object):
	def __init__(self, calfile, metafits=None, pol='X'):
		"""
		Object takes in a calfile in fits format and extracts bit and pieces of the required informations
 		- calfile : Fits file readable by astropy containing calibration solutions (support for hyperdrive
				   output only for now) and related information
		- metafits : Metafits with extension *.metafits containing information corresponding to the observation
					 for which the calibration solutions is derived
		- pol : Polarization, can be either 'X' or 'Y'. It should be specified so that information associated 
                with the given pol is provided. Default is 'X'
		"""
		self.calfile = calfile
		self.Metafits = rm.Metafits(metafits, pol)

	def data(self, hdu):
		"""
		Returns the data stored in the specified HDU column of the image
		hdu : hdu column, ranges from 0 to 5
			  0 - the calibration solution
			  1 - tiles information (antenna, tilename, flag)
			  2 - chanblocks (index, freq, flag)
			  3 - calibration results (timeblock, chan, convergence)
			  4 - weights used for each baseline
			  for more details refer to https://mwatelescope.github.io/mwa_hyperdrive/defs/cal_sols_hyperdrive.html
		"""
		return fits.open(self.calfile)[hdu].data

	def real(self):
		"""
		Returns the real part of the calibration solutions
		"""
		data = self._read_data()
		return data[:, :, :, ::2]

	def imag(self):
		"""
		Returns the imaginary part of the calibration solutions
		"""
		data = self._read_data()
		return data[:, :, :, 1::2]
	
	def gains(self):
		"""
        Combines the real and imaginary parts to form the 4 polarization (xx, xy, yx and yy)
        """
		return self.real() + self.imag() * 1j

	def gains_shape(self):
		"""
		Returns shape of the array containing the gain soultions
		"""
		return self.gains().shape

	def header(self, hdu):
		"""
		Returns the header of the specified HDU column
		- hdu : hdu column, ranges from 0 to 5
        """
		return fits.open(self.calfile)[0].header

	def _check_ref_tile_data(self, tile_ind):
		"""
		Checks if the given reference antenna is flagged due to non-convergence or any 
		malfunctioning reports
		- tile_ind : Index of the reference tile
		"""
		gains = self.gains()
		assert not np.isnan(gains[:, tile_ind, :, :]).all(), "The specified reference antenna seems to be flagged. Choose a different reference antenna"

	def _normalized_data(self, data, ref_tile_id=None):
		"""
		Normalizes the gain solutions for each timeblock given a reference tile
		- data : the input array of shape( tiles, freq, pols) containing the solutions
		- ref_tile_id: Tile ID of the reference tile e.g Tile 103. Default is set to the last antenna of the telescope.
						For example for MWA128T, the reference antennat is Tile 168
		"""
		if ref_tile_id is None:
			ref_tile_ind = -1
		else:
			ref_tile_ind = self.Metafits.get_tile_ind(ref_tile_id)[0]
		self._check_ref_tile_data(ref_tile_ind)
		refs = []
		for ref in data[ref_tile_ind].reshape((-1, 2, 2)):
			refs.append(np.linalg.inv(ref))
		refs = np.array(refs)
		div_ref = []
		for tile_i in data:
			for (i, ref) in zip(tile_i, refs):
				div_ref.append(i.reshape((2, 2)).dot(ref))
		return np.array(div_ref).reshape(data.shape)

	def normalized_gains(self, ref_tile_id=None):
		"""
		Returns the normalized gain solutions using the given reference antenna
		- ref_tile_id: Tile ID of the reference tile e.g Tile 103. Default is set to the last antenna of the telescope.
                       For example for MWA128T, the reference antennat is Tile 168
		"""
		gains = self.gains()
		ngains = copy.deepcopy(gains)
		for t in range(len(ngains)):
			ngains[t] = self._normalized_data(gains[t], ref_tile_id)
		return ngains

	def _select_gains(self, norm):
		"""
		Return normalized if norm is True else unnomalized gains 
		- norm : boolean, If True returns normalized gains else unormalized gains.
		"""
		if norm:
			return self.normalized_gains()
		else:
			return self.gains()

	def amplitudes(self, norm=True):
		"""
		Returns amplitude of the normalized gain solutions
		- norm : boolean, If True returns normalized gains else unormalized gains.
        		 Default is set to True.
		"""
		gains = self._select_gains(norm = norm)
		return np.abs(gains)

	def phases(self, norm=True):
		"""
		Returns phases in degrees of the normalized gain solutions
		- norm : boolean, If True returns normalized gains else unormalized gains.
				 Default is set to True.
		"""
		gains = self._select_gains(norm = norm)
		return np.angle(gains) * 180 / np.pi

	def gains_for_tile(self, tile_id, norm=True):
		"""
		Returns gain solutions for the given tile ID
		- tile_id : Tile ID e.g Tile103
		- norm : boolean, If True returns normalized gains else unormalized gains.
                 Default is set to True.
		"""
		gains = self._select_gains(norm = norm)
		ind = self.Metafits.get_tile_ind(tile_id)
		return gains[:, ind, :, :] 
	
	def gains_for_receiver(self, receiver, norm=True):
		"""
		Returns the dictionary of gains solutions for all the tiles (8 tiles) connected to the given reciver
		"""
		tile_ids = self.Metafits.get_tiles_for_receiver(receiver)
		gains_receiver = OrderedDict()
		for tile_id in tile_ids:
			gains_receiver[tile_id] = self.gains_for_tile(tile_id, norm = norm)
		return gains_receiver

	def gains_for_tilepair(self, tilepair, norm=True):
		"""
		Evaluates conjugation of the gain solutions for antenna pair (tile0, tile1)
		- tile_pair : tuple of tile numbers such as (11, 13)
		"""
		tile0, tile1 = 'Tile{:03g}'.format(tilepair[0]), 'Tile{:03g}'.format(tilepair[1])
		gains_t0 = self.gains_for_tile(tile0, norm = norm)
		gains_t1 = self.gains_for_tile(tile1, norm = norm)
		return gains_t0 * np.conj(gains_t1)
