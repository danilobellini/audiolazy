#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of AudioLazy, the signal processing Python package.
# Copyright (C) 2012-2014 Danilo de Jesus da Silva Bellini
#
# AudioLazy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# -*- coding: utf-8 -*-
# Created on Fri Jul 11 03:37:54 2014
# danilo [dot] bellini [at] gmail [dot] com
"""
Partial recreation of the "Windows and Figures of Merit" F. Harris comparison
table and window plots.

The original is the Table 1 found in:

  ``Harris, F. J. "On the Use of Windows for Harmonic Analysis with the
  Discrete Fourier Transform". Proceedings of the IEEE, vol. 66, no. 1,
  January 1978.``
"""

from __future__ import division

from audiolazy import (window, rst_table, Stream, line, cexp, dB10, dB20,
                       zcross, iteritems, pi)
import matplotlib.pyplot as plt
from numpy.fft import rfft
import numpy as np
from collections import OrderedDict


def enbw(wnd):
  """ Equivalent Noise Bandwidth in bins (Processing Gain reciprocal). """
  return sum(el ** 2 for el in wnd) / sum(wnd) ** 2 * len(wnd)

def coherent_gain(wnd):
  """ Coherent Gain, normalized by len(wnd). """
  return sum(wnd) / len(wnd)

def overlap_correlation(wnd, hop):
  """ Overlap correlation percent for the given overlap hop in samples. """
  return sum(wnd * Stream(wnd).skip(hop)) / sum(el ** 2 for el in wnd)

def scalloping_loss(wnd):
  """ Positive number with the scalloping loss in dB. """
  return -dB20(abs(sum(wnd * cexp(line(len(wnd), 0, -1j * pi)))) / sum(wnd))

def processing_loss(wnd):
  """ Positive number with the ENBW (processing loss) in dB. """
  return dB10(enbw(wnd))

def worst_case_processing_loss(wnd):
  return scalloping_loss(wnd) + processing_loss(wnd)


def find_xdb_bin(wnd, power=.5, res=1500):
  """
  A not so fast way to find the x-dB cutoff frequency "bin" index.

  Parameters
  ----------
  wnd:
    The window itself as an iterable.
  power:
    The power value (squared amplitude) where the x-dB value should lie,
    using ``x = dB10(power)``.
  res :
    Zero-padding factor. 1 for no zero-padding, 2 for twice the length, etc..
  """
  spectrum = dB20(rfft(wnd, res * len(wnd)))
  root_at_xdb = spectrum - spectrum[0] - dB10(power)
  return next(i for i, el in enumerate(zcross(root_at_xdb)) if el) / res


def get_peaks(blk, neighbors=2):
  """
  Get all peak indices in blk (sorted by index value) but the ones at the
  vector limits (first and last ``neighbors - 1`` values). A peak is the max
  value in a neighborhood of ``neighbors`` values for each side.
  """
  size = 1 + 2 * neighbors
  pairs = enumerate(Stream(blk).blocks(size=size, hop=1).map(list), neighbors)
  for idx, nbhood in pairs:
    center = nbhood.pop(neighbors)
    if all(center >= el for el in nbhood):
      yield idx
      next(pairs) # Skip ones we already know can't be peaks
      next(pairs)


def hsll(wnd, res=20, neighbors=2):
  """
  Highest Side Lobe Level (dB).

  Parameters
  ----------
  res :
    Zero-padding factor. 1 for no zero-padding, 2 for twice the length, etc..
  neighbors :
    Number of neighbors needed by ``get_peaks`` to define a peak.
  """
  spectrum = dB20(rfft(wnd, res * len(wnd)))
  first_peak = next(get_peaks(spectrum, neighbors=neighbors))
  return max(spectrum[first_peak:]) - spectrum[0]


def to_string(el):
  return "%01.2f" % el if isinstance(el, float) else el


table_wnds = OrderedDict([
  ("Rectangle", window.rect),
  ("Triangle", window.bartlett),
  ("Cosine", window.cos),
  ("Hann", window.hann),
  ("Cosine^3", lambda size: window.cos(size, 3)),
  ("Cosine^4", lambda size: window.cos(size, 4)),
  ("Hamming", window.hamming),
  ("Exact Blackman", lambda size: window.blackman(size, 2. * 1430 / 18608)),
  ("Blackman", window.blackman),
])


schema = OrderedDict([
  ("name", "Window"), # Window name
  ("hsll", "HSLL"), # Highest Side Lobe Level (dB)
  ("cg", "CG"), # Coherent gain
  ("enbw", "ENBW"), # Equivalent Noise Bandwidth (bins)
  ("bw3", "3dB BW"), # 50% power bandwidth (bins)
  ("scallop", "Scallop"), # Scallop loss (dB)
  ("wcpl", "Worst PL"), # Worst case process loss (dB)
  ("bw6", "6dB BW"), # 25% power bandwidth (bins)
  ("ol75", "75% OL"), # 75% overlap correlation (percent)
  ("ol50", "50% OL"), # 50% overlap correlation (percent)
])


size = 50 # Must be even!
full_size = 20 * size
table = []
for name, wnd_func in iteritems(table_wnds):
  wnd = wnd_func(size)
  spectrum = dB20(rfft(wnd, full_size))

  wnd_full = wnd_func(full_size)
  wnd_data = {
    "name": name,
    "hsll": hsll(wnd_full),
    "cg": coherent_gain(wnd_full),
    "enbw": enbw(wnd_full),
    "bw3": 2 * find_xdb_bin(wnd, .5),
    "scallop": scalloping_loss(wnd_full),
    "wcpl": worst_case_processing_loss(wnd_full),
    "bw6": 2 * find_xdb_bin(wnd, .25),
    "ol75": overlap_correlation(wnd_full, .25 * full_size) * 100,
    "ol50": overlap_correlation(wnd_full, .5 * full_size) * 100,
  }
  table.append([to_string(wnd_data[k]) for k in schema])

  wnd_symm = wnd + [wnd[0]]
  full_spectrum = np.hstack([spectrum[::-1], spectrum[1:-1]]) - spectrum[0]

  fig, (time_ax, freq_ax) = plt.subplots(2, 1, num=name)
  time_ax.vlines(np.arange(- size // 2, size // 2 + 1), 0, wnd_symm)
  time_ax.set(xlim=(-(size // 2), size // 2), ylim=(-.1, 1.1),
              xlabel="Time (samples)", title=name)
  freq_ax.plot(list(line(full_size, -1, 1)), full_spectrum)
  freq_ax.set(xlim=(-1, 1), ylim=(-90, 0), ylabel="dB",
              xlabel="Frequency (% of the Nyquist frequency)")
  fig.tight_layout()

print(__doc__)
for row in rst_table(table, schema.values()):
  print(row) # Some values aren't the same to the paper, though

plt.show()