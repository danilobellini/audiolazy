#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of AudioLazy, the signal processing Python package.
# Copyright (C) 2012-2016 Danilo de Jesus da Silva Bellini
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
"""
Partial recreation of the "Windows and Figures of Merit" F. Harris comparison
table and window plots.

The original is the Table 1 found in:

  ``Harris, F. J. "On the Use of Windows for Harmonic Analysis with the
  Discrete Fourier Transform". Proceedings of the IEEE, vol. 66, no. 1,
  January 1978.``
"""

from __future__ import division, print_function

from audiolazy import (window, rst_table, Stream, line, cexp, dB10, dB20,
                       zcross, iteritems, pi, z, inf)
import matplotlib.pyplot as plt
from numpy.fft import rfft
import numpy as np
import scipy.optimize as so
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


def slfo(wnd, res=50, neighbors=2, max_miss=.7, start_delta=1e-4):
  """
  Side Lobe Fall Off (dB/oct).

  Finds the side lobe peak fall off numerically in dB/octave by using the
  ``scipy.optimize.fmin`` function.

  Hint
  ----
  Originally, Harris rounded the results he found to a multiple of -6, you can
  use the AudioLazy ``rint`` function for that: ``rint(falloff, 6)``.

  Parameters
  ----------
  res :
    Zero-padding factor. 1 for no zero-padding, 2 for twice the length, etc..
  neighbors :
    Number of neighbors needed by ``get_peaks`` to define a peak.
  max_miss :
    Maximum percent of peaks that might be missed when approximating them
    by a line.
  start_delta :
    Minimum acceptable value for an orthogonal deviation from the
    approximation line to include a peak.
  """
  # Finds all side lobe peaks, to find the "best" line for it afterwards
  spectrum = dB20(rfft(wnd, res * len(wnd)))
  peak_indices = list(get_peaks(spectrum, neighbors=neighbors))
  log2_peak_indices = np.log2(peak_indices) # Base 2 ensures result in dB/oct
  peaks = spectrum[peak_indices]
  npeaks = len(peak_indices)

  # This length (actually, twice the length) is the "weight" of each peak
  lengths = np.array([0] + (1 - z **-2)(log2_peak_indices).skip(2).take(inf) +
                     [0]) # Extreme values weights to zero
  max_length = sum(lengths)

  # First guess for the polynomial "a*x + b" is at the center
  idx = np.searchsorted(log2_peak_indices,
                        .5 * (log2_peak_indices[-1] + log2_peak_indices[0]))
  a = ((peaks[idx+1] - peaks[idx]) /
       (log2_peak_indices[idx+1] - log2_peak_indices[idx]))
  b = peaks[idx] - a * log2_peak_indices[idx]

  # Scoring for the optimization function
  def score(vect, show=False):
    a, b = vect
    h = start_delta * (1 + a ** 2) ** .5 # Vertical deviation

    while True:
      pdelta = peaks - (a * log2_peak_indices + b)
      peaks_idx_included = np.nonzero((pdelta < h) & (pdelta > -h))
      missing = npeaks - len(peaks_idx_included[0])
      if missing < npeaks * max_miss:
        break
      h *= 2

    pdelta_included = pdelta[peaks_idx_included]
    real_delta = max(pdelta_included) - min(pdelta_included)
    total_length = sum(lengths[peaks_idx_included])

    if show: # For debug
      print(real_delta, len(peaks_idx_included[0]))

    return -total_length / max_length + 4 * real_delta ** .5

  a, b = so.fmin(score, [a, b], xtol=1e-12, ftol=1e-12, disp=False)

#  # For Debug only
#  score([a, b], show=True)
#  plt.figure()
#  plt.plot(log2_peak_indices, peaks, "x-")
#  plt.plot(log2_peak_indices, a * log2_peak_indices + b)
#  plt.show()

  return a


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


has_separator_before = ["Cosine", "Hamming", "Exact Blackman"]


schema = OrderedDict([
  ("name", "Window"), # Window name
  ("hsll", "SLobe"), # Highest Side Lobe Level (dB)
  ("slfo", "Falloff"), # Side Lobe Fall Off (dB/oct)
  ("cg", "CGain"), # Coherent gain
  ("enbw", "ENBW"), # Equivalent Noise Bandwidth (bins)
  ("bw3", "BW3dB"), # 50% power bandwidth (bins)
  ("scallop", "Scallop"), # Scallop loss (dB)
  ("wcpl", "Worst"), # Worst case process loss (dB)
  ("bw6", "BW6dB"), # 25% power bandwidth (bins)
  ("ol75", "OL75%"), # 75% overlap correlation (percent)
  ("ol50", "OL50%"), # 50% overlap correlation (percent)
])


schema_full = OrderedDict([
  ("name", "Window name"),
  ("hsll", "Highest Side Lobe Level (dB)"),
  ("slfo", "Side Lobe Fall Off (dB/oct)"),
  ("cg", "Coherent gain"),
  ("enbw", "Equivalent Noise Bandwidth (bins)"),
  ("bw3", "50% power bandwidth (bins)"),
  ("scallop", "Scallop loss (dB)"), #
  ("wcpl", "Worst case process loss (dB)"), #
  ("bw6", "25% power bandwidth (bins)"), #
  ("ol75", "75% overlap correlation (percent)"), #
  ("ol50", "50% overlap correlation (percent)"), #
])

size = 50 # Must be even!
full_size = 20 * size
table = []
for name, wnd_func in iteritems(table_wnds):
  if name in has_separator_before:
    table.append([".."] + [""] * (len(schema) - 1))

  wnd = wnd_func(size)
  spectrum = dB20(rfft(wnd, full_size))

  wnd_full = wnd_func(full_size)
  wnd_data = {
    "name": name,
    "hsll": hsll(wnd_full),
    "slfo": slfo(wnd_full),
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

  smallest_peak_idx = min(get_peaks(spectrum), key=spectrum.__getitem__)
  ymin = (spectrum[smallest_peak_idx] - spectrum[0] - 5) // 10 * 10

  fig, (time_ax, freq_ax) = plt.subplots(2, 1, num=name)
  time_ax.vlines(np.arange(- size // 2, size // 2 + 1), 0, wnd_symm)
  time_ax.set(xlim=(-(size // 2), size // 2), ylim=(-.1, 1.1),
              xlabel="Time (samples)", title=name)
  freq_ax.plot(list(line(full_size, -1, 1)), full_spectrum)
  freq_ax.set(xlim=(-1, 1), ylim=(ymin, 0), ylabel="dB",
              xlabel="Frequency (% of the Nyquist frequency)")
  fig.tight_layout()

# Prints the table and other text contents
print(__doc__)
print("""
Schema
------
""")
for row in rst_table([(v, schema_full[k]) for k, v in iteritems(schema)],
           ["Column", "Description"]):
  print(row)
print("""
Windows and Figures of Merit
----------------------------
""")
for row in rst_table(table, schema.values()):
  print(row) # Some values aren't the same to the paper, though

plt.ioff()
plt.show()