# -*- coding: utf-8 -*-
"""
Audio analysis and block processing module

Copyright (C) 2012 Danilo de Jesus da Silva Bellini

This file is part of AudioLazy, the signal processing Python package.

AudioLazy is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

Created on Sun Jul 29 2012
danilo [dot] bellini [at] gmail [dot] com
"""

from math import cos, pi

# Audiolazy internal imports
from .lazy_core import StrategyDict
from .lazy_stream import tostream

__all__ = ["window", "zcross"]


window = StrategyDict("window")


@window.strategy("hamming")
def window(size):
  """
  Hamming window function with the given size.

  Returns
  -------
  List with the desired window samples. Max value is one (1.0).

  """

  if size == 1:
    return [1.0]
  return [.54 - .46 * cos(2 * pi * n / (size - 1))
          for n in xrange(size)]


@window.strategy("rectangular", "rect")
def window(size):
  """
  Rectangular window function with the given size.

  Returns
  -------
  List with the desired window samples. All values are ones (1.0).

  """
  return [1.0 for n in xrange(size)]


@window.strategy("bartlett")
def window(size):
  """
  Bartlett (triangular with zero-valued endpoints) window function with the
  given size.

  Returns
  -------
  List with the desired window samples. Max value is one (1.0).

  See Also
  --------
  window.triangular :
    Triangular with no zero end-point.

  """
  if size == 1:
    return [1.0]
  return [1 - 2.0 / (size - 1) * abs(n - (size - 1) / 2.0)
          for n in xrange(size)]


@window.strategy("triangular", "triangle")
def window(size):
  """
  Triangular (with no zero end-point) window function with the given size.

  Returns
  -------
  List with the desired window samples. Max value is one (1.0).

  See Also
  --------
  window.bartlett :
    Bartlett window, triangular with zero-valued end-points.

  """
  if size == 1:
    return [1.0]
  return [1 - 2.0 / (size + 1) * abs(n - (size - 1) / 2.0)
          for n in xrange(size)]


@window.strategy("hann", "hanning")
def window(size):
  """
  Hann window function with the given size.

  Returns
  -------
  List with the desired window samples. Max value is one (1.0).

  """
  if size == 1:
    return [1.0]
  return [.5 * (1 - cos(2 * pi * n / (size - 1))) for n in xrange(size)]


@window.strategy("blackman")
def window(size, alpha=.16):
  """
  Blackman window function with the given size.

  Parameters
  ----------
  size :
    Window size in samples.
  alpha :
    Blackman window alpha value. Defaults to 0.16.

  Returns
  -------
  List with the desired window samples. Max value is one (1.0).

  """
  if size == 1:
    return [1.0]
  return [alpha / 2 * cos(4 * pi * n / (size - 1))
          -.5 * cos(2 * pi * n / (size - 1)) + (1 - alpha) / 2
          for n in xrange(size)]


@tostream
def zcross(seq, hysteresis=0, first_sign=0):
  """
  Zero-crossing stream.

  Parameters
  ----------
  seq :
    Any iterable to be used as input for the zero crossing analysis
  hysteresis :
    Crossing exactly zero might happen many times too fast due to high
    frequency oscilations near zero. To avoid this, you can make two
    threshold limits for the zero crossing detection: ``hysteresis`` and
    ``-hysteresis``. Defaults to zero (0), which means no hysteresis and only
    one threshold.
  first_sign :
    Optional argument with the sign memory from past. Gets the sig from any
    signed number. Defaults to zero (0), which means "any", and the first sign
    will be the first one found in data.

  Returns
  -------
  A Stream instance that outputs 1 for each crossing detected, 0 otherwise.

  """
  neg_hyst = -hysteresis
  seq_iter = iter(seq)

  # Gets the first sign
  if first_sign == 0:
    last_sign = 0
    for el in seq_iter:
      yield 0
      if (el > hysteresis) or (el < neg_hyst): # Ignores hysteresis region
        last_sign = -1 if el < 0 else 1 # Define the first sign
        break
  else:
    last_sign = -1 if first_sign < 0 else 1

  # Finds the full zero-crossing sequence
  for el in seq_iter: # Keep the same iterator (needed for non-generators)
    if el * last_sign < neg_hyst:
      last_sign = -1 if el < 0 else 1
      yield 1
    else:
      yield 0
