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

# Audiolazy internal imports
from .lazy_stream import tostream


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
