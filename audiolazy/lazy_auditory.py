#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Peripheral auditory modeling module

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

Created on Fri Sep 21 2012
danilo [dot] bellini [at] gmail [dot] com
"""

# Audiolazy internal imports
from .lazy_core import StrategyDict
from .lazy_misc import elementwise


erb = StrategyDict()

@erb.strategy("gm90", "GM90", "GlasbergMoore90", "GlasbergMoore")
@elementwise("freq", 0)
def erb(freq):
  """
    ERB model from:

      ``B. R. Glasberg and B. C. J. Moore, "Derivation of auditory filter
      shapes from notched-noise data". Hearing Research, vol. 47, 1990, pp.
      103-108.``

    Both input and output are given in Hz.
  """
  return 24.7 * (4.37e-3 * freq + 1.)

@erb.strategy("mg83", "MG83", "MooreGlasberg83")
@elementwise("freq", 0)
def erb(freq):
  """
    ERB model from:

      ``B. C. J. Moore and B. R. Glasberg, "Suggested formulae for calculating
      auditory filter bandwidths and excitation patterns". J. Acoust. Soc.
      Am., 74, 1983, pp. 750-753.``

    Both input and output are given in Hz.
  """
  return 6.23e-6 * freq ** 2 + 93.39e-3 * freq + 28.52
