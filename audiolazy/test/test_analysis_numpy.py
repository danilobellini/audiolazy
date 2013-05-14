# -*- coding: utf-8 -*-
# This file is part of AudioLazy, the signal processing Python package.
# Copyright (C) 2012-2013 Danilo de Jesus da Silva Bellini
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
# Created on Mon Feb 25 2013
# danilo [dot] bellini [at] gmail [dot] com
"""
Testing module for the lazy_analysis module by using numpy as an oracle
"""

import pytest
p = pytest.mark.parametrize

from numpy.fft import fft as np_fft

# Audiolazy internal imports
from ..lazy_analysis import dft
from ..lazy_math import pi
from ..lazy_misc import almost_eq, rint
from ..lazy_synth import line


class TestDFT(object):

  blk_table = [
    [20],
    [1, 2, 3],
    [0, 1, 0, -1],
    [5] * 8,
  ]

  @p("blk", blk_table)
  @p("size_multiplier", [.5, 1, 2, 3, 1.5, 1.2])
  def test_empty(self, blk, size_multiplier):
    full_size = len(blk)
    size = rint(full_size * size_multiplier)
    np_data = np_fft(blk, size).tolist()
    lz_data = dft(blk[:size],
                  line(size, 0, 2 * pi, finish=False),
                  normalize=False
                 )
    assert almost_eq.diff(np_data, lz_data, max_diff=1e-12)
