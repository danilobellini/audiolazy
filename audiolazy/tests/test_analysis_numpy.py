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
Testing module for the lazy_analysis module by using Numpy
"""

from __future__ import division

import pytest
p = pytest.mark.parametrize

from numpy.fft import fft, ifft

# Audiolazy internal imports
from ..lazy_analysis import dft, stft
from ..lazy_math import pi, cexp, phase
from ..lazy_misc import almost_eq, rint
from ..lazy_synth import line
from ..lazy_stream import Stream
from ..lazy_itertools import chain


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
    np_data = fft(blk, size).tolist()
    lz_data = dft(blk[:size],
                  line(size, 0, 2 * pi, finish=False),
                  normalize=False
                 )
    assert almost_eq.diff(np_data, lz_data, max_diff=1e-12)


class TestSTFT(object):

  @p("strategy", [stft.real, stft.complex, stft.complex_real])
  def test_whitenize_with_decorator_size_4_without_fft_window(self, strategy):
    @strategy(size=4, hop=4)
    def whitenize(blk):
      return cexp(phase(blk) * 1j)

    sig = Stream(0, 3, 4, 0) # fft([0, 3, 4, 0]) is [7, -4.-3.j, 1, -4.+3.j]
                             # fft([4, 0, 0, 3]) is [7,  4.+3.j, 1,  4.-3.j]
    data4003 = ifft([1,  (4+3j)/5, 1,  (4-3j)/5]) # From block [4, 0, 0, 3]
    data0340 = ifft([1, (-4-3j)/5, 1, (-4+3j)/5]) # From block [0, 3, 4, 0]

    result = whitenize(sig) # No overlap-add window (default behavior)
    assert isinstance(result, Stream)

    expected = Stream(*data0340)
    assert almost_eq(result.take(64), expected.take(64))

    # Using a "triangle" as the overlap-add window
    wnd = [0.5, 1, 1, 0.5] # Normalized triangle
    new_result = whitenize(sig, ola_wnd=[1, 2, 2, 1])
    assert isinstance(result, Stream)
    new_expected = Stream(*data0340) * Stream(*wnd)
    assert almost_eq(new_result.take(64), new_expected.take(64))

    # With real overlap
    wnd_hop2 = [1/3, 2/3, 2/3, 1/3] # Normalized triangle for the new hop
    overlap_result = whitenize(sig, hop=2, ola_wnd=[1, 2, 2, 1])
    assert isinstance(result, Stream)
    overlap_expected = Stream(*data0340) * Stream(*wnd_hop2) \
                     + chain([0, 0], Stream(*data4003) * Stream(*wnd_hop2))
    assert almost_eq(overlap_result.take(64), overlap_expected.take(64))
