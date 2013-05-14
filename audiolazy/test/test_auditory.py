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
# Created on Sun Sep 09 2012
# danilo [dot] bellini [at] gmail [dot] com
"""
Testing module for the lazy_auditory module
"""

import pytest
p = pytest.mark.parametrize

import itertools as it

# Audiolazy internal imports
from ..lazy_auditory import erb, gammatone_erb_constants, gammatone
from ..lazy_misc import almost_eq, sHz
from ..lazy_math import pi
from ..lazy_filters import CascadeFilter
from ..lazy_stream import Stream


class TestERB(object):

  @p(("freq", "bandwidth"),
     [(1000, 132.639),
      (3000, 348.517),
     ])
  def test_glasberg_moore_slaney_example(self, freq, bandwidth):
    assert almost_eq.diff(erb["gm90"](freq), bandwidth, max_diff=5e-4)

  @p("erb_func", erb)
  @p("rate", [8000, 22050, 44100])
  @p("freq", [440, 20, 2e4])
  def test_two_input_methods(self, erb_func, rate, freq):
    Hz = sHz(rate)[1]
    assert almost_eq(erb_func(freq) * Hz, erb_func(freq * Hz, Hz))
    if freq < rate:
      with pytest.raises(ValueError):
        erb_func(freq * Hz)


class TestGammatoneERBConstants(object):

  @p(("n", "an",  "aninv", "cn",  "cninv"), # Some paper values were changed:
     [(1,   3.142, 0.318,   2.000, 0.500),  # + a1 was 3.141 (it should be pi)
      (2,   1.571, 0.637,   1.287, 0.777),  # + a2 was 1.570, c2 was 1.288
      (3,   1.178, 0.849,   1.020, 0.981),  # + 1/c3 was 0.980
      (4,   0.982, 1.019,   0.870, 1.149),
      (5,   0.859, 1.164,   0.771, 1.297),  # + a5 was 0.889 (typo?), c5 was
      (6,   0.773, 1.293,   0.700, 1.429),  #   0.772 and 1/c5 was 1.296
      (7,   0.709, 1.411,   0.645, 1.550),  # + c7 was 0.646
      (8,   0.658, 1.520,   0.602, 1.662),  # Doctests also suffered from this
      (9,   0.617, 1.621,   0.566, 1.767)   # rounding issue.
     ])
  def test_annex_c_table_1(self, n, an, aninv, cn, cninv):
    x, y = gammatone_erb_constants(n)
    assert almost_eq.diff(x, aninv, max_diff=5e-4)
    assert almost_eq.diff(y, cn, max_diff=5e-4)
    assert almost_eq.diff(1./x, an, max_diff=5e-4)
    assert almost_eq.diff(1./y, cninv, max_diff=5e-4)


class TestGammatone(object):

  some_data = [pi / 7, Stream(0, 1, 2, 1), [pi/3, pi/4, pi/5, pi/6]]

  @p(("filt_func", "freq", "bw"),
     [(gf, pi / 5, pi / 19) for gf in gammatone] +
     [(gammatone.klapuri, freq, bw) for freq, bw
                                    in it.product(some_data,some_data)]
    )
  def test_number_of_poles_order(self, filt_func, freq, bw):
    cfilt = filt_func(freq=freq, bandwidth=bw)
    assert isinstance(cfilt, CascadeFilter)
    assert len(cfilt) == 4
    for filt in cfilt:
      assert len(filt.denominator) == 3
