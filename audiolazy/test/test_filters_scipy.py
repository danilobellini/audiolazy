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
# Created on Sat Oct 06 2012
# danilo [dot] bellini [at] gmail [dot] com
"""
Testing module for the lazy_filters module by using scipy as an oracle
"""

import pytest
p = pytest.mark.parametrize

from scipy.signal import lfilter
from scipy.optimize import fminbound
from math import cos, pi, sqrt

# Audiolazy internal imports
from ..lazy_filters import ZFilter, resonator
from ..lazy_misc import almost_eq
from ..lazy_compat import orange, xrange
from ..lazy_math import dB20


class TestZFilterScipy(object):

  @p("a", [[1.], [3.], [1., 3.], [15., -17.2], [-18., 9.8, 0., 14.3]])
  @p("b", [[1.], [-1.], [1., 0., -1.], [1., 3.]])
  @p("data", [orange(5), orange(5, 0, -1), [7, 22, -5], [8., 3., 15.]])
  def test_lfilter(self, a, b, data):
    filt = ZFilter(b, a)
    expected = lfilter(b, a, data).tolist()
    assert almost_eq(filt(data), expected)


class TestResonatorScipy(object):

  @p("func", resonator)
  @p("freq", [pi * k / 9 for k in xrange(1, 9)])
  @p("bw", [pi / 23, pi / 31])
  def test_max_gain_is_at_resonance(self, func, freq, bw):
    names = func.__name__.split("_")
    filt = func(freq, bw)
    resonance_freq = fminbound(lambda x: -dB20(filt.freq_response(x)),
                               0, pi, xtol=1e-10)
    resonance_gain = dB20(filt.freq_response(resonance_freq))
    assert almost_eq.diff(resonance_gain, 0., max_diff=1e-12)

    if "freq" in names: # Given frequency is at the denominator
      R = sqrt(filt.denominator[2])
      assert 0 < R < 1
      cosf = cos(freq)
      cost = -filt.denominator[1] / (2 * R)
      assert almost_eq(cosf, cost)

      if "z" in names:
        cosw = cosf * (2 * R) / (1 + R ** 2)
      elif "poles" in names:
        cosw = cosf * (1 + R ** 2) / (2 * R)

      assert almost_eq(cosw, cos(resonance_freq))

    else: # Given frequency is the resonance frequency
      assert almost_eq(freq, resonance_freq)
