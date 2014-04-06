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
Testing module for the lazy_filters module by using Numpy/Scipy as oracles
"""

import pytest
p = pytest.mark.parametrize

from scipy.signal import lfilter
from scipy.optimize import fminbound
from math import cos, pi, sqrt
from numpy import mat

# Audiolazy internal imports
from ..lazy_filters import ZFilter, resonator, z
from ..lazy_misc import almost_eq
from ..lazy_compat import orange, xrange, xzip
from ..lazy_math import dB20
from ..lazy_itertools import repeat, cycle, count
from ..lazy_stream import Stream


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


class TestZFilterMatrixNumpy(object):

  def test_matrix_coefficients_multiplication(self):
    m = mat([[1, 2], [2, 2]])
    n1 = mat([[1.2, 3.2], [1.2, 1.1]])
    n2 = mat([[-1, 2], [-1, 2]])
    a = mat([[.3, .4], [.5, .6]])

    # Time-varying filter with 2x2 matrices as coeffs
    mc = repeat(m)
    nc = cycle([n1, n2])
    ac = repeat(a)
    filt = (mc + nc * z ** -1) / (1 - ac * z ** -1)

    # For a time-varying 2x3 matrix signal
    data = [
      Stream(1, 2),
      count(),
      count(start=1, step=2),
      cycle([.2, .33, .77, pi, cos(3)]),
      repeat(pi),
      count(start=sqrt(2), step=pi/3),
    ]

    data_copy = [el.copy() for el in data]
    sig = Stream(mat(vect).reshape(2, 3) for vect in xzip(*data))
    zero = mat([[0, 0, 0], [0, 0, 0]])
    result = filt(sig, zero=zero).limit(30)

    in_sample = old_out_sample = zero
    n, not_n = n1, n2
    for expected_out_sample in result:
      old_in_sample = in_sample
      in_sample = mat([s.take() for s in data_copy]).reshape(2, 3)
      out_sample = m * in_sample + n * old_in_sample + a * old_out_sample
      assert almost_eq(out_sample.tolist(), expected_out_sample.tolist())
      n, not_n = not_n, n
      old_out_sample = out_sample
