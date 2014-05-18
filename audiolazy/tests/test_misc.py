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
# Created on Tue Jul 31 2012
# danilo [dot] bellini [at] gmail [dot] com
"""
Testing module for the lazy_misc module
"""

import pytest
p = pytest.mark.parametrize

import itertools as it

# Audiolazy internal imports
from ..lazy_misc import rint, elementwise, freq2lag, lag2freq, almost_eq
from ..lazy_compat import INT_TYPES, orange, xrange


class TestRInt(object):
  table = [
    (.499, 0),
    (-.499, 0),
    (.5, 1),
    (-.5, -1),
    (1.00001e3, 1000),
    (-227.0090239, -227),
    (-12.95, -13),
  ]

  @p(("data", "expected"), table)
  def tests_from_table_default_step(self, data, expected):
    result = rint(data)
    assert isinstance(result, INT_TYPES)
    assert result == expected

  @p(("data", "expected"), table)
  @p("n", [2, 3, 10])
  def tests_from_step_n(self, data, expected, n):
    data_n, expected_n = n * data, n * expected # Inputs aren't in step n
    result = rint(data_n, step=n)
    assert isinstance(result, INT_TYPES)
    assert result == expected_n


class TestElementwise(object):
  _data = [1, 7, 9, -11, 0, .3, "ab", True, None, rint]
  @p("data", it.chain(_data, [_data], tuple(_data),
                      it.combinations_with_replacement(_data, 2))
    )
  def test_identity_with_single_and_generic_hybrid_tuple_and_list(self, data):
    f = elementwise()(lambda x: x)
    assert f(data) == data

  def test_generator_and_lazy_range_inputs(self):
    f = elementwise()(lambda x: x*2)
    fx = f(xrange(42))
    gen = (x*4 for x in xrange(42))
    fg = f(x*2 for x in xrange(42))
    assert type(fx) == type(gen)
    assert type(fg) == type(gen)
    assert list(fx) == orange(0,42*2,2)
    assert list(fg) == list(gen)


class TestConverters(object):

  def test_freq_lag_converters_are_inverses(self):
    for v in [37, 12, .5, -2, 1, .18, 4, 1e19, 2.7e-34]:
      assert freq2lag(v) == lag2freq(v)
      values = [lag2freq(freq2lag(v)), freq2lag(lag2freq(v)), v]
      for a, b in it.permutations(values, 2):
        assert almost_eq(a, b)

  def test_freq_lag_converters_with_some_values(self):
    eq = 2.506628274631
    data = {
      2.5: 2.5132741228718345,
       30: 0.20943951023931953,
        2: 3.141592653589793,
       eq: eq,
    } # This doesn't deserve to count as more than one test...
    for k, v in data.items():
      assert almost_eq(freq2lag(k), v)
      assert almost_eq(lag2freq(k), v)
      assert almost_eq(freq2lag(v), k)
      assert almost_eq(lag2freq(v), k)
      assert almost_eq(freq2lag(-k), -v)
      assert almost_eq(lag2freq(-k), -v)
      assert almost_eq(freq2lag(-v), -k)
      assert almost_eq(lag2freq(-v), -k)
