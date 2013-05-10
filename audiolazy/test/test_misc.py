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

import struct
import itertools as it

# Audiolazy internal imports
from ..lazy_misc import (INT_TYPES, rint, chunks, array_chunks, elementwise,
                         almost_eq, rst_table, orange, xrange)


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


class TestChunks(object):

  _data = [17., -3.42, 5.4, 8.9, 27., 45.2, 1e-5, -3.7e-4, 7.2, .8272, -4.]
  _ld = len(_data)
  @p("func", [chunks, array_chunks])
  @p("size", [1, 2, 3, 4, _ld - 1, _ld, _ld + 1, 2 * _ld, 2 * _ld + 1])
  @p("given_data", (lambda d:
                      [d[:idx] for idx, unused in enumerate(d)]
                   )(d=_data)
  )
  def test_chunks(self, given_data, size, func):
    dfmt="f"
    padval=0.
    data = b"".join(func(given_data, size=size, dfmt=dfmt, padval=padval))
    samples_in = len(given_data)
    samples_out = samples_in
    if samples_in % size != 0:
      samples_out -= samples_in % -size
      assert samples_out > samples_in # Testing the tester...
    restored_data = struct.Struct(dfmt * samples_out).unpack(data)
    assert almost_eq(given_data,
                     restored_data[:samples_in],
                     ignore_type=True)
    assert almost_eq([padval]*(samples_out - samples_in),
                     restored_data[samples_in:],
                     ignore_type=True)


class TestElementwise(object):
  _data = [1, 7, 9, -11, 0, .3, "ab", True, None, chunks]
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


class TestRSTTable(object):

  simple_input = [
    [1, 2, 3, "hybrid"],
    [3, "mixed", .5, 123123]
  ]

  def test_simple_input_table(self):
    assert rst_table(
             self.simple_input,
             "this is_ a test".split()
           ) == [
             "==== ===== === ======",
             "this  is_   a   test ",
             "==== ===== === ======",
             "1    2     3   hybrid",
             "3    mixed 0.5 123123",
             "==== ===== === ======",
           ]
