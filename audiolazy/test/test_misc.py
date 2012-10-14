#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing module for the lazy_misc module

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

Created on Tue Jul 31 2012
danilo [dot] bellini [at] gmail [dot] com
"""

import pytest
p = pytest.mark.parametrize

import struct
import itertools as it

# Audiolazy internal imports
from ..lazy_misc import (chunks, array_chunks, elementwise, almost_eq,
                         factorial)


class TestChunks(object):

  _data = [17., -3.42, 5.4, 8.9, 27., 45.2, 1e-5, -3.7e-4, 7.2, .8272, -4.]
  _ld = len(_data)
  @p("func", [chunks, array_chunks])
  @p("size", [1, 2, 3, 4, _ld - 1, _ld, _ld + 1, 2 * _ld, 2 * _ld + 1])
  @p("given_data", [_data[:idx] for idx, unused in enumerate(_data)])
  def test_chunks(self, given_data, size, func):
    dfmt="f"
    padval=0.
    data = "".join(func(given_data, size=size, dfmt=dfmt, padval=padval))
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

  def test_generator_and_xrange_inputs(self):
    f = elementwise()(lambda x: x*2)
    fx = f(xrange(42))
    gen = (x*4 for x in xrange(42))
    fg = f(x*2 for x in xrange(42))
    assert type(fx) == type(gen)
    assert type(fg) == type(gen)
    assert list(fx) == range(0,42*2,2)
    assert list(fg) == list(gen)


class TestFactorial(object):

  @p(("n", "expected"), [(0, 1),
                         (1, 1),
                         (2, 2),
                         (3, 6),
                         (4, 24),
                         (5, 120),
                         (10, 3628800),
                         (14, 87178291200),
                         (29, 8841761993739701954543616000000),
                         (30, 265252859812191058636308480000000),
                         (6.0, 720),
                         (7.0, 5040)
                        ]
    )
  def test_valid_values(self, n, expected):
    assert factorial(n) == expected

  @p("n", [2.1, "7", 21j, "-8", -7.5, factorial])
  def test_non_integer(self, n):
    with pytest.raises(TypeError):
      factorial(n)

  @p("n", [-1, -2, -3, -4.0, -3.0, -factorial(30)])
  def test_negative(self, n):
    with pytest.raises(ValueError):
      factorial(n)

  @p(("n", "length"), [(2*factorial(7), 35980),
                       (factorial(8), 168187)]
    )
  def test_really_big_number_length(self, n, length):
    assert len(str(factorial(n))) == length
