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
Testing module for the lazy_math module
"""

import pytest
p = pytest.mark.parametrize

import itertools as it

# Audiolazy internal imports
from ..lazy_math import (factorial, dB10, dB20, inf, ln, log, log2, log10,
                         log1p, pi, e, absolute, sign)
from ..lazy_misc import almost_eq
from ..lazy_stream import Stream


class TestLog(object):

  funcs = {ln: e, log2: 2, log10: 10,
           (lambda x: log1p(x - 1)): e}

  @p("func", list(funcs))
  def test_zero(self, func):
    assert func(0) == func(0.) == func(0 + 0.j) == -inf

  @p("func", list(funcs))
  def test_one(self, func):
    assert func(1) == func(1.) == func(1. + 0j) == 0

  @p(("func", "base"), list(funcs.items()))
  def test_minus_one(self, func, base):
    for pair in it.combinations([func(-1),
                                 func(-1.),
                                 func(-1. + 0j),
                                 1j * pi * func(e),
                                ], 2):
      assert almost_eq(*pair)

  @p("base", [-1, -.5, 0., 1.])
  def test_invalid_bases(self, base):
    for val in [-10, 0, 10, base, base*base]:
      with pytest.raises(ValueError):
        log(val, base=base)


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


class TestDB10DB20(object):

  @p("func", [dB10, dB20])
  def test_zero(self, func):
    assert func(0) == -inf


class TestAbsolute(object):

  def test_absolute(self):
    assert absolute(25) == 25
    assert absolute(-2) == 2
    assert absolute(-4j) == 4.
    assert almost_eq(absolute(3 + 4j), 5)
    assert absolute([5, -12, 14j, -2j, 0]) == [5, 12, 14., 2., 0]
    assert almost_eq(absolute([1.2, -1.57e-3, -(pi ** 2), -2j,  8 - 4j]),
                     [1.2, 1.57e-3, pi ** 2, 2., 4 * 5 ** .5])


class TestSign(object):

  def test_ints(self):
    assert sign(25) == 1
    assert sign(-2) == -1
    assert sign(0) == 0
    assert sign([4, -1, 3, 7, 0, 1, -8]) == [1, -1, 1, 1, 0, 1, -1]

  def test_floats(self):
    assert sign(.1) == 1
    assert sign(-.4) == -1
    assert sign(0.) == 0
    assert sign(-0.) == 0
    assert sign([-1., 5.3e-18, 0., 2.3, -1e37, 1.]) == [-1, 1, 0, 1, -1, 1]

  def test_complex_and_mixed(self):
    with pytest.raises(TypeError):
      sign(2j)
    with pytest.raises(TypeError):
      sign([1, 1 + 1e-25j])
    data = sign(Stream(3, -1, .3, 0j))
    assert data.take(3) == [1, -1, 1]
    with pytest.raises(TypeError):
      data.peek() # 0j is complex
