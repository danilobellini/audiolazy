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
Testing module for the lazy_poly module by using Sympy
"""

from __future__ import division

import pytest
p = pytest.mark.parametrize

import sympy

# Audiolazy internal imports
from ..lazy_poly import Poly, x
from ..lazy_compat import builtins, PYTHON2


class TestPolySympy(object):

  def test_call_horner_simple_polinomial(self):
    poly = x ** 2 + 2 * x + 18 - x ** 7
    a = sympy.Symbol("a")
    expected_horner = 18 + (2 + (1 + -a ** 5) * a) * a
    expected_direct = 18 + 2 * a + a ** 2 - a ** 7

    # "Testing the test"
    assert expected_horner.expand() == expected_direct
    assert expected_horner != expected_direct

    # Applying the value
    assert poly(a, horner=True) == expected_horner == poly(a)
    assert poly(a, horner=False) == expected_direct

  def test_call_horner_laurent_polinomial(self):
    poly = x ** 2 + 2 * x ** -3 + 9 - 5 * x ** 7
    a = sympy.Symbol("a")
    expected_horner = (2 + (9 + (1 - 5 * a ** 5) * a ** 2) * a ** 3) * a ** -3
    expected_direct = 2 * a ** -3 + 9 + a ** 2 - 5 * a ** 7

    # "Testing the test"
    assert expected_horner.expand() == expected_direct
    assert expected_horner != expected_direct

    # Applying the value
    assert poly(a, horner=True) == expected_horner
    assert poly(a, horner=False) == expected_direct == poly(a)

  def test_call_horner_sum_of_symbolic_powers(self):

    def sorted_mock(iterable, reverse=False):
      """ Used internally by terms to sort the powers """
      data = list(iterable)
      if data and isinstance(data[0], sympy.Basic):
        return builtins.sorted(data, reverse=reverse, key=str)
      return builtins.sorted(data, reverse=reverse)

    # Mocks the sorted, but just internally to the Poly.terms
    if PYTHON2:
      terms_globals = Poly.terms.im_func.func_globals
    else:
      terms_globals = Poly.terms.__globals__
    terms_globals["sorted"] = sorted_mock

    try:
      a, b, c, d, k = sympy.symbols("a b c d k")
      poly = d * x ** c - d * x ** a + x ** b
      expected_horner = (-d + (1 + d * k ** (c - b)) * k ** (b - a)) * k ** a
      expected_direct = d * k ** c - d * k ** a + k ** b

      # "Testing the test"
      assert expected_horner.expand() == expected_direct
      assert expected_horner != expected_direct

      # Applying the value
      assert poly(k, horner=True) == expected_horner
      assert poly(k, horner=False) == expected_direct == poly(k)

    finally:
      del terms_globals["sorted"] # Clean the mock
