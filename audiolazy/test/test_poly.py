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
# Created on Mon Oct 07 2012
# danilo [dot] bellini [at] gmail [dot] com
"""
Testing module for the lazy_poly module
"""

import pytest
p = pytest.mark.parametrize

import operator

# Audiolazy internal imports
from ..lazy_poly import Poly
from ..lazy_misc import almost_eq, orange


class TestPoly(object):
  example_data = [[1, 2, 3], [-7, 0, 3, 0, 5], [1], orange(-5, 3, -1)]

  @p("data", example_data)
  def test_len_iter_from_list(self, data):
    assert len(Poly(data)) == len([x for x in data if x != 0])
    assert len(list(Poly(data).values())) == len(data)
    assert list(Poly(data).values()) == data

  def test_empty(self):
    assert len(Poly()) == 0
    assert not Poly()
    assert len(Poly([])) == 0
    assert not Poly([])

  @p(("key", "value"), [(3, 2), (7, 12), (500, 8), (0, 12), (1, 8)])
  def test_input_dict_one_item(self, key, value):
    data = {key: value}
    assert list(Poly(dict(data)).values()) == [0] * key + [value]

  def test_input_dict_three_items_and_fake_zero(self):
    data = {8: 5, 7: -1, 6: 80, 9: 0}
    polynomial = Poly(dict(data))
    assert len(polynomial) == 3
    assert list(polynomial.values()) == [0] * 6 + [80, -1, 5]

  @p("data", example_data)
  def test_output_dict(self, data):
    assert dict(Poly(data).terms()) == {k: v for k, v in enumerate(data)
                                             if v != 0}

  def test_sum(self):
    assert Poly([2, 3, 4]) + Poly([0, 5, -4, -1]) == Poly([2, 8, 0, -1])

  def test_float_sub(self):
    poly_obj = Poly([.3, 4]) - Poly() - Poly([0, 4, -4]) + Poly([.7, 0, -4])
    assert len(poly_obj) == 1
    assert almost_eq(poly_obj[0], 1)

  instances = [
    Poly([1.7, 2, 3.3]),
    Poly({-2: 1, -1: 5.1, 3: 2}),
    Poly({-1.1: 1, 1.1: .5}),
  ]

  @p("val", instances)
  @p("div", [operator.div, operator.truediv])
  @p("den", [.1, -4e-3, 2])
  def test_int_float_div(self, val, div, den):
    assert almost_eq(div(den * val, den).terms(), val.terms())
    assert almost_eq(div(den * val, -den).terms(), (-val).terms())
    assert almost_eq(div(den * -val, den).terms(), (-val).terms())
    assert almost_eq(div(-den * val, den).terms(), (-val).terms())
    expected = Poly({k: operator.truediv(v, den) for k, v in val.terms()})
    assert almost_eq(div(val, den).terms(), expected.terms())
    assert almost_eq(div(val, -den).terms(), (-expected).terms())
    assert almost_eq(div(-val, den).terms(), (-expected).terms())
    assert almost_eq(div(-val, -den).terms(), expected.terms())
