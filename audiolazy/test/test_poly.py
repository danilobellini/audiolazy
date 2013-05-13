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

from __future__ import division

import pytest
p = pytest.mark.parametrize

import operator
import types

# Audiolazy internal imports
from ..lazy_poly import Poly, lagrange, resample, x
from ..lazy_misc import almost_eq, orange, xrange, almost_eq_diff, blocks
from ..lazy_math import inf
from ..lazy_filters import z
from ..lazy_itertools import count

from . import skipper
operator.div = getattr(operator, "div", skipper("There's no operator.div"))


class TestPoly(object):
  example_data = [[1, 2, 3], [-7, 0, 3, 0, 5], [1], orange(-5, 3, -1)]

  @p("data", example_data)
  def test_len_iter_from_list(self, data):
    assert len(Poly(data)) == len([k for k in data if k != 0])
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

  polynomials = [
    12 * x ** 2 + .5 * x + 18,
    .45 * x ** 17 + 2 * x ** 5 - x + 8,
    8 * x ** 5 + .2 * x ** 3 + .1 * x ** 2,
    42.7 * x ** 4,
    8 * x ** 3 + 3 * x ** 2 + 22.2 * x + .17,
  ]

  @p("poly", instances + polynomials)
  def test_value_zero(self, poly):
    expected = ([v for k, v in poly.terms() if k == 0] + [0])[0]
    assert expected == poly(0)
    assert expected == poly(0.0)
    assert expected == poly[0]
    assert expected == poly[0.0]

  @p("poly", [Poly(v) for v in [{}, {0: 1}, {0: 0}, {0: -12.3}]])
  def test_constant_and_empty_polynomial_and_laurent(self, poly):
    assert poly.is_polynomial()
    assert poly.is_laurent()

  @p("poly", polynomials)
  def test_is_polynomial(self, poly):
    assert poly.is_polynomial()
    assert (poly + x ** 22.).is_polynomial()
    assert (poly + 8).is_polynomial()
    assert (poly * x).is_polynomial()
    assert (poly(-x) * .5).is_polynomial()
    assert (poly * .5 * x ** 2).is_polynomial()
    assert not (poly * .5 * x ** .2).is_polynomial()
    assert not (poly * x ** .5).is_polynomial()
    assert not (poly * x ** -(max(poly.terms())[0] + 1)).is_polynomial()

  @p("poly", polynomials)
  def test_is_laurent(self, poly):
    plaur = poly(x ** -1)
    assert poly.is_laurent()
    assert plaur.is_laurent()
    assert (plaur + x ** 2.).is_laurent()
    assert (plaur + 22).is_laurent()
    assert (plaur * x ** 2).is_laurent()
    assert (poly * x ** -(max(poly.terms())[0] + 1)).is_laurent()
    assert (plaur * x ** -(max(poly.terms())[0] + 1)).is_laurent()
    assert (plaur * x ** -(max(poly.terms())[0] // 2 + 1)).is_laurent()
    assert not (poly * x ** .5).is_laurent()
    assert not (poly + x ** -.2).is_laurent()

  def test_values_order_empty(self):
    poly = Poly({})
    val = poly.values()
    assert isinstance(val, types.GeneratorType)
    assert list(val) == []
    assert poly.order == 0
    assert poly.is_polynomial()
    assert poly.is_laurent()

  def test_values_order_invalid(self):
    poly = Poly({-1: 3, 1: 2})
    val = poly.values()
    with pytest.raises(AttributeError):
      next(val)
    with pytest.raises(AttributeError):
      poly.order
    assert not poly.is_polynomial()
    assert poly.is_laurent()

  @p("poly", polynomials)
  def test_values_order_valid(self, poly):
    order = max(poly.terms())[0]
    assert poly.order == order
    values = list(poly.values())
    for key, value in poly.terms():
      assert values[key] == value
      values[key] = 0
    assert values == [0] * (order + 1)

  @p("poly", polynomials)
  @p("zero", [0, 0.0, None])
  def test_values_order_valid_with_zero(self, poly, zero):
    new_poly = Poly(list(poly.values()), zero=zero)
    order = max(new_poly.terms())[0]
    assert new_poly.order == order == poly.order
    values = list(new_poly.values())
    for key, value in poly.terms():
      assert values[key] == value
      values[key] = zero
    assert values == [zero] * (order + 1)


class TestLagrange(object):

  values = [-5, 0, 14, .17]

  @p("v0", values)
  @p("v1", values)
  def test_linear_func(self, v0, v1):
    for k in [0, v0, v1]:
      pairs = [(1 + k, v0), (-1 + k, v1)]
      for interpolator in [lagrange(pairs), lagrange(reversed(pairs))]:
        assert isinstance(interpolator, types.LambdaType)
        assert almost_eq(interpolator(k), v0 * .5 + v1 * .5)
        assert almost_eq(interpolator(3 + k), v0 * 2 - v1)
        assert almost_eq(interpolator(.5 + k), v0 * .75 + v1 * .25)
        assert almost_eq(interpolator(k - .5), v0 * .25 + v1 * .75)

  @p("v0", values)
  @p("v1", values)
  def test_linear_poly(self, v0, v1):
    for k in [0, v0, v1]:
      pairs = [(1 + k, v0), (-1 + k, v1)]
      for interpolator in [lagrange.poly(pairs),
                           lagrange.poly(reversed(pairs))]:
        expected = Poly([v0 - (1 + k) * (v0 - v1) * .5, (v0 - v1) * .5])
        assert almost_eq(interpolator.values(), expected.values())

  @p("v0", values)
  @p("v1", values)
  def test_parabola_poly(self, v0, v1):
    pairs = [(0, v0), (1, v1), (v1 + .2, 0)]
    r = v1 + .2
    a = (v0 + r *(v1 - v0)) / (r * (1 - r))
    b = v1 - v0 - a
    c = v0
    expected = a * x ** 2 + b * x + c
    for interpolator in [lagrange.poly(pairs),
                         lagrange.poly(pairs[1:] + pairs[:1]),
                         lagrange.poly(reversed(pairs))]:
      assert almost_eq(interpolator.values(), expected.values())

  data = [.12, .22, -15, .7, 18, 227, .1, 4, 0, -9e3, 1, 18, 1e-4,
          44, 3, 8.00000004, 27]

  @p("poly", TestPoly.polynomials)
  def test_recover_poly_from_samples(self, poly):
    expected = list(poly.values())
    size = len(expected)
    for seq in blocks(self.data, size=size, hop=1):
      pairs = [(k, poly(k)) for k in seq]
      interpolator = lagrange.poly(pairs)
      assert almost_eq_diff(expected, interpolator.values())


class TestResample(object):

  def test_simple_downsample(self):
    data = [1, 2, 3, 4, 5]
    resampled = resample(data, old=1, new=.5, order=1)
    assert resampled.take(20) == [1, 3, 5]

  def test_simple_upsample_linear(self):
    data = [1, 2, 3, 4, 5]
    resampled = resample(data, old=1, new=2, order=1)
    expected = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    assert almost_eq(resampled.take(20), expected)

  def test_simple_upsample_linear_time_varying(self):
    acc = 1 / (1 - z ** -1)
    data = resample(xrange(50), old=1, new=1 + count() / 10, order=1)
    assert data.take() == 0.
    result = data.take(inf)
    expected = acc(1 / (1 + count() / 10))
    assert almost_eq(result, expected.limit(len(result)))
