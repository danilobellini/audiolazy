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
Testing module for the lazy_poly module
"""

from __future__ import division

import pytest
p = pytest.mark.parametrize

import operator, types, random, math
from itertools import combinations_with_replacement, combinations
from functools import reduce
from collections import OrderedDict

# Audiolazy internal imports
from ..lazy_poly import Poly, lagrange, resample, x
from ..lazy_misc import almost_eq, blocks
from ..lazy_compat import orange, xrange, iteritems, xzip, xzip_longest as xzl
from ..lazy_math import inf, e, pi
from ..lazy_filters import z
from ..lazy_itertools import count
from ..lazy_core import OpMethod
from ..lazy_stream import Stream, thub
from ..lazy_synth import white_noise

from . import skipper
operator.div = getattr(operator, "div", skipper("There's no operator.div"))


def poly_str_match(poly, string):
  return string.replace(" ", "") == str(poly).replace(" ", "")


class TestPoly(object):
  example_data = [[1, 2, 3], [-7, 0, 3, 0, 5], [1], orange(2, -6, -1)]

  instances = [
    Poly([1.7, 2, 3.3]),
    Poly({-2: 1, -1: 5.1, 3: 2}),
    Poly({-1.1: 1, 1.1: .5}),
  ]

  polynomials = [ # Simple polynomials, always from higher to lower powers
    12 * x ** 2 + .5 * x + 18,
    .45 * x ** 17 + 2 * x ** 5 - x + 8,
    8 * x ** 5 + .2 * x ** 3 + .1 * x ** 2,
    42.7 * x ** 4,
    8 * x ** 3 + 3 * x ** 2 + 22.2 * x + .17,
  ]

  diff_table = [ # Pairs (polynomial, its derivative)
    (x + 2, 1),
    (Poly({0: 22}), 0),
    (Poly({}), 0),
    (x ** 2 + 2 * x + x ** -7 + x ** -.2 - 4,
     2 * x + 2 - 7 * x ** -8 - .2 * x ** -1.2),
  ]

  to_zero_inputs = [0, 0.0, False, {}, []]

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
  @p("zero", to_zero_inputs)
  def test_values_order_valid_with_zero(self, poly, zero):
    new_poly = Poly(list(poly.values()), zero=zero)
    order = max(new_poly.terms())[0]
    assert new_poly.order == order == poly.order

    if zero != poly.zero:
      assert len(new_poly) == order + 1
      for power, coeff in enumerate(poly.values()):
        if coeff is poly.zero:
          assert new_poly[power] == coeff
          new_poly[power] = zero # Should be the same to "del new_poly[power]"

    assert len(new_poly) == len(poly)
    values = list(new_poly.values())
    for key, value in poly.terms():
      assert values[key] == value
      values[key] = zero
    assert values == [zero] * (order + 1)

  @p(("poly", "diff_poly"), diff_table)
  def test_diff(self, poly, diff_poly):
    assert poly.diff() == diff_poly

  @p(("poly", "diff_poly"), diff_table)
  def test_integrate(self, poly, diff_poly):
    if not isinstance(diff_poly, Poly):
      diff_poly = Poly(diff_poly)
    integ = diff_poly.integrate()
    poly = poly - poly[0] # Removes the constant
    assert almost_eq(integ.terms(), poly.terms())

  @p("poly", polynomials + [0])
  def test_integrate_error(self, poly):
    if not isinstance(poly, Poly):
      poly = Poly(poly)
    if poly[-1] == 0: # Ensure polynomial has the problematic term
      poly = poly + x ** -1
    with pytest.raises(ValueError):
      print(poly)
      poly.integrate()
      print(poly.integrate())

  def test_empty_comparison_to_zero(self):
    inputs = [[], {}, [0, 0], [0], {25: 0}, {0: 0}, {-.2: 0}]
    values = [0, 0.] + [Poly(k) for k in inputs]
    for a, b in combinations_with_replacement(values, 2):
      assert a == b

  @p("input_data", [[], {}, [0, 0], [0], {25: 0}, {0: 0}, {-.2: 0}])
  def test_empty_polynomial_evaluation(self, input_data):
    poly = Poly(input_data)
    assert poly(5) == poly(0) == poly(-3) == poly(.2) == poly(None) == 0
    for zero in self.to_zero_inputs:
      polyz = Poly(poly, zero=zero)
      assert polyz(5) is zero
      assert polyz(0) is zero
      assert polyz(-3) is zero
      assert polyz(.2) is zero
      assert polyz(None) is zero

  def test_not_equal(self):
    for a, b in combinations(self.polynomials, 2):
      assert a != b

  @p("op", OpMethod.get("+ - *"))
  @p("zero", to_zero_inputs)
  def test_operators_with_poly_input_keeping_zero(self, op, zero):
    if op.rev: # Testing binary reversed
      for p0, p1 in combinations_with_replacement(self.polynomials, 2):
        p0 = Poly(p0)
        p1 = Poly(p1, zero)
        result = getattr(p0, op.dname)(p1)
        assert isinstance(result, Poly)
        assert result.zero == 0
        result = getattr(p1, op.dname)(p0)
        assert isinstance(result, Poly)
        assert result.zero is zero
    elif op.arity == 2: # Testing binary
      for p0, p1 in combinations_with_replacement(self.polynomials, 2):
        p0 = Poly(p0)
        p1 = Poly(p1, zero)
        result = op.func(p0, p1)
        assert isinstance(result, Poly)
        assert result.zero == 0
        result = op.func(Poly(p1), p0) # Should keep
        assert isinstance(result, Poly)
        assert result.zero is zero
    else: # Testing unary
      for poly in self.polynomials:
        poly = Poly(poly, zero)
        result = op.func(poly)
        assert isinstance(result, Poly)
        assert result.zero is zero

  @p("op", OpMethod.get("pow truediv"))
  @p("zero", to_zero_inputs)
  def test_pow_truediv_keeping_zero(self, op, zero):
    values = [Poly(2), Poly(1, zero=[]), 3]
    values += [0, Poly()] if op.name == "pow" else [.3, -1.4]
    for value in values:
      for poly in self.polynomials:
        poly = Poly(poly, zero)
        result = op.func(poly, value)
        assert isinstance(result, Poly)
        assert result.zero is zero

  def test_pow_raise(self):
    with pytest.raises(NotImplementedError):
      (x + 2) ** (.5 + x ** -1)
    with pytest.raises(NotImplementedError):
      (x ** -1 + 2) ** (2 * x)
    with pytest.raises(TypeError):
      2 ** (2 * x)

  def test_truediv_raise(self): # In Python 2 div == truediv due to OpMethod
    with pytest.raises(NotImplementedError):
      (x + 2) / (.5 + x ** -1)
    with pytest.raises(NotImplementedError):
      (x ** -1 + 2) / (7 + 2 * x)
    with pytest.raises(TypeError):
      2 / (2 * x) # Would be "__rdiv__" in Python 2, anyway it should raise

  def test_truediv_zero_division_error(self):
    with pytest.raises(ZeroDivisionError):
      x ** 5 / (0 * x)
    with pytest.raises(ZeroDivisionError):
      (2 + x ** 1.1) / (0 * x)
    with pytest.raises(ZeroDivisionError):
      (x ** -31 + 7) / 0.

  @p("zero", to_zero_inputs)
  @p("method", ["diff", "integrate"])
  def test_eq_ne_diff_integrate_keep_zero(self, zero, method):
    if method == "diff":
      expected = Poly([3, 4])
    else:
      expected = Poly([0., 4, 1.5, 2./3])
    result = getattr(Poly([4, 3, 2], zero=zero), method)()
    if zero == 0:
      assert result == expected
    else:
      assert result != expected
    assert almost_eq(result.terms(), expected.terms())
    assert result == Poly(expected, zero=zero)
    assert result.zero is zero

  @p("op", OpMethod.get("pow truediv"))
  @p("zero", to_zero_inputs)
  def test_pow_truediv_from_empty_poly_instance(self, op, zero):
    empty = Poly(zero=zero)
    result = op.func(empty, 2)
    assert result == empty
    assert result != Poly(zero=op)
    assert len(result) == 0
    assert result.zero is zero

  @p("data", example_data)
  @p("poly", polynomials + [x + 5 * x ** 2 + 2])
  def test_stream_evaluation(self, data, poly):
    result = poly(Stream(data))
    assert isinstance(result, Stream)
    result = list(result)
    assert len(result) == len(data)
    assert result == [poly(k) for k in data]

  def test_stream_coeffs_with_integer(self): # Poly before to avoid casting
    poly = x * Stream(0, 2, 3) + x ** 2 * Stream(4, 1) + 8 # Order matters!
    assert str(poly) == "8 + a1 * x + a2 * x^2"
    result = poly(5)
    assert isinstance(result, Stream)
    expected = 5 * Stream(0, 2, 3) + 25 * Stream(4, 1) + 8
    assert all(expected.limit(50) == result.limit(50))

  def test_stream_coeffs_purely_stream(self): # Poly before to avoid casting
    poly = x * Stream(0, 2, 3) + x ** 2 * Stream(4, 1) + Stream(2, 2, 2, 2, 5)
    assert str(poly) == "a0 + a1 * x + a2 * x^2"
    result = poly(count())
    assert isinstance(result, Stream)
    expected = (count() * Stream(0, 2, 3) + count() ** 2 * Stream(4, 1) +
                Stream(2, 2, 2, 2, 5))
    assert all(expected.limit(50) == result.limit(50))

  def test_stream_coeffs_mul(self):
    poly1 = x * Stream(0, 1, 2) + 5 # Order matters
    poly2 = x ** 2 * count() - 2
    poly = poly1 * poly2
    result = poly(Stream(4, 3, 7, 5, 8))
    assert isinstance(result, Stream)
    expected = (Stream(4, 3, 7, 5, 8) ** 3 * Stream(0, 1, 2) * count()
                + Stream(4, 3, 7, 5, 8) ** 2 * 5 * count()
                - Stream(4, 3, 7, 5, 8) * Stream(0, 1, 2) * 2
                - 10
               )
    assert all(expected.limit(50) == result.limit(50))

  @p("zero", to_zero_inputs)
  def test_stream_coeffs_add_copy_with_zero(self, zero):
    poly = x * Stream(0, 4, 2, 3, 1)
    new_poly = 3 * poly.copy(zero=zero)
    new_poly += poly
    assert isinstance(new_poly, Poly)
    assert new_poly.zero is zero
    result = new_poly(Stream(1, 2, 3, 0, 4, 7, -2))
    assert isinstance(result, Stream)
    expected = 4 * Stream(1, 2, 3, 0, 4, 7, -2) * Stream(0, 4, 2, 3, 1)
    assert all(expected.limit(50) == result.limit(50))

  @p("zero", to_zero_inputs)
  @p("horner", [True, False])
  def test_stream_coeffs_mul_copy_with_zero(self, zero, horner):
    poly = x ** 2 * Stream(3, 2, 1) + 2 * count() + x ** -3
    new_poly = poly.copy(zero=zero)
    new_poly *= poly + 1
    assert isinstance(new_poly, Poly)
    assert new_poly.zero is zero
    result = new_poly(count(18) ** .5, horner=horner)
    assert isinstance(result, Stream)
    expected = (count(18) * Stream(3, 2, 1)
                + 2 * count() + count(18) ** -1.5
               ) * (count(18) * Stream(3, 2, 1)
                + 2 * count() + count(18) ** -1.5 + 1
               )
    assert almost_eq(expected.limit(50), result.limit(50))

  def test_eq_ne_of_a_stream_copy(self):
    poly = x * Stream(0, 1)
    new_poly = poly.copy()
    assert poly == poly
    assert poly != new_poly
    assert new_poly == new_poly
    other_poly = poly.copy(zero=[])
    assert new_poly != other_poly
    assert new_poly.zero != other_poly.zero

  @p("poly", polynomials)
  def test_pow_basics(self, poly):
    assert poly ** 0 == 1
    assert poly ** Poly() == 1
    assert poly ** 1 == poly
    assert poly ** 2 == poly * poly
    assert poly ** Poly(2) == poly * poly
    assert almost_eq((poly ** 3).terms(), (poly * poly * poly).terms())

  def test_power_one_keep_integer(self):
    for value in [0, -1, .5, 18]:
      poly = Poly(1) ** value
      assert poly.order == 0
      assert poly[0] == 1
      assert isinstance(poly[0], int)

  @p("zero", to_zero_inputs)
  @p("horner", [True, False])
  def test_pow_with_stream_coeff(self, zero, horner):
    poly = Poly(x ** -2 * Stream(1, 0) + 2, zero=zero)
    new_poly = poly ** 2
    assert isinstance(new_poly, Poly)
    assert new_poly.zero is zero
    result = new_poly(count(1), horner=horner)
    assert isinstance(result, Stream)
    expected = (count(1) ** -4 * Stream(1, 0)
                + 4 * count(1) ** -2 * Stream(1, 0)
                + 4
               )
    assert almost_eq(expected.limit(50), result.limit(50))

  @p("zero", to_zero_inputs)
  @p("horner", [True, False])
  def test_truediv_by_stream(self, zero, horner):
    poly = Poly(x ** .5 * Stream(.2, .4) + 7, zero=zero)
    new_poly = poly / count(2, 4)
    assert isinstance(new_poly, Poly)
    assert new_poly.zero is zero
    result = new_poly(Stream(17, .2, .1, 0, .2, .5, 99), horner=horner)
    assert isinstance(result, Stream)
    expected = (Stream(17, .2, .1, 0, .2, .5, 99) ** .5 * Stream(.2, .4) + 7
               ) / count(2, 4)
    assert almost_eq(expected.limit(50), result.limit(50))

  def test_setitem(self):
    poly = x + 2
    poly[3] = 5
    assert poly == 5 * x ** 3 + x + 2
    poly[.2] = 1
    assert poly == 5 * x ** 3 + x + 2 + x ** .2
    var_coeff = Stream(1, 2, 3)
    poly[0] = var_coeff
    term_iter = poly.terms(sort=True)
    power, item = next(term_iter)
    assert item is var_coeff
    assert power == 0
    assert Poly(dict(term_iter)) == 5 * x ** 3 + x + x ** .2

  def test_hash_makes_poly_immutable(self):
    poly = x + 1
    poly[3] = 1
    poly.zero = 0
    my_set = {poly, 27, x}
    with pytest.raises(TypeError):
      poly[3] = 0
    with pytest.raises(TypeError):
      poly.zero = 0.
    assert poly == x ** 3 + x + 1
    assert poly in my_set

  def test_hash_equalness_on_different_sorting(self):
    assert hash(x + 1) == hash(1 + x)
    assert hash(x ** 2 + x + 1) == hash(1 + (1 + x) * x)
    assert hash(x + x ** .34 + 1) == hash((1 - x + x ** .34) + 2 * x)
    assert hash(x ** (2j + 1) + x ** 3) == hash(x ** 3 + x ** (2j + 1))

  @p("poly", polynomials + [5 - x ** -2, x + 2 * x ** .3, x ** 4.3j + x - 1])
  def test_hash_non_equal_for_different_zeros(self, poly):
    class Zero(): pass
    poly_other_zero = Poly(poly, zero=Zero)
    assert hash(poly) == hash(x + poly - x)
    assert hash(poly) != hash(poly_other_zero)

  @p("poly", [x ** 2 - 2 * x + 1, .3 * x ** 7 - 4 * x ** 2 + .1])
  def test_roots(self, poly):
    prod = lambda iterable: reduce(operator.mul, iterable, Poly(1))
    rebuilt_poly = poly[poly.order] * prod(x - r for r in poly.roots)
    assert almost_eq.diff(poly.values(), rebuilt_poly.values())

  @p("poly", [5 - x ** -2, x + 2 * x ** .3])
  def test_roots_invalid(self, poly):
    with pytest.raises(AttributeError):
      poly.roots

  def test_constants_have_no_roots(self):
    assert all(Poly(c).roots == [] for c in [2, -3, 4j, .2 + 3.4j])

  @p("list_data", [[1, 2, 3], [-7, 0, 3, 0, 5], orange(-5, 3)])
  def test_list_constructor_unsorted_terms_order(self, list_data):
    random.shuffle(list_data)
    poly = Poly(list_data)
    dict_data = ((idx, el) for idx, el in enumerate(list_data) if el)
    for (idx, el), (power, coeff) in xzl(dict_data, poly.terms(sort=False)):
      assert idx == power
      assert el == coeff

  @p("dict_data", [
    [(3, 4), (7, 3), (5, 2.5)],
    [(4, 3), (19, .2), (-4, 1e-18), (7, .25)],
    [(3, 2j), (3j, 4), (.5, 0.23)],
  ])
  def test_dict_constructor_unsorted_terms_order(self, dict_data):
    random.shuffle(dict_data)
    poly = Poly(OrderedDict(dict_data))
    for (idx, el), (power, coeff) in xzl(dict_data, poly.terms(sort=False)):
      assert idx == power
      assert el == coeff

  @p("poly", instances + polynomials)
  def test_copy_keep_unsorted_terms_order(self, poly):
    poly_copy = poly.copy()
    assert hash(poly) == hash(poly_copy)
    for a, b in xzl(poly.terms(sort=False), poly_copy.terms(sort=False)):
      assert a == b

  def test_diff_keep_unsorted_terms_order(self):
    data = [3, 2, 5, -1, 8, 0, 4]
    diff1 = [2, 10, -3, 32, 0, 24]
    diff2 = [10, -6, 96, 0, 120]

    order = orange(len(data))
    random.shuffle(order)

    dict_data = OrderedDict((k, data[k]) for k in order)
    dict_diff1 = OrderedDict((k - 1, diff1[k - 1]) for k in order if k >= 1)
    dict_diff2 = OrderedDict((k - 2, diff2[k - 2]) for k in order if k >= 2)

    poly = Poly(dict_data)
    pdiff1 = poly.diff()
    pdiff2 = poly.diff(2)

    expected1 = Poly(dict_diff1)
    for a, b in xzl(pdiff1.terms(sort=False), expected1.terms(sort=False)):
      assert a == b

    expected2 = Poly(dict_diff2)
    for a, b in xzl(pdiff2.terms(sort=False), expected2.terms(sort=False)):
      assert a == b

  def test_integrate_keep_unsorted_terms_order(self):
    data = [18, 2, 12, 0, 60, -54]
    integ = [18, 1, 4, 0, 12, -9]

    order = orange(len(data))
    random.shuffle(order)

    dict_data = OrderedDict((k, data[k]) for k in order)
    dict_integ = OrderedDict((k + 1, integ[k]) for k in order)

    pinteg = Poly(dict_data).integrate()
    expected = Poly(dict_integ)
    assert almost_eq(pinteg.terms(sort=False), expected.terms(sort=False))

  def test_add_sub_keep_unsorted_terms_order(self):
    poly1 = Poly(OrderedDict([(1, 4), (3, 8)]))
    poly2 = Poly(OrderedDict([(2, 5), (1, -2)]))
    padd = poly1 + poly2
    psub = poly1 - poly2
    expected_add = Poly(OrderedDict([(1, 2), (3, 8), (2, 5)]))
    expected_sub = Poly(OrderedDict([(1, 6), (3, 8), (2, -5)]))

    for a, b in xzl(padd.terms(sort=False), expected_add.terms(sort=False)):
      assert a == b
    for a, b in xzl(psub.terms(sort=False), expected_sub.terms(sort=False)):
      assert a == b

  def test_add_mul_pow_keep_unsorted_terms_order(self):
    poly1 = x ** 2 + 6 * x + 9
    poly2 = x * (x + 6) + 9 # Horner scheme
    poly3 = (x + 3) ** 2

    revp1 = 9 + 6 * x + x ** 2
    revp2 = 9 + (6 + x) * x
    revp3 = (3 + x) ** 2

    for a, b, c, d in xzl(poly1.terms(sort=False), poly2.terms(sort=False),
                          poly3.terms(sort=False), [(2, 1), (1, 6), (0, 9)]):
      assert a == b == c == d
    for a, b, c, d in xzl(revp1.terms(sort=False), revp2.terms(sort=False),
                          revp3.terms(sort=False), [(0, 9), (1, 6), (2, 1)]):
      assert a == b == c == d

  @p("div", [operator.div, operator.truediv])
  def test_div_keep_unsorted_terms_order(self, div):
    data = [18, 2, 12, 0, 60, -54]
    denominator = 12.5
    divided = [el / denominator for el in data]

    order = orange(len(data))
    random.shuffle(order)

    dict_data = OrderedDict((k, data[k]) for k in order)
    dict_div = OrderedDict((k, divided[k]) for k in order)

    pdiv = Poly(dict_data) / denominator
    expected = Poly(dict_div)
    assert almost_eq(pdiv.terms(sort=False), expected.terms(sort=False))

  @p("op", OpMethod.get("1", without="~"))
  def test_unary_operators_keep_unsorted_terms_order(self, op):
    data = thub(white_noise(low=-20, high=20), 2)
    powers = thub(white_noise(low=-20, high=20).limit(20), 2)

    dict_data = OrderedDict(xzip(powers, data))
    dict_op = OrderedDict(xzip(powers, op.func(data)))

    poly = op.func(Poly(dict_data))
    expected = Poly(dict_op)
    assert almost_eq(poly.terms(sort=False), expected.terms(sort=False))

  @p("poly", [poly for poly in polynomials if len(poly) >= 2])
  def test_terms_sort_simple_polynomials(self, poly):
    values = [(idx, v) for idx, v in enumerate(poly.values()) if v != 0]
    terms_auto = list(poly.terms())
    terms_sorted = list(poly.terms(sort=True))
    terms_not_sorted = list(poly.terms(sort=False))

    assert terms_not_sorted != terms_sorted
    assert terms_auto == values == terms_sorted == terms_not_sorted[::-1]

  def test_terms_sort_reverse_laurent_polynomials(self):
    data = white_noise(low=-1000, high=500)
    powers = white_noise(low=-230, high=20).map(int).limit(50)
    dict_original = OrderedDict(xzip(powers, data))
    poly = Poly(dict_original)

    terms_auto = list(poly.terms())
    terms_sorted = list(poly.terms(sort=True))
    terms_not_sorted = list(poly.terms(sort=False))
    terms_auto_rev = list(poly.terms(reverse=True))
    terms_sorted_rev = list(poly.terms(sort=True, reverse=True))
    terms_not_sorted_rev = list(poly.terms(sort=False, reverse=True))

    assert terms_not_sorted != terms_sorted
    assert (terms_auto == # Default is sorted
            terms_sorted == terms_auto_rev[::-1] == terms_sorted_rev[::-1])
    assert terms_not_sorted == terms_not_sorted_rev[::-1]
    assert terms_sorted == sorted(terms_not_sorted)
    assert terms_not_sorted == list(iteritems(dict_original))

  def test_terms_sort_reverse_float_powers(self):
    data = white_noise(low=-10, high=30)
    powers = white_noise(low=-30, high=40).limit(40)
    dict_original = OrderedDict(xzip(powers, data))
    poly = Poly(dict_original)

    terms_auto = list(poly.terms())
    terms_sorted = list(poly.terms(sort=True))
    terms_not_sorted = list(poly.terms(sort=False))
    terms_auto_rev = list(poly.terms(reverse=True))
    terms_sorted_rev = list(poly.terms(sort=True, reverse=True))
    terms_not_sorted_rev = list(poly.terms(sort=False, reverse=True))

    assert terms_not_sorted != terms_sorted
    assert (terms_auto == terms_not_sorted == # Default is not sorted
            terms_auto_rev[::-1] == terms_not_sorted_rev[::-1])
    assert terms_sorted == terms_sorted_rev[::-1]
    assert terms_sorted == sorted(terms_not_sorted)
    assert terms_not_sorted == list(iteritems(dict_original))

  def test_complex_power(self):
    poly = x ** 2j + 2 * x + 1
    assert set(poly.terms()) == {(0, 1), (1, 2), (2j, 1)}
    assert poly(0) == 1
    assert poly(1) == 4
    k = e ** pi
    assert almost_eq.diff(poly(k), 2 * (1 + k))
    ln4 = math.log(4)
    assert almost_eq.diff(poly(2), math.cos(ln4) + 5 + math.sin(ln4) * 1j)

  def test_call_horner_on_complex_invalid(self):
    poly = x ** 2j + 2 * x + 1
    assert poly(0, horner=True) == 1 # For zero the scheme isn't evaluated
    with pytest.raises(ValueError):
      poly(1, horner=True) # Here it need to be evaluated

  def test_set_to_and_the_zero_with_setitem_internal_data_and_property(self):
    poly = x ** 2 - 1
    poly[3.] = 0 # Using __setitem__
    poly[-1] = 8.
    assert len(poly) == 3
    assert poly_str_match(poly, # Laurent polynomial (sorted)
                          "8 * x^-1 - 1 + x^2")

    poly._data[.18] = 0 # Without using __setitem__
    assert len(poly) == 4 # Due to an undesired extra [hacked] zero
    assert poly_str_match(poly, # Sum of powers w/ a float power (not sorted)
                          "x^2 - 1 + 8 * x^-1 + 0 * x^0.18")

    assert list(poly.terms()) == [(2, 1), (0, -1), (-1, 8), (.18, 0)]
    poly.zero = 0 # This should clean that undesired zero
    assert len(poly) == 3
    assert list(poly.terms()) == [(-1, 8), (0, -1), (2, 1)] # Laurent (sorted)

  def test_string_representation(self):
    assert str(x) == "x"
    assert all(str(Poly(k)) == str(k) for k in [2, -3, 4j, 0])
    print(2 * x ** 5 + 1.8 * x - 7)
    assert poly_str_match(2 * x ** 5 + 1.8 * x - 7,
                          "-7 + 1.8 * x + 2 * x^5") # Polynomial (sorted)
    assert poly_str_match(8. * x ** -5 + 0 * x - 1. * x ** .2 + x ** .5j,
                          "8 * x^-5 - x^0.2 + x^0.5j") # Complex power


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
      assert almost_eq.diff(expected, interpolator.values())


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
