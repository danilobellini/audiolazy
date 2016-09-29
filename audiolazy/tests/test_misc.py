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
Testing module for the lazy_misc module
"""

import pytest
p = pytest.mark.parametrize

import itertools as it
import cmath, operator
from functools import reduce

# Audiolazy internal imports
from ..lazy_misc import (rint, elementwise, freq2lag, lag2freq, almost_eq,
                         cached)
from ..lazy_compat import INT_TYPES, orange, xrange
from ..lazy_math import pi
from ..lazy_stream import Stream
from ..lazy_synth import line

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


@p("aeq", almost_eq)
class TestAlmostEq(object):

  def test_single_values(self, aeq):
    assert aeq(pi, pi)
    assert aeq(0, 0)
    assert aeq(0, 0.)
    assert aeq(0., 0.)
    assert aeq(18.7, 18.7)
    assert aeq(15, 15)
    assert not aeq(15.0001, 15)
    assert not aeq(1e-3, 1e-7)
    assert not aeq(99999, 99999.03)
    assert not aeq(.99999, .9999903)
    assert aeq(.99999, .9999901)

  def test_complex_values(self, aeq):
    assert not aeq(2j, 2)
    assert not aeq(2j + 1, 2 + 1j)
    assert not aeq(3 + 4j, 5)
    assert not aeq(3 + 4j, 3 + 4.0001j)
    assert not aeq(3 + 4j, 2.99999 + 4j)
    assert aeq(3 + 4j, 1j + 3 * (1 + 1j))
    assert aeq(2j + 1, 2j + 1 + 1e-9 - 3e-8j)
    for a, b in line(28, 0, 2j * pi, finish=True).blocks(size=2, hop=1):
      assert not aeq(a, b)
      assert aeq(a, a * cmath.exp(2e-9j * pi))
      assert aeq(b, b * cmath.exp(-3e-9j * pi))

  def test_iterable_items(self, aeq):
    items = [1, 3, 2e-4, .5, .1, pi, 0, 0, 0., 12]
    assert aeq(items, Stream(items))
    assert aeq((d for d in sorted(items)), sorted(items))
    items_changed = items[:]
    items_changed[2] = 3j
    assert not aeq(items_changed, Stream(items))
    assert aeq([i * (1 + 2e-9) for i in items], Stream(items))

  def test_empty_iterables_and_nested_ones(self, aeq):
    assert aeq([], tuple())
    assert aeq(set(), tuple())
    assert aeq([], Stream([]))
    assert aeq(([], []), [[], []])
    assert not aeq(([], [], []), [[], []])
    assert not aeq([[]], [])
    assert aeq(([], [], []), [[], []], pad=[])
    assert not aeq([], tuple(), ignore_type=False)
    assert not aeq(set(), tuple(), ignore_type=False)
    assert not aeq([], Stream([]), ignore_type=False)
    assert not aeq(([], []), [[], []], ignore_type=False)

  def test_nested_iterables(self, aeq):
    k = 1 + 1e-8
    items_list = [1, [3 + 1e-7, [2e-4 - 9e-14, .5]], [.1, pi * k], 0, [12]]
    items_tuple = [1 - 7e-8, (3, (2e-4, .5)), (.1, pi / k), 0., (12,)]
    assert almost_eq(items_list, items_tuple)
    items_list[-1][-1] = 11.9999
    assert not almost_eq(items_list, items_tuple)


class TestCached(object):

  def test_single_parameter(self):
    call_values = []

    @cached
    def square_it(n):
      call_values.append(n)
      return n ** 2

    for el in [5, 7, 1, 5, .2, 1, 0, -5, .2, 1, 1, 3, 2, 1, 0., 441]:
      assert square_it(el) == el * el

    # Zero appears once, since hash(0) == hash(0.) == 0
    assert call_values == [5, 7, 1, .2, 0, -5, 3, 2, 441]


  def test_two_parameters(self):
    call_values = []

    @cached
    def add(a, b):
      call_values.append((a, b))
      return a + b

    for x, y  in [(4, 5), ("abs", "de"), (.3, 2), ("abs", "de"), (pi, 1),
                  (4, 5), (pi, 1), (1, pi), (5, 4), ("", ""), (0., 0),
                  ("", ""), (5, 4), (0, 0.), (.3, 2.), (0., 0j)]:
      assert add(x, y) == x + y

    # Twos are same since hash(2) == hash(2.) == 2 ; The same with zeros
    # (float, int and complex), as hash(0) == hash(0.) == hash(0j) == 0
    assert call_values == [(4, 5), ("abs", "de"), (.3, 2), (pi, 1), (1, pi),
                           (5, 4), ("", ""), (0., 0)]


  def test_variable_amount_of_parameters_recursive(self):
    call_values = []

    @cached
    def prod(*args):
      call_values.append(args)
      if len(args) == 0:
        return 1
      return args[0] * prod(*args[1:])

    for blk in [[3, 5, 2], [7, .7, 7, 5], [7, 5], [8, 2, 7, 5], [.8, 5, 2]]:
      assert prod(*blk) == reduce(operator.mul, blk)

    assert call_values == [(3, 5, 2), (5, 2), (2,), tuple(),
                           (7, .7, 7, 5), (.7, 7, 5), (7, 5), (5,),
                           (8, 2, 7, 5), (2, 7, 5),
                           (.8, 5, 2)]


  def test_parameters_with_defaults(self):
    call_values = []

    @cached # The "name" and "params" can't be used as keyword args
    def lamb(eq, params=None, name=None):
      """ Just an unsafe "lambda" function "creator" """
      call_values.append([eq, params, name])
      func = eval("lambda {}: {}".format(", ".join(params or []), eq))
      if name is not None:
        func.__name__ = name
      return func

    assert lamb("2")() == 2
    assert lamb("n + m", ("n", "m"))(1, 3) == 4
    assert lamb("n + m", ("n", "m"), "named_one")("a", "b") == "ab"
    assert lamb("n + m", ("n", "m"))("c", "de") == "cde"
    assert lamb("2")() == 2
    assert lamb("n + m", ("n", "m"), None)([1], ["a"]) == [1, "a"]
    assert lamb("n + m", ("n", "m"), "named_one")(45, 0) == 45
    assert lamb("n + m", ("n", "m"))([], []) == []
    assert lamb("n + m", ("n", "m"), None)([5, 5], [5]) == [5, 5, 5]

    assert lamb("n + m", ("n", "m")).__name__ == "<lambda>"
    assert lamb("n + m", ("n", "m"), "named_one").__name__ == "named_one"
    assert lamb("n + m", ("n", "m"), None).__name__ == "<lambda>"
    assert lamb("2").__name__ == "<lambda>"

    assert call_values == [["2", None, None],
                           ["n + m", ("n", "m"), None], # Using default None
                           ["n + m", ("n", "m"), "named_one"],
                           ["n + m", ("n", "m"), None]] # Passing 3rd arg


  def test_direct_cache_access(self):
    @cached
    def yet_another_sum(*args):
      return args[0] + yet_another_sum(*args[1:]) if args else 0

    assert yet_another_sum(2, 3, 4) == 9
    assert isinstance(yet_another_sum(2, 3, 4), int)

    assert isinstance(yet_another_sum.cache, dict)
    assert len(yet_another_sum.cache) == 4 # (2, 3, 4), (3, 4), (4), tuple()

    # Let's make it work with tuples
    yet_another_sum.cache[tuple()] = tuple()
    assert yet_another_sum((2, 3, 4), (5, "why not?")
                          ) == (2, 3, 4, 5, "why not?")
    with pytest.raises(TypeError):
      assert yet_another_sum([2, 3, 5]) # Can't add 5 + tuple()
    assert yet_another_sum(2, 3, 4) == 9 # But this one was already in cache!

    # Let's make it work with int again
    yet_another_sum.cache[tuple()] = 0.

    assert yet_another_sum(2, 3, 4) == 9 # Cached
    assert isinstance(yet_another_sum(2, 3, 4), int)

    assert yet_another_sum(2, 3, 5) == 10.
    assert isinstance(yet_another_sum(2, 3, 5), float)

    assert yet_another_sum(4, 3, 4) == 11 # (3, 4) was cached
    assert isinstance(yet_another_sum(4, 3, 4), int)

    del yet_another_sum.cache[(3, 4)]
    del yet_another_sum.cache[(4,)]
    assert yet_another_sum(5, 3, 4) == 12.
    assert isinstance(yet_another_sum(5, 3, 4), float)

    yet_another_sum.cache[(3, 4)] = -27 # Using a weird cached value
    assert yet_another_sum(4, 3, 4) == 11 # (4, 3, 4) was cached
    assert yet_another_sum(7, 3, 4) == -20
    assert isinstance(yet_another_sum(7, 3, 4), int)

    yet_another_sum.cache[(tuple(), frozenset([1, 2]), 23)] = -1
    assert yet_another_sum(5, tuple(), frozenset([1, 2]), 23) == 4