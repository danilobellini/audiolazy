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
Testing module for the lazy_itertools module
"""

import pytest
p = pytest.mark.parametrize

import operator
from functools import reduce

# Audiolazy internal imports
from ..lazy_itertools import accumulate, chain, izip, count
from ..lazy_stream import Stream
from ..lazy_math import inf
from ..lazy_poly import x


@p("acc", accumulate)
class TestAccumulate(object):

  @p("empty", [[], tuple(), set(), Stream([])])
  def test_empty_input(self, acc, empty):
    data = acc(empty)
    assert isinstance(data, Stream)
    assert list(data) == []

  def test_one_input(self, acc):
    for k in [1, -5, 1e3, inf, x]:
      data = acc([k])
      assert isinstance(data, Stream)
      assert list(data) == [k]

  def test_few_numbers(self, acc):
    data = acc(Stream([4, 7, 5, 3, -2, -3, -1, 12, 8, .5, -13]))
    assert isinstance(data, Stream)
    assert list(data) == [4, 11, 16, 19, 17, 14, 13, 25, 33, 33.5, 20.5]


class TestCount(object):

  def test_no_input(self):
    data = count()
    assert isinstance(data, Stream)
    assert data.take(14) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    assert data.take(3) == [14, 15, 16]

  @p("start", [0, -1, 7])
  def test_starting_value(self, start):
    data1 = count(start)
    data2 = count(start=start)
    assert isinstance(data1, Stream)
    assert isinstance(data2, Stream)
    expected_zero = Stream([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    expected = (expected_zero + start).take(20)
    after = [14 + start, 15 + start, 16 + start]
    assert data1.take(14) == expected
    assert data2.take(13) == expected[:-1]
    assert data1.take(3) == after
    assert data2.take(4) == expected[-1:] + after

  @p("start", [0, -5, 1])
  @p("step", [1, -1, 3])
  def test_two_inputs(self, start, step):
    data1 = count(start, step)
    data2 = count(start=start, step=step)
    assert isinstance(data1, Stream)
    assert isinstance(data2, Stream)
    expected_zero = Stream([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    expected = (expected_zero * step + start).take(20)
    after = list(Stream([14, 15, 16]) * step + start)
    assert data1.take(14) == expected
    assert data2.take(13) == expected[:-1]
    assert data1.take(3) == after
    assert data2.take(4) == expected[-1:] + after


class TestChain(object):

  data = [1, 5, 3, 17, -2, 8, chain, izip, pytest, lambda x: x, 8.2]
  some_lists = [data, data[:5], data[3:], data[::-1], data[::2], data[1::3]]

  @p("blk", some_lists)
  def test_with_one_list_three_times(self, blk):
    expected = blk + blk + blk
    result = chain(blk, blk, blk)
    assert isinstance(result, Stream)
    assert list(result) == expected
    result = chain.from_iterable(3 * [blk])
    assert isinstance(result, Stream)
    assert result.take(inf) == expected

  def test_with_lists(self):
    blk = self.some_lists
    result = chain(*blk)
    assert isinstance(result, Stream)
    expected = list(reduce(operator.concat, blk))
    assert list(result) == expected
    result = chain.from_iterable(blk)
    assert isinstance(result, Stream)
    assert list(result) == expected
    result = chain.star(blk)
    assert isinstance(result, Stream)
    assert list(result) == expected

  def test_with_endless_stream(self):
    expected = [1, 2, -3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    result = chain([1, 2, -3], count())
    assert isinstance(result, Stream)
    assert result.take(len(expected)) == expected
    result = chain.from_iterable(([1, 2, -3], count()))
    assert isinstance(result, Stream)
    assert result.take(len(expected)) == expected

  def test_star_with_generator_input(self):
    def gen():
      yield [5, 5, 5]
      yield [2, 2]
      yield count(-4, 2)
    expected = [5, 5, 5, 2, 2, -4, -2, 0, 2, 4, 6, 8, 10, 12]
    result = chain.star(gen())
    assert isinstance(result, Stream)
    assert result.take(len(expected)) == expected
    assert chain.star is chain.from_iterable

  @pytest.mark.timeout(2)
  def test_star_with_endless_generator_input(self):
    def gen(): # Yields [], [1], [2, 2], [3, 3, 3], ...
      for c in count():
        yield [c] * c
    expected = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6]
    result = chain.star(gen())
    assert isinstance(result, Stream)
    assert result.take(len(expected)) == expected


class TestIZip(object):

  def test_smallest(self):
    for func in [izip, izip.smallest]:
      result = func([1, 2, 3], [4, 5])
      assert isinstance(result, Stream)
      assert list(result) == [(1, 4), (2, 5)]

  def test_longest(self):
    result = izip.longest([1, 2, 3], [4, 5])
    assert isinstance(result, Stream)
    assert list(result) == [(1, 4), (2, 5), (3, None)]

  def test_longest_fillvalue(self):
    result = izip.longest([1, -2, 3], [4, 5], fillvalue=0)
    assert isinstance(result, Stream)
    assert list(result) == [(1, 4), (-2, 5), (3, 0)]
