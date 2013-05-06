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
Testing module for the lazy_stream module
"""

import pytest
p = pytest.mark.parametrize

import itertools as it
import operator
import warnings

# Audiolazy internal imports
from ..lazy_stream import Stream, thub, MemoryLeakWarning, StreamTeeHub
from ..lazy_misc import almost_eq
from ..lazy_itertools import imap, ifilter
from ..lazy_math import inf


class TestStream(object):

  def test_no_args(self):
    with pytest.raises(TypeError):
      Stream()

  @p("is_iter", [True, False])
  @p("list_", [range(5), [], [None, Stream, almost_eq, 2.4], [1, "a", 4]])
  def test_from_list(self, list_, is_iter):
    assert list(Stream(iter(list_) if is_iter else list_)) == list_

  @p("tuple_input",[("strange", "tuple with list as fill value".split()),
                    (1,), tuple(), ("abc", [12,15], 8j)]
    )
  def test_from_tuple(self, tuple_input):
    assert tuple(Stream(tuple_input)) == tuple_input

  @p("value", [0, 1, 17, 200])
  def test_xrange(self, value):
    assert list(Stream(xrange(value))) == range(value)

  def test_class_docstring(self):
    x = Stream(it.count())
    y = Stream(3)
    z = 2*x + y
    assert z.take(8) == [3, 5, 7, 9, 11, 13, 15, 17]

  def test_mixed_inputs(self):
    with pytest.raises(TypeError):
      Stream([1, 2, 3], 5)

  @p("d2_type", [list, tuple, Stream])
  @p("d1_iter", [True, False])
  @p("d2_iter", [True, False])
  def test_multiple_inputs_iterable_iterator(self, d2_type, d1_iter, d2_iter):
    data1 = [1, "a", 4]
    data2 = d2_type([7.5, "abc", "abd", 9, it.chain])
    d2copy = data2.copy() if isinstance(data2, Stream) else data2
    data =  [iter(data1) if d1_iter else data1]
    data += [iter(data2) if d2_iter else data2]
    stream_based = Stream(*data)
    it_based = it.chain(data1, d2copy)
    assert list(stream_based) == list(it_based) # No math/casting with float

  def test_non_iterable_input(self):
    data = 25
    for idx, x in zip(range(15), Stream(data)): # Enumerate wouldn't finish
      assert x == data

  def test_multiple_non_iterable_input(self):
    data = [2j+3, 7.5, 75, type(2), Stream]
    for si, di in it.izip(Stream(*data), data * 4):
      assert si == di

  def test_init_docstring_finite(self):
    x = Stream([1, 2, 3]) + Stream([8, 5])
    assert list(x) == [9, 7]

  def test_init_docstring_endless(self):
    x = Stream(1,2,3) + Stream(8,5)
    assert x.take(6) == [9, 7, 11, 6, 10, 8]
    assert x.take(6) == [9, 7, 11, 6, 10, 8]
    assert x.take(3) == [9, 7, 11]
    assert x.take(5) == [6, 10, 8, 9, 7]
    assert x.take(15) == [11, 6, 10, 8, 9, 7, 11, 6, 10, 8, 9, 7, 11, 6, 10]

  def test_tee_copy(self):
    a = Stream([1,2,3])
    b = Stream([8,5])
    c = a.tee()
    d = b.copy()
    assert type(a) == type(c)
    assert type(b) == type(d)
    assert id(a) != id(c)
    assert id(b) != id(d)
    assert iter(a) != iter(c)
    assert iter(b) != iter(d)
    assert list(a) == [1,2,3]
    assert list(c) == [1,2,3]
    assert b.take() == 8
    assert d.take() == 8
    assert b.take() == 5
    assert d.take() == 5
    with pytest.raises(StopIteration):
      b.take()
    with pytest.raises(StopIteration):
      d.take()

  @p(("stream_size", "hop", "block_size"), [(48, 11, 15),
                                            (12, 13, 13),
                                            (42, 5, 22),
                                            (72, 14, 3),
                                            (7, 7, 7),
                                            (12, 8, 8),
                                            (12, 1, 5)]
    )
  def test_blocks(self, stream_size, hop, block_size):
    data = Stream(xrange(stream_size))
    data_copy = data.tee()
    myblocks = data.blocks(size=block_size,
                           hop=hop) # Shouldn't use "data" anymore
    myblocks_rev = Stream(reversed(list(data_copy))).blocks(size=block_size,
                                                            hop=hop)
    for idx, (x, y) in enumerate(it.izip(myblocks, myblocks_rev)):
      assert len(x) == block_size
      assert len(y) == block_size
      startx = idx * hop
      stopx = startx + block_size
      desiredx = [(k if k < stream_size else 0.)\
                   for k in xrange(startx,stopx)]
      assert list(x) == desiredx
      starty = stream_size - 1 - startx
      stopy = stream_size - 1 - stopx
      desiredy = [max(k,0.) for k in xrange(starty, stopy, -1)]
      assert list(y) == desiredy

  def test_unary_operators_and_binary_pow_xor(self):
    a = +Stream([1, 2, 3])
    b = -Stream([8, 5, 2, 17])
    c = Stream(True) ^ Stream([True, False, None]) # xor
    d = b ** a
    assert d.take(3) == [-8,25,-8]
    with pytest.raises(StopIteration):
      d.take()
    assert c.take(2) == [False, True]
    # TypeError: unsupported operand type(s) for ^: 'bool' and 'NoneType'
    with pytest.raises(TypeError):
      c.take()

  def test_getattr_with_methods_and_equalness_operator(self):
    data = "trying again with strings...a bizarre iterable"
    a = Stream(data)
    b = a.copy()
    c = Stream("trying again ", xrange(5), "string", "."*4)
    d = [True for _ in "trying again "] + \
        [False for _ in xrange(5)] + \
        [True for _ in "string"] + \
        [False, True, True, True]
    assert list(a == c) == d
    assert "".join(list(b.upper())) == data.upper()

  def test_getattr_with_non_callable_attributes(self):
           #[(-2+1j), (40+24j), (3+1j), (-3+5j), (8+16j), (8-2j)]
    data = Stream(1 + 2j, 5 + 3j) * Stream(1j, 8, 1 - 1j)
    real = Stream(-2, 40, 3, -3, 8, 8)
    imag = Stream(1, 24, 1, 5, 16, -2)
    assert data.copy().real.take(6) == real.copy().take(6)
    assert data.copy().imag.take(6) == imag.copy().take(6)
    sum_data = data.copy().real + data.copy().imag
    assert sum_data.take(6) == (real + imag).take(6)

  map_filter_data = [xrange(5), xrange(9, 0, -2), [7, 22, -5], [8., 3., 15.],
                     range(20,40,3)]

  @p("data", map_filter_data)
  @p("func", [lambda x: x ** 2, lambda x: x // 2, lambda x: 18])
  def test_map(self, data, func):
    assert map(func, data) == list(Stream(data).map(func))
    assert map(func, data) == list(imap(func, data))

  @p("data", map_filter_data)
  @p("func", [lambda x: x > 0, lambda x: x % 2 == 0, lambda x: False])
  def test_filter(self, data, func):
    assert filter(func, data) == list(Stream(data).filter(func))
    assert filter(func, data) == list(ifilter(func, data))

  def test_no_boolean(self):
    with pytest.raises(TypeError):
      bool(Stream(range(2)))

  @p("op", [operator.div, operator.truediv])
  def test_div_truediv(self, op):
    input1 = [1, 5, 7., 3.3]
    input2 = [9.2, 10, 11, 4.9]
    data = op(Stream(input1), Stream(input2))
    expected = [op(x, y) for x, y in zip(input1, input2)]
    assert isinstance(data, Stream)
    assert list(data) == expected

  def test_next(self):
    """ Streams should have no "next" method! """
    assert not hasattr(Stream(2), "next")

  def test_peek_take(self):
    data = Stream([1, 4, 3, 2])
    assert data.peek(3) == [1, 4, 3]
    assert data.peek() == 1
    assert data.take() == 1
    assert data.peek() == 4
    assert data.peek(3) == [4, 3, 2]
    assert data.peek() == 4
    assert data.take() == 4
    assert data.peek(3) == [3, 2]
    assert data.peek(3, tuple) == (3, 2)
    assert data.peek(inf, tuple) == (3, 2)
    assert data.take(inf, tuple) == (3, 2)
    assert data.peek(1) == []
    assert data.take(1) == []
    assert data.take(inf) == []
    assert Stream([1, 4, 3, 2]).take(inf) == [1, 4, 3, 2]
    with pytest.raises(StopIteration):
      data.peek()
    with pytest.raises(StopIteration):
      data.take()

  def test_skip_periodic_data(self):
    data = Stream(5, Stream, .2)
    assert data.skip(1).peek(4) == [Stream, .2, 5, Stream]
    assert data.peek(4) == [Stream, .2, 5, Stream]
    assert data.skip(3).peek(4) == [Stream, .2, 5, Stream]
    assert data.peek(4) == [Stream, .2, 5, Stream]
    assert data.skip(2).peek(4) == [5, Stream, .2, 5]
    assert data.peek(4) == [5, Stream, .2, 5]

  def test_skip_finite_data(self):
    data = Stream(xrange(25))
    data2 = data.copy()
    assert data.skip(4).peek(4) == [4, 5, 6, 7]
    assert data2.peek(4) == [0, 1, 2, 3]
    assert data2.skip(30).peek(4) == []

  def test_skip_laziness(self):
    memory = {"last": 0}
    def tg():
      while True:
        memory["last"] += 1
        yield memory["last"]

    data = Stream(tg())
    assert data.take(3) == [1, 2, 3]
    data.skip(7)
    assert memory["last"] == 3
    assert data.take() == 11
    assert memory["last"] == 11


class TestThub(object):

  @p("copies", range(5))
  @p("used_copies", range(5))
  def test_stream_tee_hub_memory_leak_warning_and_index_error(self, copies,
                                                              used_copies):
    data = Stream(.5, 8, 7 + 2j)
    data = thub(data, copies)
    assert isinstance(data, StreamTeeHub)
    if copies < used_copies:
      with pytest.raises(IndexError):
        [data * n for n in xrange(used_copies)]
    else:
      [data * n for n in xrange(used_copies)]
      warnings.simplefilter("always")
      with warnings.catch_warnings(record=True) as warnings_list:
        data.__del__()
      if copies != used_copies:
        w = warnings_list.pop()
        assert issubclass(w.category, MemoryLeakWarning)
        assert str(copies - used_copies) in str(w.message)
      assert warnings_list == []
