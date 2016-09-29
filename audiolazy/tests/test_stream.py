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
Testing module for the lazy_stream module
"""

import pytest
p = pytest.mark.parametrize

import itertools as it
import operator, warnings, sys, gc
from collections import deque

# Audiolazy internal imports
from ..lazy_stream import Stream, thub, MemoryLeakWarning, StreamTeeHub
from ..lazy_misc import almost_eq
from ..lazy_compat import (orange, xrange, xzip, xmap, xfilter, NEXT_NAME,
                           PYTHON2)
from ..lazy_math import inf, nan, pi

from . import skipper
operator.div = getattr(operator, "div", skipper("There's no operator.div"))


class TestStream(object):

  def test_no_args(self):
    with pytest.raises(TypeError):
      Stream()

  @p("is_iter", [True, False])
  @p("list_", [orange(5), [], [None, Stream, almost_eq, 2.4], [1, "a", 4]])
  def test_from_list(self, list_, is_iter):
    assert list(Stream(iter(list_) if is_iter else list_)) == list_

  @p("tuple_input",[("strange", "tuple with list as fill value".split()),
                    (1,), tuple(), ("abc", [12,15], 8j)]
    )
  def test_from_tuple(self, tuple_input):
    assert tuple(Stream(tuple_input)) == tuple_input

  @p("value", [0, 1, 17, 200])
  def test_lazy_range(self, value):
    assert list(Stream(xrange(value))) == orange(value)

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
    for idx, x in xzip(xrange(15), Stream(data)): # Enumerate wouldn't finish
      assert x == data

  def test_multiple_non_iterable_input(self):
    data = [2j+3, 7.5, 75, type(2), Stream]
    for si, di in xzip(Stream(*data), data * 4):
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

  def test_copy(self):
    a = Stream([1,2,3])
    b = Stream([8,5])
    c = a.copy()
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
    data_copy = data.copy()
    myblocks = data.blocks(size=block_size,
                           hop=hop) # Shouldn't use "data" anymore
    myblocks_rev = Stream(reversed(list(data_copy))).blocks(size=block_size,
                                                            hop=hop)
    for idx, (x, y) in enumerate(xzip(myblocks, myblocks_rev)):
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

  def test_no_boolean(self):
    with pytest.raises(TypeError):
      bool(Stream(xrange(2)))

  @p("op", [operator.div, operator.truediv])
  def test_div_truediv(self, op):
    input1 = [1, 5, 7., 3.3]
    input2 = [9.2, 10, 11, 4.9]
    data = op(Stream(input1), Stream(input2))
    expected = [operator.truediv(x, y) for x, y in xzip(input1, input2)]
    assert isinstance(data, Stream)
    assert list(data) == expected

  def test_next(self):
    """ Streams should have no "next" method! """
    assert not hasattr(Stream(2), NEXT_NAME)

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

  def test_limit_from_beginning_from_finite_stream(self):
    assert Stream(xrange(25)).limit(10).take(inf) == orange(10)
    assert Stream(xrange(25)).limit(40).take(inf) == orange(25)
    assert Stream(xrange(25)).limit(24).take(inf) == orange(24)
    assert Stream(xrange(25)).limit(25).take(inf) == orange(25)
    assert Stream(xrange(25)).limit(26).take(inf) == orange(25)

  def test_limit_with_skip_from_finite_stream(self):
    assert Stream(xrange(45)).skip(2).limit(13).take(inf) == orange(2, 15)
    assert Stream(xrange(45)).limit(13).skip(3).take(inf) == orange(3, 13)

  @p("noise", [-.3, 0, .1])
  def test_limit_from_periodic_stream(self, noise):
    assert Stream(0, 1, 2).limit(7 + noise).peek(10) == [0, 1, 2, 0, 1, 2, 0]
    data = Stream(-1, .2, it)
    assert data.skip(2).limit(9 + noise).peek(15) == [it, -1, .2] * 3

  @p("noise", [-.3, 0., .1])
  def test_take_peek_skip_with_float(self, noise):
    data = [1.2, 7.7, 1e-3, 1e-17, 2e8, 27.1, 14.003, 1.0001, 7.3e5, 0.]
    ds = Stream(data)
    assert ds.limit(5 + noise).peek(10 - noise) == data[:5]
    assert ds.skip(1 + noise).limit(3 - noise).peek(10 + noise) == data[1:4]
    ds = Stream(data)
    assert ds.skip(2 + noise).peek(20 + noise) == data[2:]
    assert ds.skip(3 - noise).peek(20 - noise) == data[5:]
    assert ds.skip(4 + noise).peek(1 + noise) == [data[9]]
    ds = Stream(data)
    assert ds.skip(4 - noise).peek(2 - noise) == data[4:6]
    assert ds.skip(1 - noise).take(2 + noise) == data[5:7]
    assert ds.peek(inf) == data[7:]
    assert ds.take(inf) == data[7:]

  def test_take_peek_inf(self):
    data = list(xrange(30))
    ds1 = Stream(data)
    ds1.take(3)
    assert ds1.peek(inf) == data[3:]
    assert ds1.take(inf) == data[3:]
    ds2 = Stream(data)
    ds2.take(4)
    assert ds2.peek(inf * 2e-18) == data[4:] # Positive float
    assert ds2.take(inf * 1e-25) == data[4:]
    ds3 = Stream(data)
    ds3.take(1)
    assert ds3.peek(inf * 43) == data[1:] # Positive int
    assert ds3.take(inf * 200) == data[1:]

  def test_take_peek_nan(self):
    for dur in [nan, nan * 23, nan * -5, nan * .3, nan * -.18, nan * 0]:
      assert Stream(29).take(dur) == []
      assert Stream([23]).peek(dur) == []

  def test_take_peek_negative_or_zero(self):
    for dur in [-inf, inf * -1e-16, -1, -2, -.18, 0, 0., -0.]:
      assert Stream(1, 2, 3).take(dur) == []
      assert Stream(-1, -23).peek(dur) == []

  def test_take_peek_not_int_nor_float(self):
    for dur in [Stream, it, [], (2, 3), 3j]:
      with pytest.raises(TypeError):
        Stream([]).take(dur)
      with pytest.raises(TypeError):
        Stream([128]).peek(dur)

  def test_take_peek_none(self):
    data = Stream([Stream, it, [], (2, 3), 3j])
    assert data.peek() is Stream
    assert data.take() is Stream
    assert data.peek() is it
    assert data.take() is it
    assert data.peek() == []
    assert data.take() == []
    assert data.peek() == (2, 3)
    assert data.take() == (2, 3)
    assert data.peek() == 3j
    assert data.take() == 3j
    with pytest.raises(StopIteration):
      data.peek()
    with pytest.raises(StopIteration):
      data.take()

  @p("constructor", [list, tuple, set, deque])
  def test_take_peek_constructor(self, constructor):
    ds = Stream([1, 2, 3] * 12)
    assert ds.peek(constructor=constructor) == 1
    assert ds.take(constructor=constructor) == 1
    assert ds.peek(constructor=constructor) == 2
    assert ds.take(constructor=constructor) == 2
    assert ds.peek(3, constructor=constructor) == constructor([3, 1, 2])
    assert ds.take(4, constructor=constructor) == constructor([3, 1, 2, 3])
    remain = constructor([1, 2, 3] * 10)
    assert ds.peek(inf, constructor=constructor) == remain
    assert ds.take(inf, constructor=constructor) == remain
    assert ds.peek(3, constructor=constructor) == constructor()
    assert ds.take(5, constructor=constructor) == constructor()

  def test_abs_step_by_step(self):
    data = [-1, 5, 2 + 3j, 8, -.7]
    ds = abs(Stream(data))
    assert isinstance(ds, Stream)
    assert ds.take(len(data)) == [abs(el) for el in data]
    assert ds.take(1) == [] # Finished

  def test_abs_take_peek(self):
    assert abs(Stream([])).peek(9) == [] # From empty Stream
    assert abs(Stream([5, -12, 14j, -2j, 0])).take(inf) == [5, 12, 14., 2., 0]
    data = abs(Stream([1.2, -1.57e-3, -(pi ** 2), -2j,  8 - 4j]))
    assert almost_eq(data.peek(inf), [1.2, 1.57e-3, pi ** 2, 2., 4 * 5 ** .5])
    assert data.take() == 1.2
    assert almost_eq(data.take(), 1.57e-3)


class TestEveryMapFilter(object):
  """
  Tests Stream.map, Stream.filter, StreamTeeHub.map, StreamTeeHub.filter,
  lazy_itertools.imap and lazy_itertools.ifilter (map and filter in Python 3)
  """

  map_filter_data = [orange(5), orange(9, 0, -2), [7, 22, -5], [8., 3., 15.],
                     orange(20,40,3)]

  @p("data", map_filter_data)
  @p("func", [lambda x: x ** 2, lambda x: x // 2, lambda x: 18])
  def test_map(self, data, func):
    expected = [func(x) for x in data]
    assert list(Stream(data).map(func)) == expected
    assert list(xmap(func, data)) == expected # Tests the test...
    dt = thub(data, 2)
    assert isinstance(dt, StreamTeeHub)
    dt_data = dt.map(func)
    assert isinstance(dt_data, Stream)
    assert dt_data.take(inf) == expected
    assert list(dt.map(func)) == expected # Second copy
    with pytest.raises(IndexError):
      dt.map(func)

  @p("data", map_filter_data)
  @p("func", [lambda x: x > 0, lambda x: x % 2 == 0, lambda x: False])
  def test_filter(self, data, func):
    expected = [x for x in data if func(x)]
    assert list(Stream(data).filter(func)) == expected
    assert list(xfilter(func, data)) == expected # Tests the test...
    dt = thub(data, 2)
    assert isinstance(dt, StreamTeeHub)
    dt_data = dt.filter(func)
    assert isinstance(dt_data, Stream)
    assert dt_data.take(inf) == expected
    assert list(dt.filter(func)) == expected # Second copy
    with pytest.raises(IndexError):
      dt.filter(func)


class TestThub(object):

  copies_pairs = [ # Pairs (copies, used_copies) that worth testing
    (0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3),
    (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
  ]

  @p(("copies", "used_copies"), copies_pairs)
  def test_memory_leak_warning_mock(self, copies, used_copies, monkeypatch):
    # Mock to bypass the warnings.filters and the weird "pre-filters"
    # __warningregistry__ behavior
    def warn(w):
      warn.warnings_list.append(w)
    warn.warnings_list = []

    gc.collect() # Clear warnings that might be pending from other tests
    monkeypatch.setattr(warnings, "warn", warn)

    def apply_test():
      types = [int, int, int, float, complex, float, type(thub), type(pytest)]
      sig = [4, 3, 2, 7e-18, 9j, .2, thub, pytest]
      data = thub(sig, copies)
      assert isinstance(data, StreamTeeHub)
      safe_copies = min(copies, used_copies)
      assert all(all(types == data.map(type)) for n in xrange(safe_copies))
      if copies < used_copies:
        with pytest.raises(IndexError):
          Stream(data)
      data.__del__() # Forcefully calls the destructor twice
      data.__del__()

    apply_test()
    gc.collect()

    # Expected results
    if copies > used_copies:
      w = warn.warnings_list.pop() # The only warning
      assert isinstance(w, MemoryLeakWarning)
      parts = ["StreamTeeHub", str(copies - used_copies)]
      assert all(part in str(w) for part in parts)
    assert not warn.warnings_list

  @p(("copies", "used_copies"), copies_pairs)
  def test_memory_leak_warning_recwarn(self, copies, used_copies, recwarn):
    sig = Stream(.5, 8, 7 + 2j)
    data = thub(sig, copies)
    assert isinstance(data, StreamTeeHub)
    if copies < used_copies:
      with pytest.raises(IndexError):
        [data * n for n in xrange(used_copies)]
    else:
      [data * n for n in xrange(used_copies)] # Use the copies

      # Clear warnings that might be pending from other tests
      gc.collect()
      recwarn.clear()

      # Clear warning registry from other tests in function globals
      if PYTHON2:
        del_globals = StreamTeeHub.__del__.im_func.func_globals
      else:
        del_globals = StreamTeeHub.__del__.__globals__
      wr = "__warningregistry__"
      if wr in del_globals:
        del_globals[wr].clear()

      # From now, look for every warning out there
      warnings.resetwarnings()
      warnings.simplefilter("always")

      # Forcefully calls the destructor more than once
      data.__del__()
      data.__del__()
      del data
      gc.collect()

      # Expected results
      if copies != used_copies:
        w = recwarn.pop()
        assert issubclass(w.category, MemoryLeakWarning)
        assert str(copies - used_copies) in str(w.message)
      assert not recwarn.list

  def test_take_peek(self):
    data = Stream(1, 2, 3).limit(50)
    data = thub(data, 2)
    assert data.peek() == 1
    assert data.peek(.2) == []
    assert data.peek(1) == [1]
    with pytest.raises(AttributeError):
      data.take()
    assert data.peek(22) == Stream(1, 2, 3).take(22)
    assert data.peek(42.2) == Stream(1, 2, 3).take(42)
    with pytest.raises(AttributeError):
      data.take(2)
    assert data.peek(57.8) == Stream(1, 2, 3).take(50)
    assert data.peek(inf) == Stream(1, 2, 3).take(50)

  @p("noise", [-.3, 0, .1])
  def test_limit(self, noise):
    source = [.1, -.2, 18, it, Stream]
    length = len(source)
    data = Stream(*source).limit(4 * length)
    data = thub(data, 3)

    # First copy
    first_copy = data.limit(length + noise)
    assert isinstance(first_copy, Stream)
    assert not isinstance(first_copy, StreamTeeHub)
    assert list(first_copy) == source

    # Second copy
    assert data.peek(3 - noise) == source[:3]
    assert Stream(data).take(inf) == 4 * source

    # Third copy
    third_copy = data.limit(5 * length + noise)
    assert isinstance(third_copy, Stream)
    assert not isinstance(third_copy, StreamTeeHub)
    assert third_copy.take(inf) == 4 * source

    # No more copies
    assert isinstance(data, StreamTeeHub)
    with pytest.raises(IndexError):
      data.limit(3)

  @p("noise", [-.3, 0, .1])
  def test_skip_append(self, noise):
    source = [9, 14, -7, noise]
    length = len(source)
    data = Stream(*source).limit(7 * length)
    data = thub(data, 3)

    # First copy
    first_copy = data.skip(length + 1)
    assert isinstance(first_copy, Stream)
    assert not isinstance(first_copy, StreamTeeHub)
    assert first_copy is first_copy.append([8])
    assert list(first_copy) == source[1:] + 5 * source + [8]

    # Second and third copies
    assert data.skip(1 + noise).peek(3 - noise) == source[1:4]
    assert data.append([1]).skip(length - noise).take(inf) == 6 * source + [1]

    # No more copies
    assert isinstance(data, StreamTeeHub)
    with pytest.raises(IndexError):
      data.skip(1)
    with pytest.raises(IndexError):
      data.append(3)

  @p("size", [4, 5, 6])
  @p("hop", [None, 1, 5])
  def test_blocks(self, size, hop):
    copies = 8 - size
    source = Stream(7, 8, 9, -1, -1, -1, -1).take(40)
    data = thub(source, copies)
    expected = list(Stream(source).blocks(size=size, hop=hop).map(list))
    for _ in xrange(copies):
      blks = data.blocks(size=size, hop=hop).map(list)
      assert isinstance(blks, Stream)
      assert not isinstance(blks, StreamTeeHub)
      assert blks.take(inf) == expected
    with pytest.raises(IndexError):
      data.blocks(size=size, hop=hop)
