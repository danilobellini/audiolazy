# -*- coding: utf-8 -*-
"""
Testing module for the lazy_analysis module

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

Created on Tue Aug 07 2012
danilo [dot] bellini [at] gmail [dot] com
"""

import pytest
p = pytest.mark.parametrize

# Audiolazy internal imports
from ..lazy_analysis import window, zcross, maverage, clip
from ..lazy_stream import Stream
from ..lazy_misc import almost_eq
from ..lazy_synth import line


class TestWindow(object):

  @p("wnd", window)
  def test_empty(self, wnd):
    assert wnd(0) == []

  @p("wnd", window)
  @p("M", [1, 2, 3, 4, 16, 128, 256, 512, 1024, 768])
  def test_min_max_len_symmetry(self, wnd, M):
    data = wnd(M)
    assert max(data) <= 1.0
    assert min(data) >= 0.0
    assert len(data) == M
    assert almost_eq(data, data[::-1])


class TestZCross(object):

  def test_empty(self):
    assert list(zcross([])) == []

  @p("n", range(1, 5))
  def test_small_sizes_no_cross(self, n):
    output = zcross(range(n))
    assert isinstance(output, Stream)
    assert list(output) == [0] * n

  @p("d0", [-1, 0, 1])
  @p("d1", [-1, 0, 1])
  def test_pair_combinations(self, d0, d1):
    crossed = 1 if (d0 + d1 == 0) and (d0 != 0) else 0
    assert tuple(zcross([d0, d1])) == (0, crossed)

  @p(("data", "expected"),
     [((0., .1, .5, 1.), (0, 0, 0, 0)),
      ((0., .12, -1.), (0, 0, 1)),
      ((0, -.1, 1), (0, 0, 0)),
      ((1., 0., -.09, .5, -1.), (0, 0, 0, 0, 1)),
      ((1., -.1, -.1, -.2, .05, .1, .05, .2), (0, 0, 0, 1, 0, 0, 0, 1))
     ])
  def test_inputs_with_dot_one_hysteresis(self, data, expected):
    assert tuple(zcross(data, hysteresis=.1)) == expected

  @p("sign", [1, -1])
  def test_first_sign(self, sign):
    data = [0, 1, -1, 3, -4, -.1, .1, 2]
    output = zcross(data, first_sign=sign)
    assert isinstance(output, Stream)
    expected = list(zcross([sign] + data))[1:] # Skip first "zero" sample
    assert list(output) == expected


class TestMAverage(object):

  @p("val", [0, 1, 2, 3., 4.8])
  @p("size", [2, 8, 15, 23])
  @p("strategy", maverage)
  def test_const_input(self, val, size, strategy):
    signal = Stream(val)
    result = strategy(size)(signal)
    small_result = result.take(size - 1)
    assert almost_eq(small_result, list(line(size, 0., val))[1:])
    const_result = result.take(int(2.5 * size))
    for el in const_result:
      assert almost_eq(el, val)


class TestClip(object):

  @p("low", [None, 0, -3])
  @p("high", [None, 0, 5])
  def test_with_range(self, low, high):
    data = range(-10, 10)
    result = clip(data, low=low, high=high)
    assert isinstance(result, Stream)
    if low is None or low < -10:
      low = -10
    if high is None or high > 10:
      high = 10
    expected = [low] * (10 + low) + range(low, high) + [high] * (10 - high)
    assert expected == list(result)

  def test_with_inverted_high_and_low(self):
    with pytest.raises(ValueError):
      clip([], low=4, high=3.9)
