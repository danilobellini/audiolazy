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
Testing module for the lazy_synth module
"""

import pytest
p = pytest.mark.parametrize

import itertools as it

# Audiolazy internal imports
from ..lazy_synth import (modulo_counter, line, impulse, ones, zeros, zeroes,
                          white_noise, gauss_noise, TableLookup, fadein,
                          fadeout, sin_table, saw_table)
from ..lazy_stream import Stream
from ..lazy_misc import almost_eq, sHz, blocks, rint, lag2freq
from ..lazy_compat import orange, xrange, xzip
from ..lazy_itertools import count
from ..lazy_math import pi, inf


class TestLineFadeInFadeOut(object):

  def test_line(self):
    s, Hz = sHz(rate=2)
    L = line(4 * s, .1, .9)
    assert almost_eq(L, (.1 * x for x in xrange(1, 9)))

  def test_line_append(self):
    s, Hz = sHz(rate=3)
    L1 = line(2 * s, 2, 8)
    L1_should = [2, 3, 4, 5, 6, 7]
    L2 = line(1 * s, 8, -1)
    L2_should = [8, 5, 2]
    L3 = line(2 * s, -1, 9, finish=True)
    L3_should = [-1, 1, 3, 5, 7, 9]
    env = L1.append(L2).append(L3)
    env = env.map(int)
    env_should = L1_should + L2_should + L3_should
    assert list(env) == env_should

  def test_fade_in(self):
    s, Hz = sHz(rate=4)
    L = fadein(2.5 * s)
    assert almost_eq(L, (.1 * x for x in xrange(10)))

  def test_fade_out(self):
    s, Hz = sHz(rate=5)
    L = fadeout(2 * s)
    assert almost_eq(L, (.1 * x for x in xrange(10, 0, -1)))


class TestModuloCounter(object):

  def test_ints(self):
    assert modulo_counter(0, 3, 2).take(8) == [0, 2, 1, 0, 2, 1, 0, 2]

  def test_floats(self):
    assert almost_eq(modulo_counter(1., 5., 3.3).take(10),
                     [1., 4.3, 2.6, .9, 4.2, 2.5, .8, 4.1, 2.4, .7])

  def test_ints_modulo_one(self):
    assert modulo_counter(0, 1, 7).take(3) == [0, 0, 0]
    assert modulo_counter(0, 1, -1).take(4) == [0, 0, 0, 0]
    assert modulo_counter(0, 1, 0).take(5) == [0, 0, 0, 0, 0]

  def test_step_zero(self):
    assert modulo_counter(7, 5, 0).take(2) == [2] * 2
    assert modulo_counter(1, -2, 0).take(4) == [-1] * 4
    assert modulo_counter(0, 3.141592653589793, 0).take(7) == [0] * 7

  def test_streamed_step(self):
    mc = modulo_counter(5, 15, modulo_counter(0, 3, 2))
    assert mc.take(18) == [5, 5, 7, 8, 8, 10, 11, 11, 13, 14, 14, 1, 2, 2, 4,
                           5, 5, 7]

  def test_streamed_start(self):
    mc = modulo_counter(modulo_counter(2, 5, 3), 7, 1)
       # start = [2,0,3,1,4,  2,0,3,1,4,  ...]
    should_mc = (Stream(2, 0, 3, 1, 4) + count()) % 7
    assert mc.take(29) == should_mc.take(29)

  @p("step", [0, 17, -17])
  def test_streamed_start_ignorable_step(self, step):
    mc = modulo_counter(it.count(), 17, step)
    assert mc.take(30) == (orange(17) * 2)[:30]

  def test_streamed_start_and_step(self):
    mc = modulo_counter(Stream(3, 3, 2), 17, it.count())
    should_step =  [0, 0, 1, 3, 6, 10, 15-17, 21-17, 28-17, 36-34, 45-34,
                    55-51, 66-68]
    should_start = [3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3]
    should_mc = [a+b for a,b in xzip(should_start, should_step)]
    assert mc.take(len(should_mc)) == should_mc

  def test_streamed_modulo(self):
    mc = modulo_counter(12, Stream(7, 5), 8)
    assert mc.take(30) == [5, 3, 4, 2, 3, 1, 2, 0, 1, 4] * 3

  def test_streamed_start_and_modulo(self):
    mc = modulo_counter(it.count(), 3 + count(), 1)
    expected = [0, 2, 4, 0, 2, 4, 6, 8, 10, 0, 2, 4, 6, 8, 10, 12,
                14, 16, 18, 20, 22, 0, 2, 4, 6, 8, 10, 12, 14, 16]
    assert mc.take(len(expected)) == expected

  def test_all_inputs_streamed(self):
    mc1 = modulo_counter(it.count(), 3 + count(), Stream(0, 1))
    mc2 = modulo_counter(0, 3 + count(), 1 + Stream(0, 1))
    expected = [0, 1, 3, 4, 6, 7, 0, 1, 3, 4, 6, 7, 9, 10, 12, 13,
                15, 16, 18, 19, 21, 22, 24, 25, 0, 1, 3, 4, 6, 7]
    assert mc1.take(len(expected)) == mc2.take(len(expected)) == expected

  classes = (float, Stream)

  @p("start", [-1e-16, -1e-100])
  @p("cstart", classes)
  @p("cmodulo", classes)
  @p("cstep", classes)
  def test_bizarre_modulo(self, start, cstart, cmodulo, cstep):
    # Not really a modulo counter issue, but used by modulo counter
    for step in xrange(2, 900):
      mc = modulo_counter(cstart(start),
                          cmodulo(step),
                          cstep(step))
      assert all(mc.limit(4) < step)

@p(("func", "data"),
   [(ones, 1.0),
    (zeros, 0.0),
    (zeroes, 0.0)
   ])
class TestOnesZerosZeroes(object):

  def test_no_input(self, func, data):
    my_stream = func()
    assert isinstance(my_stream, Stream)
    assert my_stream.take(25) == [data] * 25

  def test_inf_input(self, func, data):
    my_stream = func(inf)
    assert isinstance(my_stream, Stream)
    assert my_stream.take(30) == [data] * 30

  @p("dur", [-1, 0, .4, .5, 1, 2, 10])
  def test_finite_duration(self, func, data, dur):
    my_stream = func(dur)
    assert isinstance(my_stream, Stream)
    dur_int = max(rint(dur), 0)
    assert list(my_stream) == [data] * dur_int


class TestWhiteNoise(object):

  def test_no_input(self):
    my_stream = white_noise()
    assert isinstance(my_stream, Stream)
    for el in my_stream.take(27):
      assert -1 <= el <= 1

  @p("high", [1, 0, -.042])
  def test_inf_input(self, high):
    my_stream = white_noise(inf, high=high)
    assert isinstance(my_stream, Stream)
    for el in my_stream.take(32):
      assert -1 <= el <= high

  @p("dur", [-1, 0, .4, .5, 1, 2, 10])
  @p("low", [0, .17])
  def test_finite_duration(self, dur, low):
    my_stream = white_noise(dur, low=low)
    assert isinstance(my_stream, Stream)
    dur_int = max(rint(dur), 0)
    my_list = list(my_stream)
    assert len(my_list) == dur_int
    for el in my_list:
      assert low <= el <= 1


class TestGaussNoise(object):

  def test_no_input(self):
    my_stream = gauss_noise()
    assert isinstance(my_stream, Stream)
    assert len(my_stream.take(100)) == 100

  def test_inf_input(self):
    my_stream = gauss_noise(inf)
    assert isinstance(my_stream, Stream)
    assert len(my_stream.take(100)) == 100

  @p("dur", [-1, 0, .4, .5, 1, 2, 10])
  def test_finite_duration(self, dur):
    my_stream = gauss_noise(dur)
    assert isinstance(my_stream, Stream)
    dur_int = max(rint(dur), 0)
    my_list = list(my_stream)
    assert len(my_list) == dur_int


class TestTableLookup(object):

  def test_binary_rbinary_unary(self):
    a = TableLookup([0, 1, 2])
    b = 1 - a
    c = b * 3
    assert b.table == [1, 0, -1]
    assert (-b).table == [-1, 0, 1]
    assert c.table == [3, 0, -3]
    assert (a + b - c).table == [-2, 1, 4]

  def test_sin_basics(self):
    assert sin_table[0] == 0
    assert almost_eq(sin_table(pi, phase=pi/2).take(6), [1, -1] * 3)
    s30 = .5 * 2 ** .5 # sin(30 degrees)
    assert almost_eq(sin_table(pi/2, phase=pi/4).take(12),
                     [s30, s30, -s30, -s30] * 3)
    expected_pi_over_2 = [0., s30, 1., s30, 0., -s30, -1., -s30]
    # Assert with "diff" since it has zeros
    assert almost_eq.diff(sin_table(pi/4).take(32), expected_pi_over_2 * 4)

  def test_saw_basics(self):
    assert saw_table[0] == -1
    assert saw_table[-1] == 1
    assert saw_table[1] - saw_table[0] > 0
    data = saw_table(lag2freq(30)).take(30)
    first_step = data[1] - data[0]
    assert first_step > 0
    for d0, d1 in blocks(data, size=2, hop=1):
      assert d1 - d0 > 0 # Should be monotonically increasing
      assert almost_eq(d1 - d0, first_step) # Should have constant derivative


class TestImpulse(object):

  def test_no_input(self):
    delta = impulse()
    assert isinstance(delta, Stream)
    assert delta.take(25) == [1.] + list(zeros(24))

  def test_inf_input(self):
    delta = impulse(inf)
    assert isinstance(delta, Stream)
    assert delta.take(17) == [1.] + list(zeros(16))

  def test_integer(self):
    delta = impulse(one=1, zero=0)
    assert isinstance(delta, Stream)
    assert delta.take(22) == [1] + [0] * 21

  @p("dur", [-1, 0, .4, .5, 1, 2, 10])
  def test_finite_duration(self, dur):
    delta = impulse(dur)
    assert isinstance(delta, Stream)
    dur_int = max(rint(dur), 0)
    if dur_int == 0:
      assert list(delta) == []
    else:
      assert list(delta) == [1.] + [0.] * (dur_int - 1)
