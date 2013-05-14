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
# Created on Sun Sep 09 2012
# danilo [dot] bellini [at] gmail [dot] com
"""
Testing module for the lazy_filters module
"""

import pytest
p = pytest.mark.parametrize

import operator
import itertools as it
from math import pi
from functools import reduce

# Audiolazy internal imports
from ..lazy_filters import (ZFilter, z, CascadeFilter, ParallelFilter,
                            resonator, lowpass, highpass)
from ..lazy_misc import almost_eq, zero_pad
from ..lazy_compat import orange, xrange, xzip, xmap
from ..lazy_itertools import cycle, chain
from ..lazy_stream import Stream
from ..lazy_math import dB10, dB20, inf

from . import skipper
operator.div = getattr(operator, "div", skipper("There's no operator.div"))


class TestZFilter(object):
  data = [-7, 3] + orange(10) + [-50, 0] + orange(70, -70, -11) # Only ints
  alpha = [-.5, -.2, -.1, 0, .1, .2, .5] # Attenuation Value for filters
  delays = orange(1, 5)

  def test_z_identity(self):
    my_filter = z ** 0
    assert list(my_filter(self.data)) == self.data

  @p("amp_factor", [-10, 3, 0, .2, 8])
  def test_z_simple_amplification(self, amp_factor):
    my_filter1 = amp_factor * z ** 0
    my_filter2 = z ** 0 * amp_factor
    expected = [amp_factor * di for di in self.data]
    op = operator.eq if isinstance(amp_factor, int) else almost_eq
    assert op(list(my_filter1(self.data)), expected)
    assert op(list(my_filter2(self.data)), expected)


  @p("delay", orange(1, 5)) # Slice with zero would make data empty
  def test_z_int_delay(self, delay):
    my_filter = + z ** -delay
    assert list(my_filter(self.data)) == [0] * delay + self.data[:-delay]

  @p("amp_factor", [1, -105, 43, 0, .128, 18])
  @p("delay", orange(1, 7)) # Slice with zero would make data empty
  def test_z_int_delay_with_amplification(self, amp_factor, delay):
    my_filter1 = amp_factor * z ** -delay
    my_filter2 = z ** -delay * amp_factor
    expected = [amp_factor * di for di in ([0.] * delay + self.data[:-delay])]
    op = operator.eq if isinstance(amp_factor, int) else almost_eq
    assert op(list(my_filter1(self.data)), expected)
    assert op(list(my_filter2(self.data)), expected)

  def test_z_fir_size_2(self):
    my_filter = 1 + z ** -1
    expected = [a + b for a, b in xzip(self.data, [0] + self.data[:-1])]
    assert list(my_filter(self.data)) == expected

  def test_z_fir_size_2_hybrid_amplification(self):
    my_filter = 2 * (3. - 5 * z ** -1)
    expected = (6.*a - 10*b for a, b in xzip(self.data,
                                             [0.] + self.data[:-1]))
    assert almost_eq(my_filter(self.data), expected)

  amp_list = [1, -15, 45, 0, .81, 17]
  @p( ("num_delays", "amp_factor"),
      chain.from_iterable(
        (lambda amps, dls: [
          [(d, a) for a in it.combinations_with_replacement(amps, d + 1)]
          for d in dls
        ])(amp_list, delays)
      ).take(inf)
  )
  def test_z_many_fir_sizes_and_amplifications(self, num_delays, amp_factor):
    my_filter = sum(amp_factor[delay] * z ** -delay
                    for delay in xrange(num_delays + 1))
    expected = sum(amp_factor[delay] * (z ** -delay)(self.data)
                   for delay in xrange(num_delays + 1))
    assert almost_eq(my_filter(self.data), expected)

  def test_z_fir_multiplication(self):
    my_filter = 8 * (2 * z**-3 - 5 * z**-4) * z ** 2 * 7
    expected = [56*2*a - 56*5*b for a, b in xzip([0] + self.data[:-1],
                                                 [0, 0] + self.data[:-2])]
    assert list(my_filter(self.data)) == expected
  @p("a", alpha)
  def test_z_one_pole(self, a):
    my_filter = 1 / (1 + a * z ** -1)
    expected = [x for x in self.data]
    for idx in xrange(1,len(expected)):
      expected[idx] -= a * expected[idx-1]
    assert almost_eq(my_filter(self.data), expected)

  @p("a", alpha)
  @p("b", alpha)
  @p("idx_num1", orange(-3,1))
  @p("idx_den1", orange(-1,3,19))
  @p("idx_num2", orange(-2,0,14))
  @p("idx_den2", orange(-18,1))
  def test_z_division(self, a, b, idx_num1, idx_den1, idx_num2, idx_den2):
    fa, fb, fc, fd = (a * z ** idx_num1, 2 + b * z ** idx_den1,
                      3 * z ** idx_num2, 1 + 5 * z ** idx_den2)
    my_filter1 = fa / fb
    my_filter2 = fc / fd
    my_filter = my_filter1 / my_filter2
    idx_corr = max(idx_num2, idx_num2 + idx_den1)
    num_filter = fa * fd * (z ** -idx_corr)
    den_filter = fb * fc * (z ** -idx_corr)
    assert almost_eq(my_filter.numpoly.terms(), num_filter.numpoly.terms())
    assert almost_eq(my_filter.denpoly.terms(), den_filter.numpoly.terms())

  @p("filt", [1 / z / 1,
              (1 / z) ** 1,
              (1 / z ** -1) ** -1,
             ])
  def test_z_power_alone(self, filt):
    assert almost_eq(filt(self.data), [0.] + self.data[:-1])

  @p("a", [a for a in alpha if a != 0])
  @p("div", [operator.div, operator.truediv])
  @p("zero", [0., 0])
  def test_z_div_truediv_unit_delay_divided_by_constant(self, a, div, zero):
    for el in [a, int(10 * a)]:
      div_filter = div(z ** -1, a)
      div_expected = xmap(lambda x: operator.truediv(x, a),
                          [zero] + self.data[:-1])
      assert almost_eq(div_filter(self.data, zero=zero), div_expected)

  @p("a", alpha)
  def test_z_div_truediv_constant_over_delay(self, a):
    div_inv_filter = operator.div(a, 1 + z ** -1)
    truediv_inv_filter = operator.truediv(a, 1 + z ** -1)
    expected = [a*x for x in self.data]
    for idx in xrange(1,len(expected)):
      expected[idx] -= expected[idx-1]
    assert almost_eq(div_inv_filter(self.data), expected)
    assert almost_eq(truediv_inv_filter(self.data), expected)

  def test_z_power_with_denominator(self):
    my_filter = (z ** -1 / (1 + z ** -2)) ** 1 # y[n] == x[n-1] - y[n-2]
    expected = []
    mem2, mem1, xlast = 0., 0., 0.
    for di in self.data:
      newy = xlast - mem2
      mem2, mem1, xlast = mem1, newy, di
      expected.append(newy)
    assert almost_eq(my_filter(self.data), expected)

  @p("a", alpha)
  def test_z_grouped_powers(self, a):
    base_filter = (1 + a * z ** -1)
    my_filter1 = base_filter ** -1
    my_filter2 = 3 * base_filter ** -2
    my_filter3 = base_filter ** 2
    my_filter4 = a * base_filter ** 0
    my_filter5 = (base_filter ** 3) * (base_filter ** -4)
    my_filter6 = ((1 - a * z ** -1) / base_filter ** 2) ** 2
    assert almost_eq(my_filter1.numerator, [1.])
    assert almost_eq(my_filter1.denominator, [1., a] if a != 0 else [1.])
    assert almost_eq(my_filter2.numerator, [3.])
    assert almost_eq(my_filter2.denominator, [1., 2*a, a*a]
                                             if a != 0 else [1.])
    assert almost_eq(my_filter3.numerator, [1., 2*a, a*a] if a != 0 else [1.])
    assert almost_eq(my_filter3.denominator, [1.])
    assert almost_eq(my_filter4.numerator, [a] if a != 0 else [])
    assert almost_eq(my_filter4.denominator, [1.])
    assert almost_eq(my_filter5.numerator, [1., 3*a, 3*a*a, a*a*a]
                                           if a != 0 else [1.])
    assert almost_eq(my_filter5.denominator,
                     [1., 4*a, 6*a*a, 4*a*a*a, a*a*a*a] if a != 0 else [1.])
    assert almost_eq(my_filter6.numerator, [1., -2*a, a*a]
                                           if a != 0 else [1.])
    assert almost_eq(my_filter6.denominator,
                     [1., 4*a, 6*a*a, 4*a*a*a, a*a*a*a] if a != 0 else [1.])

  @p("a", alpha)
  def test_z_one_pole_neg_afterwards(self, a):
    my_filter = -(1 / (1 + a * z ** -1))
    expected = [x for x in self.data]
    for idx in xrange(1,len(expected)):
      expected[idx] -= a * expected[idx-1]
    expected = (-x for x in expected)
    assert almost_eq(my_filter(self.data), expected)

  @p("a", alpha)
  def test_z_one_pole_added_one_pole(self, a):
    my_filter1 = -(3 / (1 + a * z ** -1))
    my_filter2 = +(2 / (1 - a * z ** -1))
    my_filter = my_filter1 + my_filter2
    assert almost_eq(my_filter.numerator, [-1., 5*a] if a != 0 else [-1.])
    assert almost_eq(my_filter.denominator, [1., 0., -a*a]
                                            if a != 0 else [1.])

  @p("a", alpha)
  def test_z_one_pole_added_to_a_number(self, a):
    my_filter = -(5 / (1 - a * z ** -1)) + a
    assert almost_eq(my_filter.numerator, [-5 + a, -a*a] if a != 0 else [-5])
    assert almost_eq(my_filter.denominator, [1., -a] if a != 0 else [1.])

  @p("a", alpha)
  def test_one_pole_numerator_denominator_constructor(self, a):
    my_filter = ZFilter(numerator=[1.], denominator=[1., -a])
    expected = [x for x in self.data]
    for idx in xrange(1,len(expected)):
      expected[idx] += a * expected[idx-1]
    assert almost_eq(list(my_filter(self.data)), expected)

  @p("delay", orange(1, 7))
  def test_diff_twice_only_numerator_one_delay(self, delay):
    data = z ** -delay
    ddz = data.diff()
    assert almost_eq(ddz.numerator,
                     [0] * delay + [0, -delay])
    assert almost_eq(ddz.denominator, [1])
    ddz2 = ddz.diff()
    assert almost_eq(ddz2.numerator,
                     [0] * delay + [0, 0, delay * (delay + 1)])
    assert almost_eq(ddz2.denominator, [1])
    ddz2_alt = data.diff(2)
    assert almost_eq(ddz2.numerator, ddz2_alt.numerator)
    assert almost_eq(ddz2.denominator, ddz2_alt.denominator)

  def test_diff(self):
    filt = (1 + z ** -1) / (1 - z ** -1)
    ddz = filt.diff()
    assert almost_eq(ddz.numerator, [0, 0, -2])
    assert almost_eq(ddz.denominator, [1, -2, 1])

  @p("a", alpha)
  @p("A", [.9, -.2])
  @p("mul", [-z, 1/(1 + z**-2), 8])
  def test_diff_with_eq_operator_and_mul_after(self, a, A, mul):
    num = a - A * (1 - a) * z ** -1
    den = 1 - A * z ** -1 + A ** 2 * z ** -2
    filt = num / den
    numd = A * (1 - a) * z ** -2
    dend = A * z ** -2 - 2 * A ** 2 * z ** -3
    muld = mul.diff() if isinstance(mul, ZFilter) else 0
    assert almost_eq(num.diff(), numd)
    assert almost_eq(den.diff(), dend)
    assert almost_eq(num.diff(mul_after=mul), numd * mul)
    assert almost_eq(den.diff(mul_after=mul), dend * mul)
    filtd_num = numd * den - dend * num
    filtd = filtd_num / den ** 2
    assert almost_eq(filt.diff(), filtd)
    assert almost_eq(filt.diff(mul_after=mul), filtd * mul)
    numd2 = -2 * A * (1 - a) * z ** -3
    numd2ma = (numd2 * mul + muld * numd) * mul
    dend2 = -2 * A * z ** -3 + 6 * A ** 2 * z ** -4
    dend2ma = (dend2 * mul + muld * dend) * mul
    assert almost_eq(num.diff(2), numd2)
    assert almost_eq(den.diff(2), dend2)
    assert almost_eq(num.diff(n=2, mul_after=mul), numd2ma)
    assert almost_eq(den.diff(n=2, mul_after=mul), dend2ma)

    filtd2 = ((numd2 * den - num * dend2) * den - 2 * filtd_num * dend
             ) / den ** 3
    filt_to_test = filt.diff(n=2)
    assert almost_eq.diff(filt_to_test.numerator, filtd2.numerator,
                          max_diff=1e-10)
    assert almost_eq.diff(filt_to_test.denominator, filtd2.denominator,
                          max_diff=1e-10)

    if 1/(1 + z**-2) != mul: # Too difficult to group together with others
      filtd2ma = ((numd2 * den - num * dend2) * mul * den +
                  filtd_num * (muld * den - 2 * mul * dend)
                 ) * mul / den ** 3
      filt_to_testma = filt.diff(n=2, mul_after=mul)
      assert almost_eq.diff(filt_to_testma.numerator, filtd2ma.numerator,
                            max_diff=1e-10)
      assert almost_eq.diff(filt_to_testma.denominator, filtd2ma.denominator,
                            max_diff=1e-10)

  @p("delay", delays)
  def test_one_delay_variable_gain(self, delay):
    gain = cycle(self.alpha)
    filt = gain * z ** -delay
    length = 50
    assert isinstance(filt, ZFilter)
    data_stream = cycle(self.alpha) * zero_pad(cycle(self.data), left=delay)
    expected = data_stream.take(length)
    result_stream = filt(cycle(self.data))
    assert isinstance(result_stream, Stream)
    result = result_stream.take(length)
    assert almost_eq(result, expected)


@p("filt_class", [CascadeFilter, ParallelFilter])
class TestCascadeAndParallelFilters(object):

  def test_add(self, filt_class):
    filt1 = filt_class(z)
    filt2 = filt_class(z + 3)
    filt_sum = filt1 + filt2
    assert isinstance(filt_sum, filt_class)
    assert filt_sum == filt_class(z, z + 3)

  def test_mul(self, filt_class):
    filt = filt_class(1 - z ** -1)
    filt_prod = filt * 3
    assert isinstance(filt_prod, filt_class)
    assert filt_prod == filt_class(1 - z ** -1, 1 - z ** -1, 1 - z ** -1)

  @p("filts",
     [(lambda data: data ** 2),
      (z ** -1, lambda data: data + 4),
      (1 / z ** -2, (lambda data: 0.), z** -1),
     ])
  def test_non_linear(self, filts, filt_class):
    filt = filt_class(filts)
    assert isinstance(filt, filt_class)
    assert not filt.is_linear()
    with pytest.raises(AttributeError):
      filt.numpoly
    with pytest.raises(AttributeError):
      filt.denpoly
    with pytest.raises(AttributeError):
      filt.freq_response(pi / 2)

  def test_const_filter(self, filt_class):
    data = [2, 4, 3, 7 -1, -8]
    filt1 = filt_class(*data)
    filt2 = filt_class(data)
    func = operator.mul if filt_class == CascadeFilter else operator.add
    expected_value = reduce(func, data)
    count = 10
    for d in data:
      expected = [d * expected_value] * count
      assert filt1(Stream(d)).take(count) == expected
      assert filt2(Stream(d)).take(count) == expected


class TestCascadeOrParallelFilter(object):

  data_values = [orange(3),
                 Stream([5., 4., 6., 7., 12., -2.]),
                 [.2, .5, .4, .1]
                ]

  @p("data", data_values)
  def test_call_empty_cascade(self, data):
    dtest = data.copy() if isinstance(data, Stream) else data
    for el, elt in xzip(CascadeFilter()(data), dtest):
      assert el == elt

  @p("data", data_values)
  def test_call_empty_parallel(self, data):
    for el in ParallelFilter()(data):
      assert el == 0.


class TestResonator(object):

  @p("func", resonator)
  def test_zeros_and_number_of_poles(self, func):
    names = set(resonator.__name__.split("_"))
    filt = func(pi / 2, pi / 18) # Values in rad / sample
    assert isinstance(filt, ZFilter)
    assert len(filt.denominator) == 3
    num = filt.numerator
    if "z" in names:
      assert len(num) == 3
      assert num[1] == 0
      assert num[0] == -num[2]
    if "poles" in names:
      assert len(filt.numerator) == 1 # Just a constant

  @p("func", resonator)
  @p("freq", [pi / 2, pi / 3, 2 * pi / 3])
  @p("bw", [pi * k / 15 for k in xrange(1, 5)])
  def test_radius_range(self, func, freq, bw):
    filt = func(freq, bw)
    R_squared = filt.denominator[2]
    assert 0 < R_squared < 1

  @p("func", [r for r in resonator if "freq" not in r.__name__.split("_")])
  @p("freq", [pi * k / 7 for k in xrange(1, 7)])
  @p("bw", [pi / 25, pi / 30])
  def test_gain_0dB_at_given_freq(self, func, freq, bw):
    filt = func(freq, bw)
    gain = dB20(filt.freq_response(freq))
    assert almost_eq.diff(gain, 0., max_diff=5e-14)


class TestLowpassHighpass(object):

  @p("filt_func", [lowpass.pole, highpass.pole])
  @p("freq", [pi * k / 7 for k in xrange(1, 7)])
  def test_3dB_gain(self, filt_func, freq):
    filt = filt_func(freq)
    ref_gain = dB10(.5) # -3.0103 dB
    assert almost_eq.diff(dB20(filt.freq_response(freq)), ref_gain)
