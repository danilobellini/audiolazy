#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing module for the lazy_filters module

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

Created on Sun Sep 09 2012
danilo [dot] bellini [at] gmail [dot] com
"""

import pytest
p = pytest.mark.parametrize

import operator
import itertools as it

# Audiolazy internal imports
from ..lazy_filters import LTIFreq, z
from ..lazy_misc import almost_eq, almost_eq_diff


class TestLTIFreq(object):
  data = [-7, 3] + range(10) + [-50, 0] + range(70, -70, -11) # Arbitrary ints
  alpha = [-.5, -.2, -.1, 0, .1, .2, .5] # Attenuation Value for filters

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


  @p("delay", range(1, 5)) # Slice with zero would make data empty
  def test_z_int_delay(self, delay):
    my_filter = + z ** -delay
    assert list(my_filter(self.data)) == [0] * delay + self.data[:-delay]

  @p("amp_factor", [1, -105, 43, 0, .128, 18])
  @p("delay", range(1, 7)) # Slice with zero would make data empty
  def test_z_int_delay_with_amplification(self, amp_factor, delay):
    my_filter1 = amp_factor * z ** -delay
    my_filter2 = z ** -delay * amp_factor
    expected = [amp_factor * di for di in ([0.] * delay + self.data[:-delay])]
    op = operator.eq if isinstance(amp_factor, int) else almost_eq
    assert op(list(my_filter1(self.data)), expected)
    assert op(list(my_filter2(self.data)), expected)

  def test_z_fir_size_2(self):
    my_filter = 1 + z ** -1
    expected = [a + b for a, b in it.izip(self.data, [0] + self.data[:-1])]
    assert list(my_filter(self.data)) == expected

  def test_z_fir_size_2_hybrid_amplification(self):
    my_filter = 2 * (3. - 5 * z ** -1)
    expected = (6.*a - 10*b for a, b in it.izip(self.data,
                                                [0.] + self.data[:-1]))
    assert almost_eq(my_filter(self.data), expected)

  delays = range(1, 5)
  amp_list = [1, -15, 45, 0, .81, 17]
  @p(("num_delays", "amp_factor"),
     [(delay, amp) for delay in delays
                   for amp in it.combinations_with_replacement(amp_list,
                                                               delay + 1)]
    )
  def test_z_many_fir_sizes_and_amplifications(self, num_delays, amp_factor):
    my_filter = sum(amp_factor[delay] * z ** -delay
                    for delay in xrange(num_delays + 1))
    expected = sum(amp_factor[delay] * (z ** -delay)(self.data)
                   for delay in xrange(num_delays + 1))
    assert almost_eq(my_filter(self.data), expected)

  def test_z_fir_multiplication(self):
    my_filter = 8 * (2 * z**-3 - 5 * z**-4) * z ** 2 * 7
    expected = [56*2*a - 56*5*b for a, b in zip([0] + self.data[:-1],
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
  @p("idx_num1", range(-3,1))
  @p("idx_den1", range(-1,3,19))
  @p("idx_num2", range(-2,0,14))
  @p("idx_den2", range(-18,1))
  def test_z_division(self, a, b, idx_num1, idx_den1, idx_num2, idx_den2):
    fa, fb, fc, fd = (a * z ** idx_num1, 2 + b * z ** idx_den1,
                      3 * z ** idx_num2, 1 + 5 * z ** idx_den2)
    my_filter1 = fa / fb
    my_filter2 = fc / fd
    my_filter = my_filter1 / my_filter2
    idx_corr = max(idx_num2, idx_num2 + idx_den1)
    num_filter = fa * fd * (z ** -idx_corr)
    den_filter = fb * fc * (z ** -idx_corr)
    assert my_filter.numpoly == num_filter.numpoly
    assert my_filter.denpoly == den_filter.numpoly

  @p("filt", [1 / z / 1,
              (1 / z) ** 1,
              (1 / z ** -1) ** -1,
             ])
  def test_z_power_alone(self, filt):
    assert almost_eq(filt(self.data), [0.] + self.data[:-1])

  @p("a", [a for a in alpha if a != 0])
  def test_z_div_truediv_delay_over_constant(self, a):
    div_filter = operator.div(z ** -1, a)
    truediv_filter = operator.truediv(z ** -1, a)
    div_expected = it.imap(lambda x: operator.div(x, a),
                           [0.] + self.data[:-1])
    truediv_expected = it.imap(lambda x: operator.truediv(x, a),
                               [0.] + self.data[:-1])
    assert almost_eq(div_filter(self.data), div_expected)
    assert almost_eq(truediv_filter(self.data), truediv_expected)

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
    my_filter = LTIFreq(numerator=[1.], denominator=[1., -a])
    expected = [x for x in self.data]
    for idx in xrange(1,len(expected)):
      expected[idx] += a * expected[idx-1]
    assert almost_eq(list(my_filter(self.data)), expected)

  @p("delay", range(1, 7))
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
    muld = mul.diff() if isinstance(mul, LTIFreq) else 0
    assert num.diff() == numd
    assert den.diff() == dend
    assert num.diff(mul_after=mul) == numd * mul
    assert den.diff(mul_after=mul) == dend * mul
    filtd_num = numd * den - dend * num
    filtd = filtd_num / den ** 2
    assert filt.diff() == filtd
    assert filt.diff(mul_after=mul) == filtd * mul
    numd2 = -2 * A * (1 - a) * z ** -3
    numd2ma = (numd2 * mul + muld * numd) * mul
    dend2 = -2 * A * z ** -3 + 6 * A ** 2 * z ** -4
    dend2ma = (dend2 * mul + muld * dend) * mul
    assert num.diff(2) == numd2
    assert den.diff(2) == dend2
    assert num.diff(n=2, mul_after=mul) == numd2ma
    assert den.diff(n=2, mul_after=mul) == dend2ma

    filtd2 = ((numd2 * den - num * dend2) * den - 2 * filtd_num * dend
             ) / den ** 3
    filt_to_test = filt.diff(n=2)
    assert almost_eq_diff(filt_to_test.numerator, filtd2.numerator,
                          max_diff=1e-10)
    assert almost_eq_diff(filt_to_test.denominator, filtd2.denominator,
                          max_diff=1e-10)

    if 1/(1 + z**-2) != mul: # Too difficult to group together with others
      filtd2ma = ((numd2 * den - num * dend2) * mul * den +
                  filtd_num * (muld * den - 2 * mul * dend)
                 ) * mul / den ** 3
      filt_to_testma = filt.diff(n=2, mul_after=mul)
      assert almost_eq_diff(filt_to_testma.numerator, filtd2ma.numerator,
                            max_diff=1e-10)
      assert almost_eq_diff(filt_to_testma.denominator, filtd2ma.denominator,
                            max_diff=1e-10)
