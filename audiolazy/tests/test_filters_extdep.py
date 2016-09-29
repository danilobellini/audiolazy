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
Testing module for the lazy_filters module by using Numpy/Scipy/Sympy
"""

import pytest
p = pytest.mark.parametrize

from scipy.signal import lfilter
from scipy.optimize import fminbound
from math import cos, pi, sqrt
from numpy import mat
from sympy import symbols, Matrix, sqrt as symb_sqrt
import sympy

# Audiolazy internal imports
from ..lazy_filters import ZFilter, resonator, z, lowpass, highpass
from ..lazy_misc import almost_eq, elementwise
from ..lazy_compat import orange, xrange, xzip, xmap
from ..lazy_math import dB20
from ..lazy_itertools import repeat, cycle, count
from ..lazy_stream import Stream, thub


class TestZFilterNumpyScipySympy(object):

  @p("a", [[1.], [3.], [1., 3.], [15., -17.2], [-18., 9.8, 0., 14.3]])
  @p("b", [[1.], [-1.], [1., 0., -1.], [1., 3.]])
  @p("data", [orange(5), orange(5, 0, -1), [7, 22, -5], [8., 3., 15.]])
  def test_lfilter(self, a, b, data):
    filt = ZFilter(b, a)
    expected = lfilter(b, a, data).tolist()
    assert almost_eq(filt(data), expected)

  def test_matrix_coefficients_multiplication(self):
    m = mat([[1, 2], [2, 2]])
    n1 = mat([[1.2, 3.2], [1.2, 1.1]])
    n2 = mat([[-1, 2], [-1, 2]])
    a = mat([[.3, .4], [.5, .6]])

    # Time-varying filter with 2x2 matrices as coeffs
    mc = repeat(m) # "Constant" coeff
    nc = cycle([n1, n2]) # Periodic coeff
    ac = repeat(a)
    filt = (mc + nc * z ** -1) / (1 - ac * z ** -1)

    # For a time-varying 2x3 matrix signal
    data = [
      Stream(1, 2),
      count(),
      count(start=1, step=2),
      cycle([.2, .33, .77, pi, cos(3)]),
      repeat(pi),
      count(start=sqrt(2), step=pi/3),
    ]

    # Copy just for a future verification of the result
    data_copy = [el.copy() for el in data]

    # Build the 2x3 matrix input signal and find the result!
    sig = Stream(mat(vect).reshape(2, 3) for vect in xzip(*data))
    zero = mat([[0, 0, 0], [0, 0, 0]])
    result = filt(sig, zero=zero).limit(30)

    # Result found, let's just find if they're what they should be
    in_sample = old_expected_out_sample = zero
    n, not_n = n1, n2
    for out_sample in result:
      old_in_sample = in_sample
      in_sample = mat([s.take() for s in data_copy]).reshape(2, 3)
      expected_out_sample = m * in_sample + n * old_in_sample \
                                          + a * old_expected_out_sample
      assert almost_eq(out_sample.tolist(), expected_out_sample.tolist())
      n, not_n = not_n, n
      old_expected_out_sample = expected_out_sample

  def test_symbolic_signal_and_coeffs(self):
    symbol_tuple = symbols("a b c d")
    a, b, c, d = symbol_tuple
    ac, bc, cc = xmap(repeat, symbol_tuple[:-1]) # Coeffs (Stream instances)
    zero = sympy.S.Zero # Symbolic zero

    filt = (ac * z ** -1 + bc * z ** -2) / (1 + cc * z ** -2)
    sig = Stream([d] * 5, repeat(zero))
    result = filt(sig, zero=zero)

    # Let's find if that worked:
    expected = [ # out[n] = a * in[n - 1] + b * in[n - 2] - c * out[n - 2]
      zero,                      # input: d
      a*d,                       # input: d
      a*d + b*d,                 # input: d
      a*d + b*d - c*a*d,         # input: d
      a*d + b*d - c*(a*d + b*d), # input: d (last non-zero input)
      a*d + b*d - c*(a*d + b*d - c*a*d),
      b*d - c*(a*d + b*d - c*(a*d + b*d)),
      -c*(a*d + b*d - c*(a*d + b*d - c*a*d)),
    ]
    for unused in range(50): # Create some more samples
      expected.append(-c * expected[-2])
    size = len(expected)
    assert result.peek(size) == expected

    # Let's try again, now defining c as zero (with sub method from sympy)
    another_result = result.subs(c, zero).take(size)
    another_expected = [zero] * size
    for idx in xrange(5):
      another_expected[idx+1] += a*d
      another_expected[idx+2] += b*d
    assert another_result == another_expected

  def test_symbolic_matrices_sig_and_coeffs_state_space_filters(self):
    k = symbols("k:31") # Symbols k[0], k[1], k[2], ..., k[30]

    # Matrices for coeffs
    am = [                  # Internal state delta matrix
      Matrix([[k[0], k[1]], # (as a periodic sequence)
              [k[2], k[3]]]),
      Matrix([[k[1], k[2]], [k[3], k[0]]]),
      Matrix([[k[2], k[3]], [k[0], k[1]]]),
      Matrix([[k[3], k[0]], [k[1], k[2]]]),
    ]
    bm = Matrix([[k[4], k[5], k[6],  k[7]], # Input to internal state matrix
                 [k[8], k[9], k[10], k[11]]])
    cm = Matrix([[k[12], k[13]], # Internal state to output matrix
                 [k[14], k[15]],
                 [k[16], k[17]]])
    dm = Matrix([[k[18], k[19], k[20], k[21]], # Input to output matrix
                 [k[22], k[23], k[24], k[25]],
                 [k[26], k[27], k[28], k[29]]])

    # Zero is needed, not only the symbol but the matrices
    zero = sympy.S.Zero
    zero_state = Matrix([[zero]] * 2)
    zero_input = Matrix([[zero]] * 4)
    zero_kwargs = dict(zero=zero_input, memory=repeat(zero_state))

    # The state filter itself
    #   x[n] = A[n] * x[n-1] + B[n] * u[n-1]
    #   y[n] = C[n] * x[n]   + D[n] * u[n]
    ac = cycle(am)
    bc, cc, dc = xmap(repeat, [bm, cm, dm])
    xfilt = (bc * z ** -1) / (1 - ac * z ** -1)
    def filt(data):
      data = thub(data, 2)
      return cc * xfilt(data, **zero_kwargs) + dc * data

    # Data to be used as input: u[n] = k[30] / (n + r),
    # where n >= 0 and r is the row, 1 to 4
    sigs = [repeat(k[30]) / count(start=r, step=1) for r in [1, 2, 3, 4]]
    u = Stream(Matrix(vect) for vect in xzip(*sigs))
    result = filt(u)

    # Result verification
    u_list = [
      Matrix([[k[30] / el] for el in [i, i+1, i+2, i+3]])
      for i in range(1, 50)
    ]
    expected = [dm * u_list[0]]
    internal = [bm * u_list[0]]
    for idx, ui in enumerate(u_list[1:], 2):
      expected.append(cm * internal[-1] + dm * ui)
      internal.append(am[idx % 4] * internal[-1] + bm * ui)
    size = len(expected)
    assertion = result.peek(size) # Goes faster with py.test (don't generate
    assert assertion              # "text"/string representations)

  def test_symbolic_fixed_matrices_sig_and_coeffs_state_space_filters(self):
    k = symbols("k")
    size = 100
    s2, s3, s6 = xmap(symb_sqrt, [2, 3, 6])

    # Same idea from the test above, but with numbers as symbolic coeffs
    uf = [Matrix([k / el for el in [i, i+1, i+2, i+3]])
          for i in range(1, size+1)]
    af = [
      Matrix([[1, 0], [0, 0]]),
      Matrix([[0, 0], [0, 1]]),
      Matrix([[0, 0], [1, 0]]),
      Matrix([[0, 1], [0, 0]]),
    ]
    bf = Matrix([[1, 1, 1, 1], [-1, -1, -1, -1]])
    cf = Matrix([[s3, 0], [0, s3], [0, 0]])
    df = Matrix([[s2, 0, 0, 0], [0, s2, 0, 0], [0, 0, s2, 0]])

    # Zero is needed, not only the symbol but the matrices
    zero = sympy.S.Zero
    zero_state = Matrix([[zero]] * 2)
    zero_input = Matrix([[zero]] * 4)
    zero_kwargs = dict(zero=zero_input, memory=repeat(zero_state))

    # Calculates the whole symbolical sequence (fixed numbers)
    afc = cycle(af)
    bfc, cfc, dfc = xmap(repeat, [bf, cf, df])
    xffilt = (bfc * z ** -1) / (1 - afc * z ** -1)
    def ffilt(data):
      data = thub(data, 2)
      return cfc * xffilt(data, **zero_kwargs) + dfc * data
    result_f = ffilt(Stream(uf)).subs(k, s6).take(size)

    expected_f = [df * uf[0].subs(k, s6)]
    internal_f = [bf * uf[0].subs(k, s6)]
    for idx, ufi in enumerate(uf[1:], 2):
      ufis = ufi.subs(k, s6)
      expected_f.append(cf * internal_f[-1] + df * ufis)
      internal_f.append(af[idx % 4] * internal_f[-1] + bf * ufis)
    assert result_f == expected_f

    # Numerical!
    to_numpy = lambda mlist: [mat(m.applyfunc(float).tolist()) for m in mlist]
    un = to_numpy(ui.subs(k, s6) for ui in uf)
    an = to_numpy(af)
    bn, cn, dn = to_numpy([bf, cf, df])

    expected_n = [dn * un[0]]
    internal_n = [bn * un[0]]
    for idx, uni in enumerate(un[1:], 2):
      expected_n.append(cn * internal_n[-1] + dn * uni)
      internal_n.append(an[idx % 4] * internal_n[-1] + bn * uni)
    result_n = [el.tolist() for el in to_numpy(result_f)]
    assert almost_eq(result_n, [m.tolist() for m in expected_n])


class TestResonatorScipy(object):

  @p("func", resonator)
  @p("freq", [pi * k / 9 for k in xrange(1, 9)])
  @p("bw", [pi / 23, pi / 31])
  def test_max_gain_is_at_resonance(self, func, freq, bw):
    names = func.__name__.split("_")
    filt = func(freq, bw)
    resonance_freq = fminbound(lambda x: -dB20(filt.freq_response(x)),
                               0, pi, xtol=1e-10)
    resonance_gain = dB20(filt.freq_response(resonance_freq))
    assert almost_eq.diff(resonance_gain, 0., max_diff=1e-12)

    if "freq" in names: # Given frequency is at the denominator
      R = sqrt(filt.denominator[2])
      assert 0 < R < 1
      cosf = cos(freq)
      cost = -filt.denominator[1] / (2 * R)
      assert almost_eq(cosf, cost)

      if "z" in names:
        cosw = cosf * (2 * R) / (1 + R ** 2)
      elif "poles" in names:
        cosw = cosf * (1 + R ** 2) / (2 * R)

      assert almost_eq(cosw, cos(resonance_freq))

    else: # Given frequency is the resonance frequency
      assert almost_eq(freq, resonance_freq)


class TestLowpassHighpassSympy(object):

  @p("filt_func", [lowpass.z, highpass.z])
  def test_single_zero_strategies_zeroed_R_denominator_lti(self, filt_func,
                                                           monkeypatch):
    from .. import lazy_filters
    monkeypatch.setattr(lazy_filters, "sin", elementwise("x", 0)(sympy.sin))
    monkeypatch.setattr(lazy_filters, "cos", elementwise("x", 0)(sympy.cos))
    filt = filt_func(sympy.pi / 2)
    assert filt.denominator == [1] # R is zero
    assert list(xmap(abs, filt.numerator)) == [sympy.S.Half] * 2

  @p("filt_func", [lowpass.z, highpass.z])
  def test_single_zero_strategies_zeroed_R_denominator_tvar(self, filt_func,
                                                           monkeypatch):
    from .. import lazy_filters
    monkeypatch.setattr(lazy_filters, "sin", elementwise("x", 0)(sympy.sin))
    monkeypatch.setattr(lazy_filters, "cos", elementwise("x", 0)(sympy.cos))
    filt = filt_func(repeat(sympy.pi / 2))
    assert filt.denpoly[0] == 1
    pole_sig = filt.denpoly[1]
    num_sig0, num_sig1 = filt.numlist
    n = 3 # Amount of coefficient samples to get
    assert pole_sig.take(n) == [0] * n
    assert num_sig0.take(n) == [sympy.S.Half] * n
    assert abs(num_sig0).take(n) == [sympy.S.Half] * n

  @p(("filt_func", "fsign"), [(lowpass.z, 1), (highpass.z, -1)])
  def test_single_zero_strategies_2pi3_lti(self, filt_func, fsign,
                                                 monkeypatch):
    from .. import lazy_filters
    monkeypatch.setattr(lazy_filters, "sin", elementwise("x", 0)(sympy.sin))
    monkeypatch.setattr(lazy_filters, "cos", elementwise("x", 0)(sympy.cos))

    R = (2 - sympy.sqrt(3)) * fsign
    G = (R + 1) / 2

    filt = filt_func(2 * sympy.pi / 3)
    assert filt.numerator == [G, fsign * G]
    assert filt.denominator == [1, fsign * R]

  @p(("filt_func", "fsign"), [(lowpass.z, 1), (highpass.z, -1)])
  def test_single_zero_strategies_tvar(self, filt_func, fsign,
                                             monkeypatch):
    from .. import lazy_filters
    monkeypatch.setattr(lazy_filters, "sin", elementwise("x", 0)(sympy.sin))
    monkeypatch.setattr(lazy_filters, "cos", elementwise("x", 0)(sympy.cos))

    start = 2
    n = 3 # Amount of coefficient samples to get
    freqs = repeat(sympy.pi) / count(start=start)

    filt = filt_func(freqs)
    assert filt.denpoly[0] == 1

    t = sympy.Symbol("t")
    Rs = [sympy.limit(fsign * (sympy.sin(t) - 1) / sympy.cos(t),
                      t, sympy.pi / el)
          for el in xrange(2, 2 + n)]
    Gs = [(R + 1) / 2 for R in Rs]

    pole_sig = filt.denpoly[1]
    assert (fsign * pole_sig).cancel().take(n) == Rs
    assert len(filt.denpoly) == 2

    num_sig0, num_sig1 = filt.numlist
    assert num_sig0.cancel().take(n) == Gs
    assert (fsign * num_sig1).cancel().take(n) == Gs
