# -*- coding: utf-8 -*-
"""
Stream filtering module

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

Created on Wed Jul 18 2012
danilo [dot] bellini [at] gmail [dot] com
"""

import operator
from cmath import exp as complex_exp
from collections import Iterable, OrderedDict
import itertools as it

# Audiolazy internal imports
from .lazy_stream import Stream, avoid_stream, thub
from .lazy_misc import (elementwise, zero_pad, multiplication_formatter,
                        pair_strings_sum_formatter)
from .lazy_poly import Poly
from .lazy_core import AbstractOperatorOverloaderMeta, StrategyDict
from .lazy_math import exp, sin, cos, sqrt, pi, nan

__all__ = ["LinearFilterProperties", "LinearFilter", "ZFilterMeta", "ZFilter",
           "z", "CascadeFilterMeta", "CascadeFilter", "comb", "resonator",
           "lowpass", "highpass"]


class LinearFilterProperties(object):
  """
  Class with common properties in a linear filter.
  Needs only numpoly and denpoly attributes.
  This class can be used as a mixin.
  """
  def numlist(self):
    if any(key < 0 for key, value in self.numpoly.terms()):
      raise ValueError("Non-causal filter")
    return list(self.numpoly.values())
  numerator = property(numlist)
  numlist = property(numlist)

  def denlist(self):
    if any(key < 0 for key, value in self.denpoly.terms()):
      raise ValueError("Non-causal filter")
    return list(self.denpoly.values())
  denominator = property(denlist)
  denlist = property(denlist)

  @property
  def numdict(self):
    return OrderedDict(self.numpoly.terms())

  @property
  def dendict(self):
    return OrderedDict(self.denpoly.terms())


@avoid_stream
class LinearFilter(LinearFilterProperties):
  """
  Base class for Linear filters, time invariant or not.
  """
  def __init__(self, numerator=None, denominator={0: 1}):
    self.numpoly = Poly(numerator)
    self.denpoly = Poly(denominator)

    # Ensure denominator has only negative powers of z (positive powers here),
    # and a not null gain constant
    power = min(key for key, value in self.denpoly.terms())
    if power != 0:
      poly_delta = Poly([0, 1]) ** -power
      self.numpoly *= poly_delta
      self.denpoly *= poly_delta

  def __iter__(self):
    yield self.numdict
    yield self.dendict

  def __call__(self, seq, memory=None, zero=0.):
    """
    IIR and FIR linear filtering.

    Parameters
    ----------
    seq :
      Any iterable to be seem as the input stream for the filter.
    memory :
      Might be an iterable or a callable. Generally, as a iterable, the first
      needed elements from this input will be used directly as the memory
      (not the last ones!), and as a callable, it will be called with the
      size as the only positional argument, and should return an iterable.
      If ``None`` (default), memory is initialized with zeros.
    zero :
      Value to fill the memory, when needed, and to be seem as previous
      input when there's a delay. Defaults to ``0.0``.

    Returns
    -------
    A Stream that have the data from the input sequence filtered.

    """
    # Data check
    if any(key < 0 for key, value in it.chain(self.numpoly.terms(),
                                              self.denpoly.terms())
          ):
      raise ValueError("Non-causal filter")
    if self.denpoly[0] == 0:
      raise ZeroDivisionError("Invalid filter gain")

    # Lengths
    la, lb = len(self.denominator), len(self.numerator)
    lm = la - 1 # Memory size

    # Convert memory input to a list with size exactly equals to lm
    if memory is None:
      memory = [zero for unused in xrange(lm)]
    else: # Get data from iterable
      if not isinstance(memory, Iterable): # Function with 1 parameter: size
        memory = memory(lm)
      tw = it.takewhile(lambda (idx, data): idx < lm,
                        enumerate(memory))
      memory = [data for idx, data in tw]
      actual_len = len(memory)
      if actual_len < lm:
        memory = list(zero_pad(memory, lm - actual_len, zero=zero))

    # Creates the expression in a string
    data_sum = []

    num_iterables = []
    for delay, coeff in self.numdict.iteritems():
      if isinstance(coeff, Iterable):
        num_iterables.append(delay)
        data_sum.append("d{idx} * b{idx}.next()".format(idx=delay))
      elif coeff == 1:
        data_sum.append("d{idx}".format(idx=delay))
      elif coeff == -1:
        data_sum.append("-d{idx}".format(idx=delay))
      elif coeff != 0:
        data_sum.append("d{idx} * {value}".format(idx=delay, value=coeff))

    den_iterables = []
    for delay, coeff in self.dendict.iteritems():
      if isinstance(coeff, Iterable):
        den_iterables.append(delay)
        data_sum.append("-m{idx} * a{idx}.next()".format(idx=delay))
      elif delay == 0:
        gain = coeff
      elif coeff == -1:
        data_sum.append("m{idx}".format(idx=delay))
      elif coeff == 1:
        data_sum.append("-m{idx}".format(idx=delay))
      elif coeff != 0:
        data_sum.append("-m{idx} * {value}".format(idx=delay, value=coeff))

    # Creates the generator function for this call
    if len(data_sum) == 0:
      gen_func =  ["def gen(seq):",
                   "  for unused in seq:",
                   "    yield {zero}".format(zero=zero)
                  ]
    else:
      expr = " + ".join(data_sum)
      if gain == -1:
        expr = "-({expr})".format(expr=expr)
      elif gain != 1:
        expr = "({expr}) / {gain}".format(expr=expr, gain=gain)

      arg_names = ["seq"]
      arg_names.extend("b{idx}".format(idx=idx) for idx in num_iterables)
      arg_names.extend("a{idx}".format(idx=idx) for idx in den_iterables)
      gen_func =  ["def gen({args}):".format(args=", ".join(arg_names))]
      gen_func += ["  m{idx} = {value}".format(idx=idx, value=value)
                   for idx, value in enumerate(memory, 1)]
      gen_func += ["  d{idx} = {value}".format(idx=idx, value=zero)
                   for idx in xrange(1, lb)]
      gen_func += ["  for d0 in seq:",
                   "    m0 = {expr}".format(expr=expr),
                   "    yield m0"]
      gen_func += ["    m{idx} = m{idxold}".format(idx=idx, idxold=idx - 1)
                   for idx in xrange(lm, 0, -1)]
      gen_func += ["    d{idx} = d{idxold}".format(idx=idx, idxold=idx - 1)
                   for idx in xrange(lb - 1, 0, -1)]

    # Uses the generator function to return the desired values
    ns = {}
    exec "\n".join(gen_func) in ns
    arguments = [iter(seq)]
    arguments.extend(iter(self.numpoly[idx]) for idx in num_iterables)
    arguments.extend(iter(self.denpoly[idx]) for idx in den_iterables)
    return Stream(ns["gen"](*arguments))


  @elementwise("freq", 1)
  def freq_response(self, freq):
    """
    Frequency response for this filter.

    Parameters
    ----------
    freq :
      Frequency, in rad/sample. Can be an iterable with frequencies.

    Returns
    -------
    Complex number with the frequency response of the filter.

    See Also
    --------
    dB10 :
      Logarithmic power magnitude from data with squared magnitude.
    dB20 :
      Logarithmic power magnitude from raw complex data or data with linear
      amplitude.
    phase :
      Phase from complex data.

    """
    z_ = complex_exp(-1j * freq)
    num = self.numpoly(z_)
    den = self.denpoly(z_)
    if not isinstance(den, Stream):
      if den == 0:
        return nan
    return num / den

  def is_lti(self):
    """
    Test if this filter is LTI (Linear Time Invariant).

    Returns
    -------
    Boolean returning True if this filter is LTI, False otherwise.

    """
    return not any(isinstance(value, Iterable)
                   for delay, value in it.chain(self.numpoly.terms(),
                                                self.denpoly.terms()))

  def is_causal(self):
    """
    Causality test for this filter.

    Returns
    -------
    Boolean returning True if this filter is causal, False otherwise.

    """
    return all(delay >= 0 for delay, value in self.numpoly.terms())

  def linearize(self):
    """
    Linear interpolation of fractional delay values.

    Returns
    -------
    A new linear filter, with the linearized delay values.

    Examples
    --------

    >>> filt = z ** -4.3
    >>> filt.linearize()
    0.7 * z^-4 + 0.3 * z^-5

    """
    data = []
    for poly in [self.numpoly, self.denpoly]:
      data.append({})
      new_poly = data[-1]
      for k, v in poly.terms():
        if isinstance(k, int) or (isinstance(k, float) and k.is_integer()):
          pairs = [(int(k), v)]
        else:
          left = int(k)
          right = left + 1
          weight_right = k - left
          weight_left = 1. - weight_right
          pairs = [(left, v * weight_left), (right, v * weight_right)]
        for key, value in pairs:
          if key in new_poly:
            new_poly[key] += value
          else:
            new_poly[key] = value
    return self.__class__(*data)


class ZFilterMeta(AbstractOperatorOverloaderMeta):
  __operators__ = ("pos neg add radd sub rsub mul rmul div rdiv "
                   "truediv rtruediv pow "
                   "eq ne " # almost_eq comparison of Poly terms
                  )

  def __rbinary__(cls, op_func):
    def dunder(self, other):
      if isinstance(other, LinearFilter):
        raise ValueError("Filter equations have different domains")
      return op_func(cls([other]), self) # The "other" is probably a number
    return dunder

  def __unary__(cls, op_func):
    def dunder(self):
      return cls(op_func(self.numpoly), self.denpoly)
    return dunder


@avoid_stream
class ZFilter(LinearFilter):
  """
  Linear filters based on Z-transform frequency domain equations.

  Examples
  --------

  Using the ``z`` object (float output because default filter memory has
  float zeros, and the delay in the numerator creates another float zero as
  "pre-input"):

  >>> filt = (1 + z ** -1) / (1 - z ** -1)
  >>> data = [1, 5, -4, -7, 9]
  >>> stream_result = filt(data) # Lazy iterable
  >>> list(stream_result) # Freeze
  [1.0, 7.0, 8.0, -3.0, -1.0]

  Same example with the same filter, but with a memory input, and using
  lists for filter numerator and denominator instead of the ``z`` object:

  >>> b = [1, 1]
  >>> a = [1, -1] # Each index ``i`` has the coefficient for z ** -i
  >>> filt = ZFilter(b, a)
  >>> data = [1, 5, -4, -7, 9]
  >>> stream_result = filt(data, memory=[3], zero=0) # Lazy iterable
  >>> result = list(stream_result) # Freeze
  >>> result
  [4, 10, 11, 0, 2]
  >>> filt2 = filt * z ** -1 # You can add a delay afterwards easily
  >>> final_result = filt2(result, zero=0)
  >>> list(final_result)
  [0, 4, 18, 39, 50]

  """
  __metaclass__ = ZFilterMeta

  def __add__(self, other):
    if isinstance(other, ZFilter):
      if self.denpoly == other.denpoly:
        return ZFilter(self.numpoly + other.numpoly, self.denpoly)
      return ZFilter(self.numpoly * other.denpoly +
                     other.numpoly * self.denpoly,
                     self.denpoly * other.denpoly)
    if isinstance(other, LinearFilter):
      raise ValueError("Filter equations have different domains")
    return self + ZFilter([other]) # Other is probably a number

  def __sub__(self, other):
    return self + (-other)

  def __mul__(self, other):
    if isinstance(other, ZFilter):
      return ZFilter(self.numpoly * other.numpoly,
                     self.denpoly * other.denpoly)
    if isinstance(other, LinearFilter):
      raise ValueError("Filter equations have different domains")
    return ZFilter(self.numpoly * other, self.denpoly)

  def __div__(self, other):
    if isinstance(other, ZFilter):
      return ZFilter(self.numpoly * other.denpoly,
                     self.denpoly * other.numpoly)
    if isinstance(other, LinearFilter):
      raise ValueError("Filter equations have different domains")
    return self * operator.div(1, other)

  def __truediv__(self, other):
    if isinstance(other, ZFilter):
      return ZFilter(self.numpoly * other.denpoly,
                     self.denpoly * other.numpoly)
    if isinstance(other, LinearFilter):
      raise ValueError("Filter equations have different domains")
    return self * operator.truediv(1, other)

  def __pow__(self, other):
    if (other < 0) and (len(self.numpoly) >= 2 or len(self.denpoly) >= 2):
      return ZFilter(self.denpoly, self.numpoly) ** -other
    if isinstance(other, (int, float)):
      return ZFilter(self.numpoly ** other, self.denpoly ** other)
    raise ValueError("Z-transform powers only valid with integers")

  def __eq__(self, other):
    if isinstance(other, LinearFilter):
      return self.numpoly == other.numpoly and self.denpoly == other.denpoly
    return False

  def __ne__(self, other):
    if isinstance(other, LinearFilter):
      return self.numpoly != other.numpoly and self.denpoly != other.denpoly
    return False

  def __str__(self):
    num_term_strings = []
    for power, value in self.numpoly.terms():
      if isinstance(value, Iterable):
        value = "b{0}".format(power)
      if value != 0.:
        num_term_strings.append(multiplication_formatter(-power, value, "z"))
    num = "0" if len(num_term_strings) == 0 else \
          reduce(pair_strings_sum_formatter, num_term_strings)

    den_term_strings = []
    for power, value in self.denpoly.terms():
      if isinstance(value, Iterable):
        value = "a{0}".format(power)
      if value != 0.:
        den_term_strings.append(multiplication_formatter(-power, value, "z"))
    den = reduce(pair_strings_sum_formatter, den_term_strings)

    if den == "1": # No feedback
      return num

    line = "-" * max(len(num), len(den))
    spacer_offset = abs(len(num) - len(den)) // 2
    if spacer_offset > 0:
      centralize_spacer = " " * spacer_offset
      if len(num) > len(den):
        den = centralize_spacer + den
      else: # len(den) > len(num)
        num = centralize_spacer + num

    breaks = len(line) // 80
    slices = [slice(b * 80,(b + 1) * 80) for b in range(breaks + 1)]
    outputs = ["\n".join([num[s], line[s], den[s]]) for s in slices]
    return "\n\n    ...continue...\n\n".join(outputs)

  __repr__ = __str__

  def diff(self, n=1, mul_after=1):
    """
    Takes n-th derivative, multiplying each m-th derivative filter by
    mul_after before taking next (m+1)-th derivative or returning.
    """
    if isinstance(mul_after, ZFilter):
      den = ZFilter(self.denpoly)
      return reduce(lambda num, order: mul_after *
                      (num.diff() * den - order * num * den.diff()),
                    xrange(1, n + 1),
                    ZFilter(self.numpoly)
                   ) / den ** (n + 1)

    inv_sign = Poly({-1: 1}) # Since poly variable is z ** -1
    den = self.denpoly(inv_sign)
    return ZFilter(reduce(lambda num, order: mul_after *
                            (num.diff() * den - order * num * den.diff()),
                          xrange(1, n + 1),
                          self.numpoly(inv_sign))(inv_sign),
                   self.denpoly ** (n + 1))


z = ZFilter({-1: 1})


class CascadeFilterMeta(AbstractOperatorOverloaderMeta):
  __operators__ = ("add mul rmul lt le eq ne gt ge")

  def __binary__(cls, op_func):
    def dunder(self, other):
      return cls(getattr(super(cls, self), dunder.__name__)(other))
    return dunder

  __rbinary__ = __binary__


@avoid_stream
class CascadeFilter(list, LinearFilterProperties):
  """
  Filter cascade as a list of filters.
  A filter is any callable that receives an iterable as input and returns a
  Stream.
  """
  __metaclass__ = CascadeFilterMeta

  def __init__(self, *filters):
    if len(filters) == 1 and isinstance(filters[0], Iterable) \
                         and not isinstance(filters[0], LinearFilter):
      self.extend(filters[0])
    else:
      self.extend(filters)

  def __call__(self, seq):
    return reduce(lambda data, filt: filt(data), self, seq)

  @property
  def numpoly(self):
    try:
      return reduce(operator.mul, (filt.numpoly for filt in self))
    except AttributeError:
      raise AttributeError("Non-linear filter")

  @property
  def denpoly(self):
    try:
      return reduce(operator.mul, (filt.denpoly for filt in self))
    except AttributeError:
      raise AttributeError("Non-linear filter")

  @elementwise("freq", 1)
  def freq_response(self, freq):
    return reduce(operator.mul, (filt.freq_response(freq) for filt in self))

  def is_linear(self):
    return all(isinstance(filt, LinearFilter) for filt in self)


def comb(delay, alpha=1):
  return 1 / (1 - alpha * z ** -delay)


resonator = StrategyDict("resonator")


@resonator.strategy("poles_exp")
def resonator(freq, bandwidth):
  """
  Resonator filter with 2-poles (conjugated pair) and no zeros (constant
  numerator), with exponential approximation for bandwidth calculation.

  Parameters
  ----------
  freq :
    Resonant frequency in rad/sample (max gain).
  bandwidth :
    Bandwidth frequency range in rad/sample following the equation:

      ``R = exp(-bandwidth / 2)``

    where R is the pole amplitude (radius).

  Returns
  -------
  A ZFilter object.
  Gain is normalized to have peak with 0 dB (1.0 amplitude).

  """
  bandwidth = thub(bandwidth, 1)
  R = exp(-bandwidth * .5)
  R = thub(R, 5)
  cost = cos(freq) * (2 * R) / (1 + R ** 2)
  cost = thub(cost, 2)
  gain = (1 - R ** 2) * sqrt(1 - cost ** 2)
  denominator = 1 - 2 * R * cost * z ** -1 + R ** 2 * z ** -2
  return gain / denominator


@resonator.strategy("freq_poles_exp")
def resonator(freq, bandwidth):
  """
  Resonator filter with 2-poles (conjugated pair) and no zeros (constant
  numerator), with exponential approximation for bandwidth calculation.
  Given frequency is the denominator frequency, not the resonant frequency.

  Parameters
  ----------
  freq :
    Denominator frequency in rad/sample (not the one with max gain).
  bandwidth :
    Bandwidth frequency range in rad/sample following the equation:

      ``R = exp(-bandwidth / 2)``

    where R is the pole amplitude (radius).

  Returns
  -------
  A ZFilter object.
  Gain is normalized to have peak with 0 dB (1.0 amplitude).

  """
  bandwidth = thub(bandwidth, 1)
  R = exp(-bandwidth * .5)
  R = thub(R, 3)
  freq = thub(freq, 2)
  gain = (1 - R ** 2) * sin(freq)
  denominator = 1 - 2 * R * cos(freq) * z ** -1 + R ** 2 * z ** -2
  return gain / denominator


@resonator.strategy("z_exp")
def resonator(freq, bandwidth):
  """
  Resonator filter with 2-zeros and 2-poles (conjugated pair). The zeros are
  at the `1` and `-1` (both at the real axis, i.e., at the DC and the Nyquist
  rate), with exponential approximation for bandwidth calculation.

  Parameters
  ----------
  freq :
    Resonant frequency in rad/sample (max gain).
  bandwidth :
    Bandwidth frequency range in rad/sample following the equation:

      ``R = exp(-bandwidth / 2)``

    where R is the pole amplitude (radius).

  Returns
  -------
  A ZFilter object.
  Gain is normalized to have peak with 0 dB (1.0 amplitude).

  """
  bandwidth = thub(bandwidth, 1)
  R = exp(-bandwidth * .5)
  R = thub(R, 5)
  cost = cos(freq) * (1 + R ** 2) / (2 * R)
  gain = (1 - R ** 2) * .5
  numerator = 1 - z ** -2
  denominator = 1 - 2 * R * cost * z ** -1 + R ** 2 * z ** -2
  return gain * numerator / denominator


@resonator.strategy("freq_z_exp")
def resonator(freq, bandwidth):
  """
  Resonator filter with 2-zeros and 2-poles (conjugated pair). The zeros are
  at the `1` and `-1` (both at the real axis, i.e., at the DC and the Nyquist
  rate), with exponential approximation for bandwidth calculation.
  Given frequency is the denominator frequency, not the resonant frequency.

  Parameters
  ----------
  freq :
    Denominator frequency in rad/sample (not the one with max gain).
  bandwidth :
    Bandwidth frequency range in rad/sample following the equation:

      ``R = exp(-bandwidth / 2)``

    where R is the pole amplitude (radius).

  Returns
  -------
  A ZFilter object.
  Gain is normalized to have peak with 0 dB (1.0 amplitude).

  """
  bandwidth = thub(bandwidth, 1)
  R = exp(-bandwidth * .5)
  R = thub(R, 3)
  gain = (1 - R ** 2) * .5
  numerator = 1 - z ** -2
  denominator = 1 - 2 * R * cos(freq) * z ** -1 + R ** 2 * z ** -2
  return gain * numerator / denominator


lowpass = StrategyDict("lowpass")


@lowpass.strategy("pole")
def lowpass(cutoff):
  """
  Low-pass filter with one pole and no zeros (constant numerator), with
  high-precision cut-off frequency calculation.

  Parameters
  ----------
  cutoff :
    Cut-off frequency in rad/sample. It defines the filter frequency in which
    the squared gain is `50%` (a.k.a. magnitude gain is `sqrt(2) / 2` and
    power gain is about `3.0103 dB`).
    Should be a value between 0 and pi.

  Returns
  -------
  A ZFilter object.
  Gain is normalized to have peak with 0 dB (1.0 amplitude) at the DC
  frequency (zero rad/sample).

  """
  cutoff = thub(cutoff, 1)
  x = 2 - cos(cutoff)
  x = thub(x,2)
  R = x - sqrt(x ** 2 - 1)
  R = thub(R, 2)
  return (1 - R) / (1 - R * z ** -1)


@lowpass.strategy("pole_exp")
def lowpass(cutoff):
  """
  Low-pass filter with one pole and no zeros (constant numerator), with
  exponential approximation for cut-off frequency calculation, found by a
  matching the one-pole Laplace lowpass filter.

  Parameters
  ----------
  cutoff :
    Cut-off frequency in rad/sample following the equation:

      ``R = exp(-cutoff)``

    where R is the pole amplitude (radius).

  Returns
  -------
  A ZFilter object.
  Gain is normalized to have peak with 0 dB (1.0 amplitude) at the DC
  frequency (zero rad/sample).

  """
  cutoff = thub(cutoff, 1)
  R = exp(-cutoff)
  R = thub(R, 2)
  return (1 - R) / (1 - R * z ** -1)


highpass = StrategyDict("highpass")


@highpass.strategy("pole")
def highpass(cutoff):
  """
  High-pass filter with one pole and no zeros (constant numerator), with
  high-precision cut-off frequency calculation.

  Parameters
  ----------
  cutoff :
    Cut-off frequency in rad/sample. It defines the filter frequency in which
    the squared gain is `50%` (a.k.a. magnitude gain is `sqrt(2) / 2` and
    power gain is about `3.0103 dB`).
    Should be a value between 0 and pi.

  Returns
  -------
  A ZFilter object.
  Gain is normalized to have peak with 0 dB (1.0 amplitude) at the Nyquist
  frequency (pi rad/sample).

  """
  rev_cutoff = thub(pi - cutoff, 1)
  x = 2 - cos(rev_cutoff)
  x = thub(x,2)
  R = x - sqrt(x ** 2 - 1)
  R = thub(R, 2)
  return (1 - R) / (1 + R * z ** -1)
