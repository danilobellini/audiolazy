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
from cmath import exp
from collections import deque
import itertools as it

# Audiolazy internal imports
from .lazy_stream import Stream
from .lazy_misc import (elementwise, blocks, zero_pad,
                        multiplication_formatter, pair_strings_sum_formatter)
from .lazy_poly import Poly
from .lazy_core import AbstractOperatorOverloaderMeta

__all__ = ["LTI", "LTIFreqMeta", "LTIFreq", "z"]


class LTI(object):
  """
  Base class for Linear Time Invariant filters.
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

  @property
  def numerator(self):
    if any(key < 0 for key, value in self.numpoly.terms()):
      raise ValueError("Non-causal filter")
    return list(self.numpoly.values())

  @property
  def denominator(self):
    if any(key < 0 for key, value in self.denpoly.terms()):
      raise ValueError("Non-causal filter")
    return list(self.denpoly.values())

  def __call__(self, seq, memory=None, zero=0.):
    """
    IIR and FIR linear filtering.

    Parameters
    ----------
    seq :
      Any iterable to be seem as the input stream for the filter.
    memory :
      A sequence with length, such as a list or a Numpy 1D array. If it has
      more items from needed by the filter, it must implement slices. Less
      items than needed is completed with zeros. If ``None`` (default),
      memory is initialized with zeros.
    zero :
      Value to fill the memory, when needed, and to be seem as previous
      input when there's a delay. Defaults to ``0.0``.

    Returns
    -------
    A Stream that have the data from the input sequence filtered.

    """
    if isinstance(seq, Stream):
      seq = seq.data
    b, a = self.numerator, self.denominator # Just for type check
    rb, ra = list(reversed(b)), list(reversed(a[1:]))

    if b == []: # No numerator: input is fully neglect
      b = [zero]

    la, lb = len(a), len(b)
    gain = a[0]

    if la == 1: # No memory needed
      def gen(): # Filter loop
        for blk in blocks(zero_pad(seq, left=lb - 1), lb, 1):
          numerator = sum(it.imap(operator.mul, blk, rb))
          yield numerator / gain
    else:
      # Convert memory input to a deque with size exactly equals to len(a) - 1
      if memory is None:
        memory = [zero for unused in xrange(la - 1)]
      else:
        lm = len(memory)
        if lm > la - 1:
          memory = memory[:la - 1]
        elif memory < la - 1:
          memory = list(zero_pad(memory, la - 1 - lm, zero=zero))
      memory = deque(memory, maxlen=la - 1)

      def gen(): # Filter loop
        for blk in blocks(zero_pad(seq, left=lb - 1, zero=zero), lb, 1):
          numerator = sum(it.imap(operator.mul, blk, rb))
          denominator = sum(it.imap(operator.mul, memory, ra))
          next_val = (numerator - denominator) / gain
          yield next_val
          memory.append(next_val)
    return Stream(gen())

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
    Complex number with the frequency response of the filter. You can use
    ``20 * log10(abs(result))`` to get its power magnitude in dB, and
    Numpy ``angle(result)`` or built-in ``phase(result)`` to get its phase.

    """
    z_ = exp(-1j * freq)
    num = self.numpoly(z_)
    den = self.denpoly(z_)
    return num / den

  def is_causal(self):
    """
    Causality test for this filter.

    Returns
    -------
    Boolean returning True if this filter is causal, False otherwise.

    """
    return all(delay >= 0 for delay, value in self.numpoly.terms())


class LTIFreqMeta(AbstractOperatorOverloaderMeta):
  __operators__ = ("pos neg add radd sub rsub mul rmul div rdiv "
                   "truediv rtruediv pow "
                   "eq ne " # almost_eq comparison of Poly terms
                  )

  def __rbinary__(cls, op_func):
    def dunder(self, other):
      if isinstance(other, LTI):
        raise ValueError("Filter equations have different domains")
      return op_func(cls([other]), self) # The "other" is probably a number
    return dunder

  def __unary__(cls, op_func):
    def dunder(self):
      return cls(op_func(self.numpoly), self.denpoly)
    return dunder


class LTIFreq(LTI):
  """
  Linear Time Invariant filters based on Z-transform frequency domain
  equations.

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
  >>> filt = LTIFreq(b, a)
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
  __metaclass__ = LTIFreqMeta

  def __add__(self, other):
    if isinstance(other, LTIFreq):
      return LTIFreq(self.numpoly * other.denpoly +
                     other.numpoly * self.denpoly,
                     self.denpoly * other.denpoly)
    if isinstance(other, LTI):
      raise ValueError("Filter equations have different domains")
    return self + LTIFreq([other]) # Other is probably a number

  def __sub__(self, other):
    return self + (-other)

  def __mul__(self, other):
    if isinstance(other, LTIFreq):
      return LTIFreq(self.numpoly * other.numpoly,
                     self.denpoly * other.denpoly)
    if isinstance(other, LTI):
      raise ValueError("Filter equations have different domains")
    return LTIFreq(self.numpoly * other, self.denpoly)

  def __div__(self, other):
    if isinstance(other, LTIFreq):
      return LTIFreq(self.numpoly * other.denpoly,
                     self.denpoly * other.numpoly)
    if isinstance(other, LTI):
      raise ValueError("Filter equations have different domains")
    return self * operator.div(1, other)

  def __truediv__(self, other):
    if isinstance(other, LTIFreq):
      return LTIFreq(self.numpoly * other.denpoly,
                     self.denpoly * other.numpoly)
    if isinstance(other, LTI):
      raise ValueError("Filter equations have different domains")
    return self * operator.truediv(1, other)

  def __pow__(self, other):
    if (other < 0) and (len(self.numpoly) >= 2 or len(self.denpoly) >= 2):
      return LTIFreq(self.denpoly, self.numpoly) ** -other
    if isinstance(other, int):
      return LTIFreq(self.numpoly ** other, self.denpoly ** other)
    raise ValueError("Z-transform powers only valid with integers")

  def __eq__(self, other):
    if isinstance(other, LTI):
      return self.numpoly == other.numpoly and self.denpoly == other.denpoly
    return False

  def __ne__(self, other):
    if isinstance(other, LTI):
      return self.numpoly != other.numpoly and self.denpoly != other.denpoly
    return False

  def __str__(self):
    num_term_strings = [multiplication_formatter(-power, value, "z")
                        for power, value in self.numpoly.terms()
                        if value != 0.]
    num = "0" if len(num_term_strings) == 0 else \
          reduce(pair_strings_sum_formatter, num_term_strings)
    den_term_strings = [multiplication_formatter(-power, value, "z")
                        for power, value in self.denpoly.terms()
                        if value != 0.]
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
    if isinstance(mul_after, LTIFreq):
      den = LTIFreq(self.denpoly)
      return reduce(lambda num, order: mul_after *
                      (num.diff() * den - order * num * den.diff()),
                    xrange(1, n + 1),
                    LTIFreq(self.numpoly)
                   ) / den ** (n + 1)

    inv_sign = Poly({-1: 1}) # Since poly variable is z ** -1
    den = self.denpoly(inv_sign)
    return LTIFreq(reduce(lambda num, order: mul_after *
                            (num.diff() * den - order * num * den.diff()),
                          xrange(1, n + 1),
                          self.numpoly(inv_sign))(inv_sign),
                   self.denpoly ** (n + 1))

z = LTIFreq({-1: 1})
