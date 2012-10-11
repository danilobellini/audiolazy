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
from cmath import exp, pi
from collections import deque
import itertools as it

# Audiolazy internal imports
from .lazy_stream import Stream
from .lazy_misc import (elementwise, blocks, zero_pad,
                        multiplication_formatter, pair_strings_sum_formatter)
from .lazy_poly import Poly
from .lazy_core import AbstractOperatorOverloaderMeta


class LTI(object):
  """
  Base class for Linear Time Invariant Filters.
  """
  def __init__(self, numerator=None, denominator={0: 1.}):
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

  def __call__(self, sequence, memory=None):
    """
    IIR and FIR linear filtering.
    Returns a Stream that have the data from the input sequence filtered.
    """
    if isinstance(sequence, Stream):
      sequence = sequence.data
    b, a = self.numerator, self.denominator # Just for type check
    rb, ra = list(reversed(b)), list(reversed(a[1:]))

    if b == []:
      b = [0.]

    la, lb = len(a), len(b)
    gain = a[0]

    if la == 1: # No memory needed
      def gen(): # Filter loop
        for blk in blocks(zero_pad(sequence, left=lb - 1), lb, 1):
          numerator = sum(it.imap(operator.mul, blk, rb))
          yield numerator / gain
    else:
      # Convert memory input to a deque with size exactly equals to len(a) - 1
      if memory is None:
        memory = [0. for _ in xrange(la - 1)]
      else:
        lm = len(memory)
        if lm > la - 1:
          memory = memory[:la - 1]
        elif memory < la - 1:
          memory = list(zero_pad(memory, la - 1 - lm))
      memory = deque(memory, maxlen=la - 1)

      def gen(): # Filter loop
        for blk in blocks(zero_pad(sequence, left=lb - 1), lb, 1):
          numerator = sum(it.imap(operator.mul, blk, rb))
          denominator = sum(it.imap(operator.mul, memory, ra))
          next_val = (numerator - denominator) / gain
          yield next_val
          memory.append(next_val)
    return Stream(gen())

  @elementwise(1)
  def freq_response(self, freq):
    """
    Frequency response for this filter. Frequency should be given in rad/s.
    """
    z_ = exp(-2j * pi * freq / rate)
    num = self.numpoly(z_)
    den = self.denpoly(z_)
    return num / den

  def is_causal(self):
    return all(delay >= 0 for delay, value in self.numpoly.terms()) and \
           all(delay >= 0 for delay, value in self.denpoly.terms())


class LTIFreqMeta(AbstractOperatorOverloaderMeta):
  __operators__ = ("pos neg and radd sub rsub mul rmul div rdiv "
                   "truediv rtruediv pow "
                   "eq ne " # almost_eq comparison of Poly terms
                  )

  def reverse_binary_dunder(cls, op_func):
    def dunder(self, other):
      if isinstance(other, LTI):
        raise ValueError("Filter equations have different domains")
      return op_func(cls([other]), self) # The "other" is probably a number
    return dunder

  def unary_dunder(cls, op_func):
    def dunder(self):
      return cls(op_func(self.numpoly), self.denpoly)
    return dunder


class LTIFreq(LTI):
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
