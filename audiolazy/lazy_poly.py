#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Polynomial model

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

Created on Sun Oct 07 2012
danilo [dot] bellini [at] gmail [dot] com
"""

import operator

# Audiolazy internal imports
from .lazy_stream import AbstractOperatorOverloaderMeta
from .lazy_misc import (almost_eq, multiplication_formatter,
                        pair_strings_sum_formatter)


class PolyMeta(AbstractOperatorOverloaderMeta):
  """
  Poly metaclass. This class overloads few operators to the Poly class.
  All binary dunders (non reverse) should be implemented on the Poly class
  """
  __operators__ = ("and radd sub rsub " # elementwise
                   "mul rmul " # cross
                   "pow div truediv " # to mul, only when other is not Poly
                   "eq ne " # almost_eq comparison of Poly.terms
                   "pos neg " # simple unary elementwise
                  )

  def reverse_binary_dunder(cls, op_func):
    def dunder(self, other):
      return op_func(cls(other), self) # The "other" is probably a number
    return dunder

  def unary_dunder(cls, op_func):
    def dunder(self):
      return cls({k: op_func(v) for k, v in self.data.iteritems()})
    return dunder


class Poly(object):
  """
  Model for a polynomial.
  That's not a dict and not a list but behaves like something in between.
  The "values" method allows casting to list with list(Poly.values())
  The "terms" method allows casting to dict with dict(Poly.terms()), and give
  the terms sorted by their power value.
  """

  __metaclass__ = PolyMeta
  def __init__(self, data=None, zero=0):
    """
    Inits a polynomial from given data, which can be a list or a dict.
    A list [a_0, a_1, a_2, a_3, ...] inits a polynomial like
      a_0 + a_1 * x + a_2 * x**2 + a_3 * x**3 + ...
    If data is a dictionary, powers are the keys and the a_i factors are the
    values, so negative powers are allowed and you can neglect the zeros in
    between, i.e., a dict vith terms like {power: value} can also be used.
    """
    self.zero = zero
    if isinstance(data, list):
      self.data = {power: value for power, value in enumerate(data)}
    elif isinstance(data, dict):
      self.data = data.copy()
    elif isinstance(data, Poly):
      self.data = data.data.copy()
    elif data is None:
      self.data = {}
    else:
      self.data = {0: data}
    self._compact_zeros()

  def _compact_zeros(self):
    keys = [key for key, value in self.data.iteritems() if value == 0.]
    for key in keys:
      del self.data[key]

  def values(self):
    """
    Array values generator for powers from zero to upper power. Useful
    to cast as list/tuple and for numpy/scipy integration (be careful:
    numpy and scipy use the reversed from the output of this function used
    as input to a list or a tuple constructor)
    """
    max_key = max(key for key in self.data) if self.data else -1
    return (self.data[key] if key in self.data else self.zero
            for key in range(max_key + 1))

  def terms(self):
    """
    Pairs (2-tuple) generator where each tuple has a (power, value) term,
    sorted by power. Useful for casting as dict.
    """
    for key in sorted(self.data):
      yield key, self.data[key]

  def __len__(self):
    """
    Number of polynomial terms, not values (be careful).
    """
    return len(self.data)

  def diff(self, n=1):
    """
    Differentiate (n-th derivative, where the default n is 1).
    """
    return reduce(lambda dict_, order: # Derivative order can be ignored
                    {k - 1: k * v for k, v in dict_.iteritems() if k != 0},
                  xrange(n), self.data)

  def integrate(self):
    """
    Integrate without adding an integration constant.
    """
    if -1 in self.data:
      raise ValueError("Unable to integrate the polynomial term that powers "
                       "to -1, since the logarithm is not a polynomial")
    return {k + 1: v / (k + 1) for k, v in self.terms()}

  def __call__(self, value):
    """
    Apply value to the Poly, where value can be other Poly.
    """
    data_generator = (v * value ** k for k, v in self.terms())
    if isinstance(value, Poly):
      return sum(data_generator)
    return sum(sorted(data_generator,
                      key=lambda v: -abs(v)))

  # ---------------------
  # Elementwise operators
  # ---------------------
  def __add__(self, other):
    if not isinstance(other, Poly):
      other = Poly(other) # The "other" is probably a number
    intersect = [(key, self.data[key] + other.data[key])
                 for key in set(self.data).intersection(other.data)]
    return Poly(dict(self.data.items() + other.data.items() + intersect))

  def __sub__(self, other):
    return self + (-other)

  # -----------------------------
  # Cross-product based operators
  # -----------------------------
  def __mul__(self, other):
    if not isinstance(other, Poly):
      other = Poly(other) # The "other" is probably a number
    new_data = {}
    for k1, v1 in self.data.iteritems():
      for k2, v2 in other.data.iteritems():
        if k1 + k2 in new_data:
          new_data[k1 + k2] += v1 * v2
        else:
          new_data[k1 + k2] = v1 * v2
    return Poly(new_data)

  # -----------------------
  # Comparison (not strict)
  # -----------------------
  def __eq__(self, other):
    if not isinstance(other, Poly):
      other = Poly(other) # The "other" is probably a number
    return almost_eq(sorted(self.data.iteritems()),
                     sorted(other.data.iteritems())
                    )

  def __ne__(self, other):
    return not(self == other)

  # -----------------------------------------
  # Operators (mainly) for non-Poly instances
  # -----------------------------------------
  def __pow__(self, other):
    """
    Power operator. The "other" parameter should be an int (or anything like),
    but it works with float when the Poly has only one term.
    """
    if other == 0:
      return Poly(1)
    if len(self.data) == 0:
      return Poly()
    if len(self.data) == 1:
      return Poly({k * other: v for k, v in self.data.iteritems()})
    return reduce(operator.mul, [self] * other)

  def __div__(self, other):
    if isinstance(other, Poly):
      if len(other) == 1:
        delta, value = other.data.items()[0]
        return Poly({(k - delta): operator.div(v, other)
                     for k, v in self.data.iteritems()})
      raise NotImplementedError("Can't divide general Poly instances")
    return Poly({k: operator.div(v, other)
                 for k, v in self.data.iteritems()})

  def __truediv__(self, other):
    if isinstance(other, Poly):
      if len(other) == 1:
        delta, value = other.data.items()[0]
        return Poly({(k - delta): operator.truediv(v, other)
                     for k, v in self.data.iteritems()})
      raise NotImplementedError("Can't divide general Poly instances")
    return Poly({k: operator.truediv(v, other)
                 for k, v in self.data.iteritems()})

  # ---------------------
  # String representation
  # ---------------------
  def __str__(self):
    term_strings = [multiplication_formatter(power, value, "x")
                    for power, value in self.terms() if value != 0.]
    return "0" if len(term_strings) == 0 else \
           reduce(pair_strings_sum_formatter, term_strings)

  __repr__ = __str__
