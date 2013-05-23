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
# Created on Sun Oct 07 2012
# danilo [dot] bellini [at] gmail [dot] com
"""
Polynomial model and Waring-Lagrange polynomial interpolator
"""

from __future__ import division

import operator
from collections import Iterable, deque
from functools import reduce
import itertools as it

# Audiolazy internal imports
from .lazy_core import AbstractOperatorOverloaderMeta, StrategyDict
from .lazy_text import multiplication_formatter, pair_strings_sum_formatter
from .lazy_misc import rint
from .lazy_compat import (meta, iteritems, xrange, xzip, INT_TYPES,
                          xzip_longest)
from .lazy_stream import Stream, tostream, thub

__all__ = ["PolyMeta", "Poly", "x", "lagrange", "resample"]


class PolyMeta(AbstractOperatorOverloaderMeta):
  """
  Poly metaclass. This class overloads few operators to the Poly class.
  All binary dunders (non reverse) should be implemented on the Poly class
  """
  __operators__ = ("+ - " # elementwise
                   "* " # cross
                   "pow truediv " # when other is not Poly (no reverse)
                   "eq ne ") # comparison of Poly terms

  def __rbinary__(cls, op):
    op_func = op.func
    def dunder(self, other): # The "other" is probably a number
      return op_func(cls(other, zero=self.zero), self)
    return dunder

  def __unary__(cls, op):
    op_func = op.func
    def dunder(self):
      return cls({k: op_func(v) for k, v in self.terms()},
                 zero=self.zero)
    return dunder


class Poly(meta(metaclass=PolyMeta)):
  """
  Model for a polynomial, Laurent polynomial or a sum of powers.

  That's not a dict and not a list but behaves like something in between.
  The "values" method allows casting to list with list(Poly.values())
  The "terms" method allows casting to dict with dict(Poly.terms()), and give
  the terms sorted by their power value if used in a loop instead of casting.

  Usually the instances of this class should be seen as immutable (this is
  a hashable instance), although there's no enforcement for that (and item
  set is allowed).

  You can use the ``x`` object and operators to create your own instances.

  Examples
  --------
  >>> x ** 5 - x + 7
  7 - x + x^5
  >>> type(x + 1)
  <class 'audiolazy.lazy_poly.Poly'>
  >>> (x + 2)(17)
  19
  >>> (x ** 2 + 2 * x + 1)(2)
  9
  >>> (x ** 2 + 2 * x + 1)(.5)
  2.25
  >>> (x ** -2 + x)(10)
  10.01
  >>> spow = x ** -2.1 + x ** .3 + x ** -1 + x - 6
  >>> value = spow(5)
  >>> "{:.6f}".format(value) # Just to see the first few digits
  '0.854710'

  """
  def __init__(self, data=None, zero=None):
    """
    Inits a polynomial from given data, which can be a list or a dict.

    A list :math:`[a_0, a_1, a_2, a_3, ...]` inits a polynomial like

    .. math::

      a_0 + a_1 . x + a_2 . x^2 + a_3 . x^3 + ...

    If data is a dictionary, powers are the keys and the :math:`a_i` factors
    are the values, so negative powers are allowed and you can neglect the
    zeros in between, i.e., a dict vith terms like ``{power: value}`` can also
    be used.

    """
    self.zero = 0. if zero is None else zero
    if isinstance(data, list):
      self._data = {power: value for power, value in enumerate(data)}
    elif isinstance(data, dict):
      self._data = dict(data)
    elif isinstance(data, Poly):
      self._data = data._data.copy()
      self.zero = data.zero if zero is None else zero
    elif data is None:
      self._data = {}
    else:
      self._data = {0: data}

    # Compact zeros
    for key, value in list(iteritems(self._data)):
      if isinstance(key, float):
        if key.is_integer():
          del self._data[key]
          key = rint(key)
          self._data[key] = value
      if not isinstance(value, Stream):
        if value == 0:
          del self._data[key]

  def __hash__(self):
    self._hashed = True
    return hash(tuple(self.terms()))

  def values(self):
    """
    Array values generator for powers from zero to upper power. Useful to cast
    as list/tuple and for numpy/scipy integration (be careful: numpy use the
    reversed from the output of this function used as input to a list or a
    tuple constructor).
    """
    if self._data:
      for key in xrange(self.order + 1):
        yield self[key]

  def terms(self):
    """
    Pairs (2-tuple) generator where each tuple has a (power, value) term,
    sorted by power. Useful for casting as dict.
    """
    for key in sorted(self._data):
      yield key, self._data[key]

  def __len__(self):
    """
    Number of terms, not values (be careful).
    """
    return len(self._data)

  def is_polynomial(self):
    """
    Tells whether it is a linear combination of natural powers of ``x``.
    """
    return all(isinstance(k, INT_TYPES) and k >= 0 for k in self._data)

  def is_laurent(self):
    """
    Boolean that indicates whether is a Laurent polynomial or not.

    A Laurent polynomial is any sum of integer powers of ``x``.

    Examples
    --------
    >>> (x + 4).is_laurent()
    True
    >>> (x ** -3 + 4).is_laurent()
    True
    >>> (x ** -3 + 4).is_polynomial()
    False
    >>> (x ** 1.1 + 4).is_laurent()
    False

    """
    return all(isinstance(k, INT_TYPES) for k in self._data)

  @property
  def order(self):
    """
    Finds the polynomial order.

    Examples
    --------
    >>> (x + 4).order
    1
    >>> (x + 4 - x ** 18).order
    18
    >>> (x - x).order
    0
    >>> (x ** -3 + 4).order
    Traceback (most recent call last):
      ...
    AttributeError: Power needs to be positive integers

    """
    if not self.is_polynomial():
      raise AttributeError("Power needs to be positive integers")
    return max(key for key in self._data) if self._data else 0

  def copy(self, zero=None):
    """
    Returns a Poly instance with the same terms, but as a "T" (tee) copy
    when they're Stream instances, allowing maths using a polynomial more
    than once.
    """
    return Poly({k: v.copy() if isinstance(v, Stream) else v
                 for k, v in self.terms()},
                zero=self.zero if zero is None else zero)

  def diff(self, n=1):
    """
    Differentiate (n-th derivative, where the default n is 1).
    """
    return Poly(reduce(lambda d, order: # Derivative order can be ignored
                         {k - 1: k * v for k, v in iteritems(d) if k != 0},
                       xrange(n), self._data),
                zero=self.zero)

  def integrate(self):
    """
    Integrate without adding an integration constant.
    """
    if -1 in self._data:
      raise ValueError("Unable to integrate term that powers to -1")
    return Poly({k + 1: v / (k + 1) for k, v in self.terms()},
                zero=self.zero)

  def __call__(self, value):
    """
    Apply value to the Poly, where value can be other Poly.
    When value is a number, a Horner-like scheme is done.
    """
    if isinstance(value, Poly):
      return Poly(sum(coeff * value ** power
                      for power, coeff in iteritems(self._data)),
                  self.zero)
    if not self._data:
      return self.zero
    if not isinstance(value, Stream):
      if value == 0:
        return self[0]

    value = thub(value, len(self))
    return reduce(
      lambda old, new: (new[0], new[1] + old[1] * value ** (old[0] - new[0])),
      sorted(iteritems(self._data), reverse=True) + [(0, 0)]
    )[1]

  def __getitem__(self, item):
    if item in self._data:
      return self._data[item]
    else:
      return self.zero

  def __setitem__(self, power, item):
    if getattr(self, "_hashed", False):
      raise TypeError("Used this Poly instance as a hashable before")
    self._data[power] = item

  # ---------------------
  # Elementwise operators
  # ---------------------
  def __add__(self, other):
    if not isinstance(other, Poly):
      other = Poly(other) # The "other" is probably a number
    intersect = [(key, self._data[key] + other._data[key])
                 for key in set(self._data).intersection(other._data)]
    return Poly(dict(it.chain(iteritems(self._data),
                              iteritems(other._data), intersect)),
                zero=self.zero)

  def __sub__(self, other):
    return self + (-other)

  # -----------------------------
  # Cross-product based operators
  # -----------------------------
  def __mul__(self, other):
    if not isinstance(other, Poly):
      other = Poly(other) # The "other" is probably a number
    new_data = {}
    thubbed_self = [(k, thub(v, len(other._data)))
                    for k, v in iteritems(self._data)]
    thubbed_other = [(k, thub(v, len(self._data)))
                     for k, v in iteritems(other._data)]
    for k1, v1 in thubbed_self:
      for k2, v2 in thubbed_other:
        if k1 + k2 in new_data:
          new_data[k1 + k2] += v1 * v2
        else:
          new_data[k1 + k2] = v1 * v2
    return Poly(new_data, zero=self.zero)

  # ----------
  # Comparison
  # ----------
  def __eq__(self, other):
    if not isinstance(other, Poly):
      other = Poly(other, zero=self.zero) # The "other" is probably a number

    def sorted_flattenizer(instance):
      return reduce(operator.concat, instance.terms(), tuple())

    def is_pair_equal(a, b):
      if isinstance(a, Stream) or isinstance(b, Stream):
        return a is b
      return a == b

    for pair in xzip_longest(sorted_flattenizer(self),
                             sorted_flattenizer(other)):
      if not is_pair_equal(*pair):
        return False
    return is_pair_equal(self.zero, other.zero)

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
    if isinstance(other, Poly):
      if any(k != 0 for k, v in other.terms()):
        raise NotImplementedError("Can't power general Poly instances")
      other = other[0]
    if other == 0:
      return Poly(1, zero=self.zero)
    if len(self._data) == 0:
      return Poly(zero=self.zero)
    if len(self._data) == 1:
      return Poly({k * other: 1 if v == 1 else v ** other # To avoid casting
                   for k, v in iteritems(self._data)},
                  zero=self.zero)
    return reduce(operator.mul, [self.copy()] * (other - 1) + [self])

  def __truediv__(self, other):
    if isinstance(other, Poly):
      if len(other) == 1:
        delta, value = next(iteritems(other._data))
        return Poly({(k - delta): operator.truediv(v, value)
                     for k, v in iteritems(self._data)},
                    zero=self.zero)
      elif len(other) == 0:
        raise ZeroDivisionError("Dividing Poly instance by zero")
      raise NotImplementedError("Can't divide general Poly instances")
    other = thub(other, len(self))
    return Poly({k: operator.truediv(v, other)
                 for k, v in iteritems(self._data)},
                zero=self.zero)

  # ---------------------
  # String representation
  # ---------------------
  def __str__(self):
    term_strings = []
    for power, value in self.terms():
      if isinstance(value, Iterable):
        value = "a{}".format(power).replace(".", "_").replace("-", "m")
      if value != 0.:
        term_strings.append(multiplication_formatter(power, value, "x"))
    return "0" if len(term_strings) == 0 else \
           reduce(pair_strings_sum_formatter, term_strings)

  __repr__ = __str__

  # -----------
  # NumPy-based
  # -----------
  @property
  def roots(self):
    """
    Returns a list with all roots. Needs Numpy.
    """
    import numpy as np
    return np.roots(list(self.values())[::-1]).tolist()


x = Poly({1: 1})
lagrange = StrategyDict("lagrange")


@lagrange.strategy("func")
def lagrange(pairs):
  """
  Waring-Lagrange interpolator function.

  Parameters
  ----------
  pairs :
    Iterable with pairs (tuples with two values), corresponding to points
    ``(x, y)`` of the function.

  Returns
  -------
  A function that returns the interpolator result for a given ``x``.

  """
  prod = lambda args: reduce(operator.mul, args)
  xv, yv = xzip(*pairs)
  return lambda k: sum( yv[j] * prod( (k - rk) / (rj - rk)
                                      for rk in xv if rj != rk )
                        for j, rj in enumerate(xv) )


@lagrange.strategy("poly")
def lagrange(pairs):
  """
  Waring-Lagrange interpolator polynomial.

  Parameters
  ----------
  pairs :
    Iterable with pairs (tuples with two values), corresponding to points
    ``(x, y)`` of the function.

  Returns
  -------
  A Poly instance that allows finding the interpolated value for any ``x``.

  """
  return lagrange.func(pairs)(x)


@tostream
def resample(sig, old=1, new=1, order=3, zero=0.):
  """
  Generic resampler based on Waring-Lagrange interpolators.

  Parameters
  ----------
  sig :
    Input signal (any iterable).
  old :
    Time duration reference (defaults to 1, allowing percentages to the ``new``
    keyword argument). This can be float number, or perhaps a Stream instance.
  new :
    Time duration that the reference will have after resampling.
    For example, if ``old = 1, new = 2``, then
    there will be 2 samples yielded for each sample from input.
    This can be a float number, or perhaps a Stream instance.
  order :
    Lagrange interpolator order. The amount of neighboring samples to be used by
    the interpolator is ``order + 1``.
  zero :
    The input should be thought as zero-padded from the left with this value.

  Returns
  -------
  The first value will be the first sample from ``sig``, and then the
  interpolator will find the next samples towards the end of the ``sig``.
  The actual sampling interval (or time step) for this interpolator obeys to
  the ``old / new`` relationship.

  Hint
  ----
  The time step can also be time-varying, although that's certainly difficult
  to synchonize (one sample is needed for each output sample). Perhaps the
  best approach for this case would be a ControlStream keeping the desired
  value at any time.

  Note
  ----
  The input isn't zero-padded at right. It means that the last output will be
  one with interpolated with known data. For endless inputs that's ok, this
  makes no difference, but for finite inputs that may be undesirable.

  """
  sig = Stream(sig)
  threshold = .5 * (order + 1)
  step = old / new
  data = deque([zero] * (order + 1), maxlen=order + 1)
  data.extend(sig.take(rint(threshold)))
  idx = int(threshold)
  isig = iter(sig)
  if isinstance(step, Iterable):
    step = iter(step)
    while True:
      yield lagrange(enumerate(data))(idx)
      idx += next(step)
      while idx > threshold:
        data.append(next(isig))
        idx -= 1
  else:
    while True:
      yield lagrange(enumerate(data))(idx)
      idx += step
      while idx > threshold:
        data.append(next(isig))
        idx -= 1
