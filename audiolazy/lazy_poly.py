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
Polynomial model and Waring-Lagrange polynomial interpolator
"""

from __future__ import division

import operator
from collections import Iterable, deque, OrderedDict
from functools import reduce
import itertools as it

# Audiolazy internal imports
from .lazy_core import AbstractOperatorOverloaderMeta, StrategyDict
from .lazy_text import multiplication_formatter, pair_strings_sum_formatter
from .lazy_misc import rint
from .lazy_compat import meta, iteritems, xrange, xzip, INT_TYPES
from .lazy_stream import Stream, tostream, thub

__all__ = ["PolyMeta", "Poly", "x", "lagrange", "resample"]


class PolyMeta(AbstractOperatorOverloaderMeta):
  """
  Poly metaclass. This class overloads few operators to the Poly class.
  All binary dunders (non reverse) should be implemented on the Poly class
  """
  __operators__ = ("+ - " # elementwise / unary
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
      return cls(OrderedDict((k, op_func(v))
                             for k, v in iteritems(self._data)),
                 zero=self.zero)
    return dunder


class Poly(meta(metaclass=PolyMeta)):
  """
  Model for a polynomial, a Laurent polynomial or a sum of arbitrary powers.

  The "values" method allows casting simple polynomials to a list with
  ``list(Poly.values())`` where the index values are the powers.
  The "terms" method allows casting any sum of powers to dict with
  ``dict(Poly.terms())`` where the keys are the powers and the values are the
  coefficients, and this method can also give the terms sorted by their power
  value if needed. Both methods return a generator.

  Usually the instances of this class should be seen as immutable (this is
  a hashable instance), although there's no enforcement for that (and item
  set is allowed) until the hash is required.

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
    zeros in between, i.e., a dict with terms like ``{power: value}`` can also
    be used.

    """
    self._zero = 0. if zero is None else zero
    if isinstance(data, list):
      self._data = OrderedDict(enumerate(data))
    elif isinstance(data, dict):
      self._data = OrderedDict(data)
    elif isinstance(data, Poly):
      self._data = OrderedDict(data._data)
      self._zero = data._zero if zero is None else zero
    elif data is None:
      self._data = OrderedDict()
    else:
      self._data = OrderedDict([(0, data)])

    # Compact zeros
    for key, value in list(iteritems(self._data)):
      if isinstance(key, float) and key.is_integer():
        del self._data[key]
        key = rint(key)
        self._data[key] = value
      if (not isinstance(value, Stream)) and value == self.zero:
        del self._data[key]

  @property
  def zero(self):
    return self._zero

  @zero.setter
  def zero(self, value):
    if hasattr(self, "_hash"):
      raise TypeError("Used this Poly instance as a hashable before")
    self._zero = value
    for power, coeff in list(iteritems(self._data)):
      if (not isinstance(coeff, Stream)) and coeff == value:
        del self._data[power]

  def __hash__(self):
    if not hasattr(self, "_hash"): # Should make this instance immutable
      self._hash = hash((frozenset(iteritems(self._data)), self.zero))
    return self._hash

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

  def terms(self, sort="auto", reverse=False):
    """
    Pairs (2-tuple) generator where each tuple has a (power, value) term,
    perhaps sorted by power. Useful for casting as dict.

    Parameters
    ----------
    sort :
      A boolean value or ``"auto"`` (default) which chooses whether the terms
      should be sorted. The ``"auto"`` value means ``True`` for Laurent
      polynomials (i.e., integer powers), ``False`` otherwise. If sorting is
      disabled, this method will yield the terms in the creation order.
    reverse :
      Boolean to chooses whether the [sorted or creation] order should be
      reversed when yielding the pairs. If False (default), yields in
      ascending or creation order (not counting possible updates in the
      coefficients).
    """
    if sort == "auto":
      sort = self.is_laurent()

    if sort:
      keys = sorted(self._data, reverse=reverse)
    elif reverse:
      keys = reversed(list(self._data))
    else:
      keys = self._data

    return ((k, self._data[k]) for k in keys)

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
    return Poly(OrderedDict((k, v.copy() if isinstance(v, Stream) else v)
                            for k, v in iteritems(self._data)),
                zero=self.zero if zero is None else zero)

  def diff(self, n=1):
    """
    Differentiate (n-th derivative, where the default n is 1).
    """
    d = self._data
    for unused in xrange(n):
      d = OrderedDict((k - 1, k * v) for k, v in iteritems(d) if k != 0)
    return Poly(d, zero=self.zero)

  def integrate(self):
    """
    Integrate without adding an integration constant.
    """
    if -1 in self._data:
      raise ValueError("Unable to integrate term that powers to -1")
    return Poly(OrderedDict((k + 1, v / (k + 1))
                            for k, v in iteritems(self._data)),
                zero=self.zero)

  def __call__(self, value, horner="auto"):
    """
    Apply the given value to the Poly.

    Parameters
    ----------
    value :
      The value to be applied. This can possibly be another Poly, or perhaps
      a Stream instance.
    horner :
      A value in [True, False, "auto"] which chooses whether a Horner-like
      scheme should be used for the evaluation. Defaults to ``"auto"``,
      which means the scheme is done only for simple polynomials. This scheme
      can be forced on Laurent polynomials and any other sum of powers that
      have sortable powers. This input is neglect when ``value`` is a Poly
      instance.

    Note
    ----
    For a polynomial with all values, the Horner scheme is done like expected,
    e.g. replacing ``x ** 2 + 2 * x + 1`` by ``x * (x + 2) + 1`` in the
    evaluation. However, for zeroed coeffs, it will merge steps, using
    powers instead of several multiplications, e.g. ``x ** 7 + x ** 6 + 4``
    would evaluate as ``x ** 6 * (x + 1) + 4``, taking the ``x ** 6`` instead
    of multiplying ``(x + 1)`` by ``x`` six times (i.e., the evaluation order
    is changed). This allows a faster approach for sparse polynomials and
    extends the scheme for any sum of powers with sortable powers.
    """
    # Polynomial value ("x" variable replacement)
    if isinstance(value, Poly):
      return Poly(sum(coeff * value ** power
                      for power, coeff in iteritems(self._data)),
                  self.zero)

    # Empty polynomial
    if not self._data:
      return self.zero

    # Evaluation for "x = 0"
    if not isinstance(value, Stream):
      if value == 0:
        return self[0]

    # Some initialization for the general process
    value = thub(value, len(self))
    if horner == "auto":
      horner = self.is_polynomial()

    # Horner scheme
    if horner:
      try:
        pairs = self.terms(sort=True, reverse=True)
      except TypeError: # No ordering relation defined
        raise ValueError("Can't apply Horner-like scheme")

      def horner_step(old, new):
        opower, oresult = old
        npower, ncoeff = new
        scale = value if opower == npower + 1 else value ** (opower - npower)
        return npower, ncoeff + oresult * scale

      last_power, result = reduce(horner_step, pairs)
      return result * value ** last_power

    # General case (Laurent, float/complex powers, weird things, etc.)
    return sum(coeff * value ** power for power, coeff in self.terms())

  def __getitem__(self, item):
    if item in self._data:
      return self._data[item]
    else:
      return self.zero

  def __setitem__(self, power, coeff):
    if getattr(self, "_hash", False):
      raise TypeError("Used this Poly instance as a hashable before")

    if isinstance(power, float) and power.is_integer():
      power = rint(power)

    if isinstance(coeff, Stream) or coeff != self.zero:
      self._data[power] = coeff
    elif power in self._data:
      del self._data[power]


  # ---------------------
  # Elementwise operators
  # ---------------------
  def __add__(self, other):
    if not isinstance(other, Poly):
      other = Poly(other) # The "other" is probably a number
    intersect = [(key, self._data[key] + other._data[key])
                 for key in set(self._data).intersection(other._data)]
    return Poly(OrderedDict(it.chain(iteritems(self._data),
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
    new_data = OrderedDict()
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
      other = Poly(other, zero=self.zero) # To compare only Poly instances

    def is_pair_equal(a, b):
      if isinstance(a, Stream) or isinstance(b, Stream):
        return a is b
      return a == b

    def dicts_equal(a, b):
      return len(a) == len(b) and \
             all(k in b and is_pair_equal(v, b[k]) for k, v in iteritems(a))

    return is_pair_equal(self._zero, other._zero) and \
           dicts_equal(self._data, other._data)

  def __ne__(self, other):
    return not (self == other)

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
      return Poly(OrderedDict((k * other,
                               1 if v == 1 else v ** other) # Avoid casting
                              for k, v in iteritems(self._data)),
                  zero=self.zero)
    return reduce(operator.mul, [self.copy()] * (other - 1) + [self])

  def __truediv__(self, other):
    if isinstance(other, Poly):
      if len(other) == 1:
        delta, value = next(iteritems(other._data))
        return Poly(OrderedDict(((k - delta), operator.truediv(v, value))
                                for k, v in iteritems(self._data)),
                    zero=self.zero)
      elif len(other) == 0:
        raise ZeroDivisionError("Dividing Poly instance by zero")
      raise NotImplementedError("Can't divide general Poly instances")
    other = thub(other, len(self))
    return Poly(OrderedDict((k, operator.truediv(v, other))
                            for k, v in iteritems(self._data)),
                zero=self.zero)

  # ---------------------
  # String representation
  # ---------------------
  def __str__(self):
    term_strings = []
    for power, value in self.terms():
      if isinstance(value, Iterable):
        value = "a{}".format(power).replace(".", "_").replace("-", "m")
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
