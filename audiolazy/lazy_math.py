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
Math modules "decorated" and complemented to work elementwise when needed
"""

import math
import cmath
import operator
import itertools as it
from functools import reduce

# Audiolazy internal imports
from .lazy_misc import elementwise
from .lazy_compat import INT_TYPES

__all__ = ["absolute", "pi", "e", "cexp", "ln", "log", "log1p", "log10",
           "log2", "factorial", "dB10", "dB20", "inf", "nan", "phase", "sign"]

# All functions from math with one numeric input
_math_names = ["acos", "acosh", "asin", "asinh", "atan", "atanh", "ceil",
               "cos", "cosh", "degrees", "erf", "erfc", "exp", "expm1",
               "fabs", "floor", "frexp", "gamma", "isinf", "isnan", "lgamma",
               "modf", "radians", "sin", "sinh", "sqrt", "tan", "tanh",
               "trunc"]
__all__.extend(_math_names)


for func in [getattr(math, name) for name in _math_names]:
  locals()[func.__name__] = elementwise("x", 0)(func)


@elementwise("x", 0)
def log(x, base=None):
  if base is None:
    if x == 0:
      return -inf
    elif isinstance(x, complex) or x < 0:
      return cmath.log(x)
    else:
      return math.log(x)
  else: # base is given
    if base <= 0 or base == 1:
      raise ValueError("Not a valid logarithm base")
    elif x == 0:
      return -inf
    elif isinstance(x, complex) or x < 0:
      return cmath.log(x, base)
    else:
      return math.log(x, base)


@elementwise("x", 0)
def log1p(x):
  if x == -1:
    return -inf
  elif isinstance(x, complex) or x < -1:
    return cmath.log(1 + x)
  else:
    return math.log1p(x)


def log10(x):
  return log(x, 10)


def log2(x):
  return log(x, 2)


ln = log
absolute = elementwise("number", 0)(abs)
pi = math.pi
e = math.e
cexp = elementwise("x", 0)(cmath.exp)
inf = float("inf")
nan = float("nan")
phase = elementwise("z", 0)(cmath.phase)


@elementwise("n", 0)
def factorial(n):
  """
  Factorial function that works with really big numbers.
  """
  if isinstance(n, float):
    if n.is_integer():
      n = int(n)
  if not isinstance(n, INT_TYPES):
    raise TypeError("Non-integer input (perhaps you need Euler Gamma "
                    "function or Gauss Pi function)")
  if n < 0:
    raise ValueError("Input shouldn't be negative")
  return reduce(operator.mul,
                it.takewhile(lambda m: m <= n, it.count(2)),
                1)


@elementwise("data", 0)
def dB10(data):
  """
  Convert a gain value to dB, from a squared amplitude value to a power gain.
  """
  return 10 * math.log10(abs(data)) if data != 0 else -inf


@elementwise("data", 0)
def dB20(data):
  """
  Convert a gain value to dB, from an amplitude value to a power gain.
  """
  return 20 * math.log10(abs(data)) if data != 0 else -inf


@elementwise("x", 0)
def sign(x):
  """
  Signal of ``x``: 1 if positive, -1 if negative, 0 otherwise.
  """
  return +(x > 0) or -(x < 0)
