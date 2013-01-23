# -*- coding: utf-8 -*-
"""
Math module "decorated" and complemented to work elementwise when needed

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

Created on Thu Nov 08 2012
danilo [dot] bellini [at] gmail [dot] com
"""

import math
import cmath
import operator
import itertools as it

# Audiolazy internal imports
from .lazy_misc import elementwise

__all__ = ["abs", "pi", "e", "cexp", "ln", "log2", "factorial", "dB10",
           "dB20", "inf", "nan"]

# All functions from math with one numeric input
math_names = ["acos", "acosh", "asin", "asinh", "atan", "atanh", "ceil",
              "cos", "cosh", "degrees", "erf", "erfc", "exp", "expm1",
              "fabs", "floor", "frexp", "gamma", "isinf", "isnan", "lgamma",
              "log", "log10", "log1p", "modf", "radians", "sin", "sinh",
              "sqrt", "tan", "tanh", "trunc"]
__all__.extend(math_names)


for func in [getattr(math, name) for name in math_names]:
  locals()[func.__name__] = elementwise("x", 0)(func)


abs = elementwise("number", 0)(abs)
pi = math.pi
e = math.e
cexp = elementwise("x", 0)(cmath.exp)
ln = elementwise("x", 0)(math.log)
inf = float("inf")
nan = float("nan")
phase = elementwise("z", 0)(cmath.phase)


@elementwise("x", 0)
def log2(x):
  return math.log(x, 2)


@elementwise("n", 0)
def factorial(n):
  """
  Factorial function that works with really big numbers.
  """
  if isinstance(n, float):
    if n.is_integer():
      n = int(n)
  if not isinstance(n, (int, long)):
    raise TypeError("non-integer input (perhaps you need Euler Gamma "
                    "function or Gauss Pi function)")
  if n < 0:
    raise ValueError("input shouldn't be negative")
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
  Convert a gain value to dB, from a amplitude value to a power gain.
  """
  return 20 * math.log10(abs(data)) if data != 0 else -inf
