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
# Created on Fri Jul 20 2012
# danilo [dot] bellini [at] gmail [dot] com
"""
Common miscellanous tools and constants for general use
"""

from collections import deque, Iterable
from functools import wraps
import itertools as it
import sys
from math import pi
import operator

# Audiolazy internal imports
from .lazy_compat import (xrange, xzip_longest, STR_TYPES, SOME_GEN_TYPES,
                          iteritems)
from .lazy_core import StrategyDict

__all__ = ["DEFAULT_SAMPLE_RATE", "rint", "blocks", "zero_pad", "elementwise",
           "almost_eq", "sHz"]

DEFAULT_SAMPLE_RATE = 44100 # Hz (samples/second)


def rint(x, step=1):
  """
  Round to integer.

  Parameters
  ----------
  x :
    Input number (integer or float) to be rounded.
  step :
    Quantization level (defaults to 1). If set to 2, the output will be the
    "best" even number.

  Result
  ------
  The step multiple nearest to x. When x is exactly halfway between two
  possible outputs, it'll result the one farthest to zero.

  """
  div, mod = divmod(x, step)
  err = min(step / 10., .1)
  result = div * step
  if x > 0:
    result += err
  elif x < 0:
    result -= err
  if (operator.ge if x >= 0 else operator.gt)(2 * mod, step):
    result += step
  return int(result)


def blocks(seq, size=None, hop=None, padval=0.):
  """
  General iterable blockenizer.

  Generator that gets ``size`` elements from ``seq``, and outputs them in a
  collections.deque (mutable circular queue) sequence container. Next output
  starts ``hop`` elements after the first element in last output block. Last
  block may be appended with ``padval``, if needed to get the desired size.

  The ``seq`` can have hybrid / hetherogeneous data, it just need to be an
  iterable. You can use other type content as padval (e.g. None) to help
  segregate the padding at the end, if desired.

  Note
  ----
  When hop is less than size, changing the returned contents will keep the
  new changed value in the next yielded container.

  """
  # Initialization
  res = deque(maxlen=size) # Circular queue
  idx = 0
  last_idx = size - 1
  if hop is None:
    hop = size
  reinit_idx = size - hop

  # Yields each block, keeping last values when needed
  if hop <= size:
    for el in seq:
      res.append(el)
      if idx == last_idx:
        yield res
        idx = reinit_idx
      else:
        idx += 1

  # Yields each block and skips (loses) data due to hop > size
  else:
    for el in seq:
      if idx < 0: # Skips data
        idx += 1
      else:
        res.append(el)
        if idx == last_idx:
          yield res
          #res = dtype()
          idx = size-hop
        else:
          idx += 1

  # Padding to finish
  if idx > max(size-hop, 0):
    for _ in xrange(idx,size):
      res.append(padval)
    yield res


def zero_pad(seq, left=0, right=0, zero=0.):
  """
  Zero padding sample generator (not a Stream!).

  Parameters
  ----------
  seq :
    Sequence to be padded.
  left :
    Integer with the number of elements to be padded at left (before).
    Defaults to zero.
  right :
    Integer with the number of elements to be padded at right (after).
    Defaults to zero.
  zero :
    Element to be padded. Defaults to a float zero (0.0).

  Returns
  -------
  A generator that pads the given ``seq`` with samples equals to ``zero``,
  ``left`` times before and ``right`` times after it.

  """
  for unused in xrange(left):
    yield zero
  for item in seq:
    yield item
  for unused in xrange(right):
    yield zero


def elementwise(name="", pos=None):
  """
  Function auto-map decorator broadcaster.

  Creates an "elementwise" decorator for one input parameter. To create such,
  it should know the name (for use as a keyword argument and the position
  "pos" (input as a positional argument). Without a name, only the
  positional argument will be used. Without both name and position, the
  first positional argument will be used.

  """
  if (name == "") and (pos is None):
    pos = 0
  def elementwise_decorator(func):
    """
    Element-wise decorator for functions known to have 1 input and 1
    output be applied directly on iterables. When made to work with more
    than 1 input, all "secondary" parameters will the same in all
    function calls (i.e., they will not even be a copy).

    """
    @wraps(func)
    def wrapper(*args, **kwargs):

      # Find the possibly Iterable argument
      positional = (pos is not None) and (pos < len(args))
      arg = args[pos] if positional else kwargs[name]

      if isinstance(arg, Iterable) and not isinstance(arg, STR_TYPES):
        if positional:
          data = (func(*(args[:pos] + (x,) + args[pos+1:]),
                       **kwargs)
                  for x in arg)
        else:
          data = (func(*args,
                       **dict(it.chain(iteritems(kwargs), [(name, x)])))
                  for x in arg)

        # Generators should still return generators
        if isinstance(arg, SOME_GEN_TYPES):
          return data

        # Cast to numpy array or matrix, if needed, without actually
        # importing its package
        type_arg = type(arg)
        try:
          is_numpy = type_arg.__module__ == "numpy"
        except AttributeError:
          is_numpy = False
        if is_numpy:
          np_type = {"ndarray": sys.modules["numpy"].array,
                     "matrix": sys.modules["numpy"].mat
                    }[type_arg.__name__]
          return np_type(list(data))

        # If it's a Stream, let's use the Stream constructor
        from .lazy_stream import Stream
        if issubclass(type_arg, Stream):
          return Stream(data)

        # Tuple, list, set, dict, deque, etc.. all falls here
        return type_arg(data)

      return func(*args, **kwargs) # wrapper returned value
    return wrapper # elementwise_decorator returned value
  return elementwise_decorator


almost_eq = StrategyDict("almost_eq")


@almost_eq.strategy("bits")
def almost_eq(a, b, bits=32, tol=1, ignore_type=True, pad=0.):
  """
  Almost equal, based on the amount of floating point significand bits.

  Alternative to "a == b" for float numbers and iterables with float numbers,
  and tests for sequence contents (i.e., an elementwise a == b, that also
  works with generators, nested lists, nested generators, etc.). If the type
  of both the contents and the containers should be tested too, set the
  ignore_type keyword arg to False.
  Default version is based on 32 bits IEEE 754 format (23 bits significand).
  Could use 64 bits (52 bits significand) but needs a
  native float type with at least that size in bits.
  If a and b sizes differ, at least one will be padded with the pad input
  value to keep going with the comparison.

  Note
  ----
  Be careful with endless generators!

  """
  if not (ignore_type or type(a) == type(b)):
    return False
  is_it_a = isinstance(a, Iterable)
  is_it_b = isinstance(b, Iterable)
  if is_it_a != is_it_b:
    return False
  if is_it_a:
    return all(almost_eq.bits(ai, bi, bits, tol, ignore_type)
               for ai, bi in xzip_longest(a, b, fillvalue=pad))
  significand = {32: 23, 64: 52, 80: 63, 128: 112
                }[bits] # That doesn't include the sign bit
  power = tol - significand - 1
  return abs(a - b) <= 2 ** power * abs(a + b)


@almost_eq.strategy("diff")
def almost_eq(a, b, max_diff=1e-7, ignore_type=True, pad=0.):
  """
  Almost equal, based on the :math:`|a - b|` value.

  Alternative to "a == b" for float numbers and iterables with float numbers.
  See almost_eq for more information.

  This version based on the non-normalized absolute diff, similar to what
  unittest does with its assertAlmostEquals. If a and b sizes differ, at least
  one will be padded with the pad input value to keep going with the
  comparison.

  Note
  ----
  Be careful with endless generators!

  """
  if not (ignore_type or type(a) == type(b)):
    return False
  is_it_a = isinstance(a, Iterable)
  is_it_b = isinstance(b, Iterable)
  if is_it_a != is_it_b:
    return False
  if is_it_a:
    return all(almost_eq.diff(ai, bi, max_diff, ignore_type)
               for ai, bi in xzip_longest(a, b, fillvalue=pad))
  return abs(a - b) <= max_diff


def sHz(rate):
  """
  Unit conversion constants.

  Useful for casting to/from the default package units (number of samples for
  time and rad/second for frequency). You can use expressions like
  ``440 * Hz`` to get a frequency value, or assign like ``kHz = 1e3 * Hz`` to
  get other unit, as you wish.

  Parameters
  ----------
  rate :
    Sample rate in samples per second

  Returns
  -------
  A tuple ``(s, Hz)``, where ``s`` is the second unit and ``Hz`` is the hertz
  unit, as the number of samples and radians per sample, respectively.

  """
  return float(rate), 2 * pi / rate
