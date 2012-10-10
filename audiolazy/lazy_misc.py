#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common miscellanous tools and constants for general use

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

Created on Fri Jul 20 2012
danilo [dot] bellini [at] gmail [dot] com
"""

import struct
import array
from collections import deque, Iterable
from functools import wraps
import types
import itertools as it
import sys

# Useful constants
DEFAULT_SAMPLE_RATE = 44100 # Hz (samples/second)
DEFAULT_CHUNK_SIZE = 16 #2048 # Samples


def blocks(seq, size=DEFAULT_CHUNK_SIZE, hop=None, padval=0.):
  """
    Generator that gets -size- elements from -seq-, and outputs them in a
    collections.deque (mutable circular queue) sequence container. Next output
    starts -hop- elements after the first element in last output block. Last
    block may be appended with -padval-, if needed to get the desired size.\n
    Note: when the hop is less than size, changing the returned contents
          will keep the new changed value in the next yielded container.
    Seq can have hybrid / hetherogeneous data, it just need to be an iterable.
    You can use other type content as padval (e.g. None) to help segregate the
    padding at the end, if desired.\n
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


def chunks(seq, size=DEFAULT_CHUNK_SIZE, dfmt="f", byte_order=None,
           padval=0.):
  """
    Chunk generator to write any iterable directly in a file.
    The dfmt should be one char, chosen from the ones in link:
      http://docs.python.org/library/struct.html#format-characters
    Useful examples (integer are signed, use upper case for unsigned ones):
      "b" for 8 bits (1 byte) integer
      "h" for 16 bits (2 bytes) integer
      "i" for 32 bits (4 bytes) integer
      "f" for 32 bits (4 bytes) float (default)
      "d" for 64 bits (8 bytes) float (double)
    Byte order follows native system defaults. Other options are in the site:
      http://docs.python.org/library/struct.html#struct-alignment
    They are:
      "<" means little-endian
      ">" means big-endian
  """
  dfmt = str(size) + dfmt
  if byte_order is None:
    struct_string = dfmt
  else:
    struct_string = byte_order + dfmt
  s = struct.Struct(struct_string)
  for block in blocks(seq, size, padval=padval):
    yield s.pack(*block)


def array_chunks(seq, size=DEFAULT_CHUNK_SIZE, dfmt="f", byte_order=None,
                 padval=0.):
  """
    Generator: Another Repetitive Replacement Again Yielding chunks, this is
    an audiolazy.chunks(...) clone using array.array (random access by
    indexing management) instead of struct.Struct and blocks/deque (circular
    queue appending). Try before to find the faster one for your machine.
    Be careful: dfmt symbols for arrays might differ from structs' defaults.
  """
  counter = range(size)
  chunk = array.array(dfmt, counter)
  idx = 0

  for el in seq:
    chunk[idx] = el
    idx += 1
    if idx == size:
      yield chunk.tostring()
      idx = 0

  if idx != 0:
    for idx in range(idx, size):
      chunk[idx] = padval
    yield chunk.tostring()


def zero_pad(seq, left=0, right=0, zero_data=0.):
  """
    Zero padding sample generator.
    Pads the given -seq- with -zero_data- samples before (-left- times) and
    after (-right- times) it, as a generator.
  """
  for _ in xrange(left): yield zero_data
  for item in seq: yield item
  for _ in xrange(right): yield zero_data


def elementwise(parameter=0):
  """
    Creates an elementwise decorator for a parameter other than the first.
    Parameter can be a string (for a keyword argument) or a number (for a
    positional argument).
  """
  if isinstance(parameter, int):
    def elementwise_decorator(func):
      """
        Element-wise decorator for functions known to have 1 input and 1
        output be applied directly on iterables. When made to work with more
        than 1 input, all "secondary" parameters will the same in all
        function calls (i.e., they will not even be a copy).
      """
      @wraps(func)
      def wrapper(*args, **kwargs):
        arg = args[parameter]
        if isinstance(arg, Iterable):
          argsp = list(args[:parameter]) # Since tuple is immutable
          argsn = list(args[parameter+1:])

          # An always-recursive Iterable that must be joined afterwards
          if isinstance(arg, str):
            return "".join(func(*(argsp + [x] + argsn),
                                **kwargs)
                           for x in arg)

          # Generators should still return generators
          if isinstance(arg, (xrange, types.GeneratorType)):
            return (func(*(argsp + [x] + argsn), **kwargs) for x in arg)

          # Cast to numpy array or matrix, without actually importing its
          # package, if needed
          try:
            is_numpy = type(arg).__module__ == "numpy"
          except AttributeError:
            is_numpy = False
          if is_numpy:
            np_type = {"ndarray": sys.modules["numpy"].array,
                       "matrix": sys.modules["numpy"].mat
                      }[type(arg).__name__]
            list_version = [func(*(argsp + [x] + argsn),
                                 **kwargs)
                            for x in arg] # Since numpy isn't graceful with
            return np_type(list_version)  # generators

          # Tuple, list, set, dict, deque, etc.. all falls here
          return type(arg)(func(*(argsp + [x] + argsn),
                                **kwargs)
                           for x in arg)

        return func(*args, **kwargs)
      return wrapper

  elif isinstance(parameter, str):
    def elementwise_decorator(func):
      """
        Element-wise decorator for functions known to have 1 input and 1
        output be applied directly on iterables. When made to work with more
        than 1 input, all "secondary" parameters will the same in all
        function calls (i.e., they will not even be a copy).
      """
      @wraps(func)
      def wrapper(*args, **kwargs):
        arg = kwargs[parameter]
        if isinstance(arg, Iterable):

          # String is an always-recursive Iterable that must be joined
          # afterwards
          if isinstance(arg, str):
            return "".join(func(*args,
                                **dict(kwargs.items() + (parameter, x)))
                           for x in arg)

          # Generators should still return generators
          if isinstance(arg, (xrange, types.GeneratorType)):
            return (func(*args,
                         **dict(kwargs.items() + (parameter, x)))
                    for x in arg)

          # Cast to numpy array or matrix, without actually importing its
          # package, if needed
          try:
            is_numpy = type(arg).__module__ == "numpy"
          except AttributeError:
            is_numpy = False
          if is_numpy:
            np_type = {"ndarray": sys.modules["numpy"].array,
                       "matrix": sys.modules["numpy"].mat
                      }[type(arg).__name__]
            list_version = [func(*args,
                                 **dict(kwargs.items() + (parameter, x)))
                            for x in arg] # Since numpy isn't graceful with
            return np_type(list_version)  # generators

          # Tuple, list, set, dict, deque, etc.. all falls here
          return type(arg)(func(*args,
                                **dict(kwargs.items() + (parameter, x)))
                           for x in arg)

        return func(*args, **kwargs)
      return wrapper

  else:
    raise TypeError("Parameter should be int (positional) or str (keyword)")
  return elementwise_decorator


def almost_eq_diff(a, b, max_diff=1e-7, ignore_type=True, pad=0.):
  """
  Alternative to "a == b" for float numbers and iterables with float numbers.
  See almost_eq for more information. This version based on the non-normalized
  absolute diff, similar to what unittest does.
  If a and b sizes differ, at least one will be padded with the pad input
  value to keep going with the comparison. Be careful with endless generators!
  """
  if not (ignore_type or type(a) == type(b)):
    return False
  is_it_a = isinstance(a, Iterable)
  is_it_b = isinstance(b, Iterable)
  if is_it_a != is_it_b:
    return False
  if is_it_a:
    return all(almost_eq_diff(ai, bi, max_diff, ignore_type)
               for ai, bi in it.izip_longest(a, b, fillvalue=pad))
  return abs(a - b) <= max_diff


def almost_eq(a, b, bits=32, tol=1, ignore_type=True, pad=0.):
  """
  Alternative to "a == b" for float numbers and iterables with float numbers,
  and tests for sequence contents (i.e., an elementwise a == b, that also
  works with generators, nested lists, nested generators, etc.). If the type
  of both the contents and the containers should be tested too, set the
  ignore_type keyword arg to False.
  Default version is based on 32 bits IEEE 754 format (23 bits significand).
  Could use 64 bits (52 bits significand) but needs a
  native float type with at least that size in bits.
  If a and b sizes differ, at least one will be padded with the pad input
  value to keep going with the comparison. Be careful with endless generators!
  """
  if not (ignore_type or type(a) == type(b)):
    return False
  is_it_a = isinstance(a, Iterable)
  is_it_b = isinstance(b, Iterable)
  if is_it_a != is_it_b:
    return False
  if is_it_a:
    return all(almost_eq(ai, bi, bits, tol, ignore_type)
               for ai, bi in it.izip_longest(a, b, fillvalue=pad))
  significand = {32: 23, 64: 52, 80: 63, 128: 112
                }[bits] # That doesn't include the sign bit
  power = tol - significand - 1
  return abs(a - b) <= 2 ** power * abs(a + b)


def multiplication_formatter(power, value, symbol):
  if isinstance(value, float):
    if value.is_integer():
      value = int(value) # Hides ".0" when possible
  if power != 0:
    suffix = "" if power == 1 else "^{p}".format(p=power)
    if value == 1:
      return "{0}{1}".format(symbol, suffix)
    if value == -1:
      return "-{0}{1}".format(symbol, suffix)
    return "{v} * {0}{1}".format(symbol, suffix, v=value)
  else:
    return str(value)


def pair_strings_sum_formatter(a, b):
  """
  Formats the sum of a and b, where both are already strings.
  """
  if b[:1] == "-":
    return "{0} - {1}".format(a, b[1:])
  return "{0} + {1}".format(a, b)