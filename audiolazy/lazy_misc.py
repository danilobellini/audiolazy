# -*- coding: utf-8 -*-
# This file is part of AudioLazy, the signal processing Python package.
# Copyright (C) 2012 Danilo de Jesus da Silva Bellini
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

import struct
import array
from collections import deque, Iterable
from functools import wraps
import types
import itertools as it
import sys
from math import pi
from fractions import Fraction


__all__ = ["DEFAULT_SAMPLE_RATE", "DEFAULT_CHUNK_SIZE", "LATEX_PI_SYMBOL",
           "blocks", "chunks", "array_chunks", "zero_pad", "elementwise",
           "almost_eq_diff", "almost_eq", "multiplication_formatter",
           "pair_strings_sum_formatter", "rational_formatter",
           "pi_formatter", "auto_formatter", "rst_table", "small_doc", "sHz"]

# Useful constants
DEFAULT_SAMPLE_RATE = 44100 # Hz (samples/second)
DEFAULT_CHUNK_SIZE = 2048 # Samples
LATEX_PI_SYMBOL = r"$\pi$"


def blocks(seq, size=DEFAULT_CHUNK_SIZE, hop=None, padval=0.):
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


def chunks(seq, size=DEFAULT_CHUNK_SIZE, dfmt="f", byte_order=None,
           padval=0.):
  """
  Chunk generator, or a blockenizer for homogeneous data, to help writing an
  iterable into a file. This implementation is based on the struct module.

  The dfmt should be one char, chosen from the ones in link:

    `<http://docs.python.org/library/struct.html#format-characters>`_

  Useful examples (integer are signed, use upper case for unsigned ones):

  - "b" for 8 bits (1 byte) integer
  - "h" for 16 bits (2 bytes) integer
  - "i" for 32 bits (4 bytes) integer
  - "f" for 32 bits (4 bytes) float (default)
  - "d" for 64 bits (8 bytes) float (double)

  Byte order follows native system defaults. Other options are in the site:

    `<http://docs.python.org/library/struct.html#struct-alignment>`_

  They are:

  - "<" means little-endian
  - ">" means big-endian

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
  Chunk generator based on the array module (Python standard library).

  Generator: Another Repetitive Replacement Again Yielding chunks, this is
  an audiolazy.chunks(...) clone using array.array (random access by
  indexing management) instead of struct.Struct and blocks/deque (circular
  queue appending). Try before to find the faster one for your machine.

  Note
  ----
  The ``dfmt`` symbols for arrays might differ from structs' defaults.

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

      if isinstance(arg, Iterable) and not isinstance(arg, (str, unicode)):
        if positional:
          data = (func(*(args[:pos] + (x,) + args[pos+1:]),
                       **kwargs)
                  for x in arg)
        else:
          data = (func(*args,
                       **dict(kwargs.items() + (name, x)))
                  for x in arg)

        # Generators should still return generators
        if isinstance(arg, (xrange, types.GeneratorType)):
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


def almost_eq_diff(a, b, max_diff=1e-7, ignore_type=True, pad=0.):
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
    return all(almost_eq_diff(ai, bi, max_diff, ignore_type)
               for ai, bi in it.izip_longest(a, b, fillvalue=pad))
  return abs(a - b) <= max_diff


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
    return all(almost_eq(ai, bi, bits, tol, ignore_type)
               for ai, bi in it.izip_longest(a, b, fillvalue=pad))
  significand = {32: 23, 64: 52, 80: 63, 128: 112
                }[bits] # That doesn't include the sign bit
  power = tol - significand - 1
  return abs(a - b) <= 2 ** power * abs(a + b)


def multiplication_formatter(power, value, symbol):
  """
  Formats a number ``value * symbol ** power`` as a string, where symbol is
  already a string and both other inputs are numbers.

  """
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
  Formats the sum of a and b, where both are numbers already converted
  to strings.

  """
  if b[:1] == "-":
    return "{0} - {1}".format(a, b[1:])
  return "{0} + {1}".format(a, b)


@elementwise("value", 0)
def rational_formatter(value, symbol_str="", symbol_value=1, after=False,
                       max_denominator=1000000):
  """
  Converts a given numeric value to a string based on rational fractions of
  the given symbol, useful for labels in plots.

  Parameters
  ----------
  value :
    A float number or an iterable with floats.
  symbol_str :
    String data that will be in the output representing the data as a
    numerator multiplier, if needed. Defaults to an empty string.
  symbol_value :
    The conversion value for the given symbol (e.g. pi = 3.1415...). Defaults
    to one (no effect).
  after :
    Chooses the place where the symbol_str should be written. If True, that's
    the end of the string. If False, that's in between the numerator and the
    denominator, before the slash. Defaults to False.
  max_denominator :
    An int instance, used to round the float following the given limit.
    Defaults to the integer 1,000,000 (one million).

  Returns
  -------
  A string with the rational number written into, with or without the symbol.

  Examples
  --------
  >>> rational_formatter(12.5)
  '25/2'
  >>> rational_formatter(0.333333333333333)
  '1/3'
  >>> rational_formatter(0.333)
  '333/1000'
  >>> rational_formatter(0.333, max_denominator=100)
  '1/3'
  >>> rational_formatter(0.125, symbol_str="steps")
  'steps/8'
  >>> rational_formatter(0.125, symbol_str=" Hz",
  ...                    after=True) # The symbol includes whitespace!
  '1/8 Hz'

  See Also
  --------
  pi_formatter :
    Curried rational_formatter for the "pi" symbol.

  """
  if value == 0:
    return "0"

  frac = Fraction(value/symbol_value).limit_denominator(max_denominator)
  num, den = frac.numerator, frac.denominator

  output_data = []

  if num < 0:
    num = -num
    output_data.append("-")

  if (num != 1) or (symbol_str == "") or after:
    output_data.append(str(num))

  if (value != 0) and not after:
    output_data.append(symbol_str)

  if den != 1:
    output_data.extend(["/", str(den)])

  if after:
    output_data.append(symbol_str)

  return "".join(output_data)


def pi_formatter(value, after=False, max_denominator=1000000):
  """
  String formatter for fractions of :math:`\pi`.

  Alike the rational_formatter, but fixed to the symbol string
  LATEX_PI_SYMBOL and symbol value ``pi``, for direct use with MatPlotLib
  labels.

  See Also
  --------
  rational_formatter :
    Float to string conversion, perhaps with a symbol as a multiplier.

  """
  return rational_formatter(value, symbol_str=LATEX_PI_SYMBOL,
                            symbol_value=pi, after=after,
                            max_denominator=max_denominator)


def auto_formatter(value, order="pprpr", size=[4, 5, 3, 6, 4],
                   after=False, max_denominator=1000000):
  """
  Automatic string formatter for integer fractions, fractions of :math:`\pi`
  and float numbers with small number of digits.

  Chooses between pi_formatter, rational_formatter without a symbol and
  a float representation by counting each digit, the "pi" symbol and the
  slash as one char each, trying in the given ``order`` until one gets at
  most the given ``size`` limit parameter as its length.

  Parameters
  ----------
  value :
    A float number or an iterable with floats.
  order :
    A string that gives the order to try formatting. Each char should be:

    - "p" for pi_formatter;
    - "r" for rational_formatter without symbol;
    - "f" for the float representation.

    Defaults to "pprpr". If no trial has the desired size, returns the
    float representation.
  size :
    The max size allowed for each formatting in the ``order``, respectively.
    Defaults to [4, 5, 3, 6, 4].
  after :
    Chooses the place where the LATEX_PI_SYMBOL symbol, if that's the case.
    If True, that's the end of the string. If False, that's in between the
    numerator and the denominator, before the slash. Defaults to False.
  max_denominator :
    An int instance, used to round the float following the given limit.
    Defaults to the integer 1,000,000 (one million).

  Returns
  -------
  A string with the number written into.

  Note
  ----
  You probably want to keep ``max_denominator`` high to avoid rounding.

  """
  if len(order) != len(size):
    raise ValueError("Arguments 'order' and 'size' should have the same size")

  str_data = {
    "p": pi_formatter(value, after=after, max_denominator=max_denominator),
    "r": rational_formatter(value, max_denominator=max_denominator),
    "f": elementwise("v", 0)(lambda v: "{0:g}".format(v))(value)
  }

  sizes = {k: len(v) for k, v in str_data.iteritems()}
  sizes["p"] = max(1, sizes["p"] - len(LATEX_PI_SYMBOL) + 1)

  for char, max_size in it.izip(order, size):
    if sizes[char] <= max_size:
      return str_data[char]
  return str_data["f"]


def rst_table(data, schema=None):
  """
  Creates a reStructuredText simple table (list of strings) from a list of
  lists.
  """
  # Process multi-rows (replaced by rows with empty columns when needed)
  pdata = []
  for row in data:
    prow = [el if isinstance(el, list) else [el] for el in row]
    pdata.extend(pr for pr in it.izip_longest(*prow, fillvalue=""))

  # Find the columns sizes
  sizes = [max(len("{0}".format(el)) for el in column)
           for column in it.izip(*pdata)]
  sizes = [max(size, len(sch)) for size, sch in it.izip(sizes, schema)]

  # Creates the title and border rows
  if schema is None:
    schema = pdata[0]
    pdata = pdata[1:]
  border = " ".join("=" * size for size in sizes)
  titles = " ".join("{1:^{0}}".format(*pair)
                    for pair in it.izip(sizes, schema))

  # Creates the full table and returns
  rows = [border, titles, border]
  rows.extend(" ".join("{1:<{0}}".format(*pair)
                       for pair in it.izip(sizes, row))
              for row in pdata)
  rows.append(border)
  return rows


def small_doc(obj, indent="", max_width=80):
  """
  Finds a useful small doc representation of an object.

  Parameters
  ----------
  obj :
    Any object, which the documentation representation should be taken from.
  indent :
    Result indentation string to be insert in front of all lines.
  max_width :
    Each line of the result may have at most this length.

  Returns
  -------
  For classes, modules, functions, methods, properties and StrategyDict
  instances, returns the first paragraph in the doctring of the given object,
  as a list of strings, stripped at right and with indent at left.
  For other inputs, it will use themselves cast to string as their docstring.

  """
  # Not something that normally have a docstring
  from .lazy_core import StrategyDict
  if not isinstance(obj, (StrategyDict, types.FunctionType, types.MethodType,
                          types.ModuleType, type, property)):
    data = [el.strip() for el in str(obj).splitlines()]
    if len(data) == 1:
      if data[0].startswith("<audiolazy.lazy_"): # Instance
        data = data[0].split("0x", -1)[0] + "0x...>" # Hide its address
      else:
        data = "".join(["``", data[0], "``"])
    else:
      data == " ".join(data)

  # No docstring
  elif (not obj.__doc__) or (obj.__doc__.strip() == ""):
    data = "\ * * * * ...no docstring... * * * * \ "

  # Docstring
  else:
    data = (el.strip() for el in obj.__doc__.strip().splitlines())
    data = " ".join(it.takewhile(lambda el: el != "", data))

  # Ensure max_width (word wrap)
  max_width -= len(indent)
  result = []
  for word in data.split():
    if len(word) <= max_width:
      if result:
        if len(result[-1]) + len(word) + 1 <= max_width:
          word = " ".join([result.pop(), word])
        result.append(word)
      else:
        result = [word]
    else: # Splits big words
      result.extend("".join(w) for w in blocks(word, max_width, padval=""))

  # Apply indentation and finishes
  return [indent + el for el in result]


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
