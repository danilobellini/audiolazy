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
Strings, reStructuredText, docstrings and other general text processing
"""

from __future__ import division

import itertools as it
from fractions import Fraction

# Audiolazy internal imports
from .lazy_compat import xzip, xzip_longest, iteritems
from .lazy_misc import rint, elementwise, blocks
from .lazy_core import StrategyDict
from .lazy_math import pi

__all__ = ["multiplication_formatter", "pair_strings_sum_formatter",
           "float_str", "rst_table", "small_doc", "format_docstring"]


def multiplication_formatter(power, value, symbol):
  """
  Formats a ``value * symbol ** power`` as a string.

  Usually ``symbol`` is already a string and both other inputs are numbers,
  however this isn't strictly needed. If ``symbol`` is a number, the
  multiplication won't be done, keeping its default string formatting as is.

  """
  if isinstance(value, float):
    if value.is_integer():
      value = rint(value) # Hides ".0" when possible
    else:
      value = "{:g}".format(value)
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
  Formats the sum of a and b.

  Note
  ----
  Both inputs are numbers already converted to strings.

  """
  if b[:1] == "-":
    return "{0} - {1}".format(a, b[1:])
  return "{0} + {1}".format(a, b)


float_str = StrategyDict("float_str")
float_str.__class__.pi_symbol = r"$\pi$"
float_str.__class__.pi_value = pi


@float_str.strategy("auto")
def float_str(value, order="pprpr", size=[4, 5, 3, 6, 4],
              after=False, max_denominator=1000000):
  """
  Pretty string from int/float.

  "Almost" automatic string formatter for integer fractions, fractions of
  :math:`\pi` and float numbers with small number of digits.

  Outputs a representation among ``float_str.pi``, ``float_str.frac`` (without
  a symbol) strategies, as well as the usual float representation. The
  formatter is chosen by counting the resulting length, trying each one in the
  given ``order`` until one gets at most the given ``size`` limit parameter as
  its length.

  Parameters
  ----------
  value :
    A float number or an iterable with floats.
  order :
    A string that gives the order to try formatting. Each char should be:

    - ``"p"`` for pi formatter (``float_str.pi``);
    - ``"r"`` for ratio without symbol (``float_str.frac``);
    - ``"f"`` for the float usual base 10 decimal representation.

    Defaults to ``"pprpr"``. If no trial has the desired size, returns the
    float representation.
  size :
    The max size allowed for each formatting in the ``order``, respectively.
    Defaults to ``[4, 5, 3, 6, 4]``.
  after :
    Chooses the place where the :math:`\pi` symbol should appear, when such
    formatter apply. If ``True``, that's the end of the string. If ``False``,
    that's in between the numerator and the denominator, before the slash.
    Defaults to ``False``.
  max_denominator :
    The data in ``value`` is rounded following the limit given by this
    parameter when trying to represent it as a fraction/ratio.
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
    "p": float_str.pi(value, after=after, max_denominator=max_denominator),
    "r": float_str.frac(value, max_denominator=max_denominator),
    "f": elementwise("v", 0)(lambda v: "{0:g}".format(v))(value)
  }

  sizes = {k: len(v) for k, v in iteritems(str_data)}
  sizes["p"] = max(1, sizes["p"] - len(float_str.pi_symbol) + 1)

  for char, max_size in xzip(order, size):
    if sizes[char] <= max_size:
      return str_data[char]
  return str_data["f"]


@float_str.strategy("frac", "fraction", "ratio", "rational")
@elementwise("value", 0)
def float_str(value, symbol_str="", symbol_value=1, after=False,
              max_denominator=1000000):
  """
  Pretty rational string from float numbers.

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
    Chooses the place where the ``symbol_str`` should be written. If ``True``,
    that's the end of the string. If ``False``, that's in between the
    numerator and the denominator, before the slash. Defaults to ``False``.
  max_denominator :
    An int instance, used to round the float following the given limit.
    Defaults to the integer 1,000,000 (one million).

  Returns
  -------
  A string with the rational number written into as a fraction, with or
  without a multiplying symbol.

  Examples
  --------
  >>> float_str.frac(12.5)
  '25/2'
  >>> float_str.frac(0.333333333333333)
  '1/3'
  >>> float_str.frac(0.333)
  '333/1000'
  >>> float_str.frac(0.333, max_denominator=100)
  '1/3'
  >>> float_str.frac(0.125, symbol_str="steps")
  'steps/8'
  >>> float_str.frac(0.125, symbol_str=" Hz",
  ...                after=True) # The symbol includes whitespace!
  '1/8 Hz'

  See Also
  --------
  float_str.pi :
    This fraction/ratio formatter, but configured with the "pi" symbol.

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


@float_str.strategy("pi")
def float_str(value, after=False, max_denominator=1000000):
  """
  String formatter for fractions of :math:`\pi`.

  Alike the rational_formatter, but fixed to the symbol string
  ``float_str.pi_symbol`` and value ``float_str.pi_value`` (both can be
  changed, if needed), mainly intended for direct use with MatPlotLib labels.

  Examples
  --------
  >>> float_str.pi_symbol = "pi" # Just for printing sake
  >>> float_str.pi(pi / 2)
  'pi/2'
  >>> float_str.pi(pi * .333333333333333)
  'pi/3'
  >>> float_str.pi(pi * .222222222222222)
  '2pi/9'
  >>> float_str.pi_symbol = " PI" # With the space
  >>> float_str.pi(pi / 2, after=True)
  '1/2 PI'
  >>> float_str.pi(pi * .333333333333333, after=True)
  '1/3 PI'
  >>> float_str.pi(pi * .222222222222222, after=True)
  '2/9 PI'

  See Also
  --------
  float_str.frac :
    Float to string conversion, perhaps with a symbol as a multiplier.

  """
  return float_str.frac(value, symbol_str=float_str.pi_symbol,
                        symbol_value=float_str.pi_value, after=after,
                        max_denominator=max_denominator)


def rst_table(data, schema=None):
  """
  Creates a reStructuredText simple table (list of strings) from a list of
  lists.
  """
  # Process multi-rows (replaced by rows with empty columns when needed)
  pdata = []
  for row in data:
    prow = [el if isinstance(el, list) else [el] for el in row]
    pdata.extend(pr for pr in xzip_longest(*prow, fillvalue=""))

  # Find the columns sizes
  sizes = [max(len("{0}".format(el)) for el in column)
           for column in xzip(*pdata)]
  sizes = [max(size, len(sch)) for size, sch in xzip(sizes, schema)]

  # Creates the title and border rows
  if schema is None:
    schema = pdata[0]
    pdata = pdata[1:]
  border = " ".join("=" * size for size in sizes)
  titles = " ".join("{1:^{0}}".format(*pair)
                    for pair in xzip(sizes, schema))

  # Creates the full table and returns
  rows = [border, titles, border]
  rows.extend(" ".join("{1:<{0}}".format(*pair)
                       for pair in xzip(sizes, row))
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
  if not getattr(obj, "__doc__", False):
    data = [el.strip() for el in str(obj).splitlines()]
    if len(data) == 1:
      if data[0].startswith("<audiolazy.lazy_"): # Instance
        data = data[0].split("0x", -1)[0] + "0x...>" # Hide its address
      else:
        data = "".join(["``", data[0], "``"])
    else:
      data = " ".join(data)

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


def format_docstring(template_="{__doc__}", *args, **kwargs):
  r"""
  Parametrized decorator for adding/changing a function docstring.

  For changing a already available docstring in the function, the
  ``"{__doc__}"`` in the template is replaced by the original function
  docstring.

  Parameters
  ----------
  template_ :
    A format-style template.
  *args, **kwargs :
    Positional and keyword arguments passed to the formatter.

  Examples
  --------
  Closure docstring personalization:

  >>> def add(n):
  ...   @format_docstring(number=n)
  ...   def func(m):
  ...     '''Adds {number} to the given value.'''
  ...     return n + m
  ...   return func
  >>> add(3).__doc__
  'Adds 3 to the given value.'
  >>> add("__").__doc__
  'Adds __ to the given value.'

  Same but using a lambda (you can also try with ``**locals()``):

  >>> def add_with_lambda(n):
  ...   return format_docstring("Adds {0}.", n)(lambda m: n + m)
  >>> add_with_lambda(15).__doc__
  'Adds 15.'
  >>> add_with_lambda("something").__doc__
  'Adds something.'

  Mixing both template styles with ``{__doc__}``:

  >>> templ = "{0}, {1} is my {name} docstring:{__doc__}->\nEND!"
  >>> @format_docstring(templ, "zero", "one", "two", name="testing", k=[1, 2])
  ... def test():
  ...   '''
  ...   Not empty!
  ...   {2} != {k[0]} but {2} == {k[1]}
  ...   '''
  >>> print(test.__doc__)
  zero, one is my testing docstring:
    Not empty!
    two != 1 but two == 2
    ->
  END!
  """
  def decorator(func):
    if func.__doc__:
      kwargs["__doc__"] = func.__doc__.format(*args, **kwargs)
    func.__doc__ = template_.format(*args, **kwargs)
    return func
  return decorator
