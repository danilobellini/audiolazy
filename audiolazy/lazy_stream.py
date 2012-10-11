#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stream class definition module

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

Created on Sun Jul 22 2012
danilo [dot] bellini [at] gmail [dot] com
"""

import itertools as it
import collections
from functools import wraps

# Audiolazy internal imports
from .lazy_misc import blocks
from .lazy_core import AbstractOperatorOverloaderMeta


class StreamMeta(AbstractOperatorOverloaderMeta):
  """
  Stream metaclass. This class overloads all operators to the Stream class,
  but cmp/rcmp (deprecated), ternary pow (could be called with Stream.map) as
  well as divmod (same as pow, but this will result in a Stream of tuples).
  """
  __operators__ = ("add radd sub rsub mul rmul pow rpow div rdiv mod rmod "
                   "truediv rtruediv floordiv rfloordiv "
                   "pos neg lshift rshift rlshift rrshift "
                   "and rand or ror xor rxor invert "
                   "lt le eq ne gt ge")

  def binary_dunder(cls, op_func):
    def dunder(self, other):
      if isinstance(other, Stream):
        return Stream(it.imap(op_func, self.data, other.data))
      if isinstance(other, collections.Iterable):
        return Stream(it.imap(op_func, self.data, other))
      return Stream(it.imap(lambda a: op_func(a, other), self.data))
    return dunder

  def reverse_binary_dunder(cls, op_func):
    def dunder(self, other):
      if isinstance(other, collections.Iterable):
        return Stream(it.imap(op_func, other, self.data))
      return Stream(it.imap(lambda a: op_func(other, a), self.data))
    return dunder

  def unary_dunder(cls, op_func):
    def dunder(self):
      return Stream(it.imap(op_func, self.data))
    return dunder


class Stream(collections.Iterable):
  """
  Streams generalize generators to allow perform common operations over
  its contents. If you want something like:

    import itertools

    x = itertools.count()
    y = itertools.repeat(3)
    z = 2*x + y

  That's an error: __add__ isn't supported by those types. With this class
  you could call instead:

    x = Stream(itertools.count()) # Iterable
    y = Stream(3) # Non-iterable repeats endlessly
    z = 2*x + y

  And the result is another Stream, that can be seen as a generator that
  yields:

    3, 5, 7, 9, 11, 13, ...

  All operations over Stream objects are lazy and not thread-safe.

  BE CAREFUL:
    In that example, after declaring z as function of x and y, you should
    not use x and y anymore. Use x.tee() or x.copy() instead, if you need
    to use x again otherwhere.
  """
  __metaclass__ = StreamMeta

  def __init__(self, *dargs):
    """
    Constructor for a Stream.

    The parameters should be iterables that will be chained together. If
    they're not iterables, the stream will be an endless repeat of the
    given elements. If any parameter is a generator and its contents is
    used elsewhere, you should use the "tee" (Stream method or itertools
    function) before.

    All operations that works on the elements will work with this iterator
    in a element-wise fashion (like Numpy 1D arrays). When the stream
    sizes differ, the resulting stream have the size of the shortest
    operand.

      Stream([1,2,3]) + Stream([8,5])

    yields the [lazy] finite sequence (9,7), but be careful that:

      Stream(1,2,3) + Stream(8,5)

    yields the [lazy] sequence (9,7,11,6,10,8, 9,7,11,6,10, 9,7,11, ...)
    """
    if len(dargs) == 0:
      raise TypeError("Missing argument(s)")

    elif len(dargs) == 1:
      if isinstance(dargs[0], collections.Iterator):
        self.data = dargs[0]
      elif isinstance(dargs[0], collections.Iterable):
        self.data = iter(dargs[0])
      else:
        self.data = it.repeat(dargs[0])

    else:
      if all(isinstance(arg, collections.Iterable) for arg in dargs):
        self.data = it.chain(*dargs)
      elif not any(isinstance(arg, collections.Iterable) for arg in dargs):
        self.data = it.cycle(dargs)
      else:
        raise TypeError("Input with both iterables and non-iterables")

  def __iter__(self):
    """
    Returns the generator object.
    """
    return self.data

  def __nonzero__(self):
    """
    Boolean value of a stream, called by the bool() built-in and by "if"
    tests. As boolean operators "and", "or" and "not" couldn't be overloaded,
    any trial to cast an instance of this class to a boolean should be seen
    as a mistake.
    """
    raise TypeError("Streams can't be used as booleans.\n"
                    "If you need a boolean stream, try using bitwise "
                    "operators & and | instead of 'and' & 'or'. If using "
                    "'not', you can use the inversion operator ~, casting "
                    "its returned int back to bool.\n"
                    "If you're using it in a 'if' comparison (e.g. for unit "
                    "testing), try to freeze the stream before with "
                    "list(my_stream) or tuple(my_stream).")

  def blocks(self, *args, **kwargs):
    """
    Interface to apply audiolazy.blocks directly in a stream, returning
    another stream. Use keyword args.
    """
    return Stream(blocks(self, *args, **kwargs))

  def take(self, n=None, constructor=list):
    """
    Returns a list with the n first elements. Use this without args if you
    need only one element outside a list. By taking 1 element using n=1 as
    parameter, it will be in a list instead.
    You should avoid using take() as if this would be an iterator. Streams
    are iterables that can be easily part of a "for" loop, and their
    iterators (the ones automatically used in for loops) are
    slightly faster generators. Use iter() builtin if you need that
    instead.
    If there are less than n samples in the stream, raises StopIteration,
    and the at most n-1 last samples are lost.
    """
    if n is None:
      return next(self.data)
    return constructor(next(self.data) for _ in xrange(n))

  def tee(self):
    """
    Returns a "T" (tee) copy of the given stream, allowing the calling
    stream to continue being used.
    """
    a, b = it.tee(self.data) # 2 generators, not thread-safe
    self.data = a
    return Stream(b)

  # Copy is just another useful common name for "tee"
  copy = tee

  def __getattr__(self, name):
    """
    Creates a function that returns stream by applying the given method
    (found by the name string) on each element of the stream.
    """
    return lambda *args, **kwargs: \
             Stream((getattr(a, name)(*args, **kwargs) for a in self.data))

  def append(self, *other):
    """
    Append self with other stream(s). Chaining this way has the behaviour:\n
      self = Stream(self, *others)\n
    """
    self.data = it.chain(self.data, Stream(*other).data)
    return self

  def map(self, func):
    """
    A lazy way to apply the given function to each element in the stream.
    Useful for type casting, like:\n
      Stream(itertools.count()).map(float)\n
    that returns a float counter.
    """
    self.data = it.imap(func, self.data)
    return self

  def filter(self, func):
    """
    A lazy way to skip elements in the stream that gives False for the given
    function.
    """
    self.data = it.ifilter(func, self.data)
    return self


def tostream(func):
  """
  Decorator to convert the function output into a Stream. Useful for
  generators.
  """
  @wraps(func)
  def new_func(*args, **kwargs):
    return Stream(func(*args, **kwargs))
  return new_func


class ControlStream(Stream):
  """
  A Stream that yields a control value that can be changed at any time.
  You just need to set the attribute "value" for doing so, and the next
  value the Stream will yield is the given value.
  """
  def __init__(self, value):
    self.value = value

    def data_generator():
      while True:
        yield self.value

    self.data = data_generator()
