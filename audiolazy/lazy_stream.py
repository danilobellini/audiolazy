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
from collections import Iterable
from functools import wraps
from warnings import warn

# Audiolazy internal imports
from .lazy_misc import blocks
from .lazy_core import AbstractOperatorOverloaderMeta

__all__ = ["StreamMeta", "Stream", "avoid_stream", "tostream",
           "ControlStream", "MemoryLeakWarning", "StreamTeeHub", "thub"]


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

  def __binary__(cls, op_func):
    def dunder(self, other):
      if isinstance(other, cls.__ignored_classes__):
        return NotImplemented
      if isinstance(other, Iterable):
        return Stream(it.imap(op_func, iter(self), iter(other)))
      return Stream(it.imap(lambda a: op_func(a, other), iter(self)))
    return dunder

  def __rbinary__(cls, op_func):
    def dunder(self, other):
      if isinstance(other, cls.__ignored_classes__):
        return NotImplemented
      if isinstance(other, Iterable):
        return Stream(it.imap(op_func, iter(other), iter(self)))
      return Stream(it.imap(lambda a: op_func(other, a), iter(self)))
    return dunder

  def __unary__(cls, op_func):
    def dunder(self):
      return Stream(it.imap(op_func, iter(self)))
    return dunder


class Stream(Iterable):
  """
  Stream class. Stream instances are iterables that can be seem as generators
  with elementwise operators.

  Examples
  --------
  If you want something like:

  >>> import itertools
  >>> x = itertools.count()
  >>> y = itertools.repeat(3)
  >>> z = 2*x + y
  Traceback (most recent call last):
      ...
  TypeError: unsupported operand type(s) for *: 'int' and 'itertools.count'

  That won't work with standard itertools. That's an error, and not only
  __mul__ but also __add__ isn't supported by their types. On the other hand,
  you can use this Stream class:

  >>> x = Stream(itertools.count()) # Iterable
  >>> y = Stream(3) # Non-iterable repeats endlessly
  >>> z = 2*x + y
  >>> z
  <audiolazy.lazy_stream.Stream object at 0x...>
  >>> z.take(12)
  [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]

  If you just want to use your existing code, an "itertools" alternative is
  already done to help you:

  >>> from audiolazy import lazy_itertools as itertools
  >>> x = itertools.count()
  >>> y = itertools.repeat(3)
  >>> z = 2*x + y
  >>> w = itertools.takewhile(lambda (idx, el): idx < 10, enumerate(z))
  >>> list(el for idx, el in w)
  [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

  All operations over Stream objects are lazy and not thread-safe.

  See Also
  --------
  thub :
    "Tee" hub to help using the Streams like numbers in equations and filters.
  tee :
    Just like itertools.tee, but returns a tuple of Stream instances.
  Stream.tee :
    Keeps the Stream usable and returns a copy to be used safely.
  Stream.copy :
    Same to ``Stream.tee``.

  Notes
  -----
  In that example, after declaring z as function of x and y, you should
  not use x and y anymore. Use the thub() or the tee() functions, or
  perhaps the x.tee() or x.copy() Stream methods instead, if you need
  to use x again otherwhere.

  """
  __metaclass__ = StreamMeta
  __ignored_classes__ = tuple()

  def __init__(self, *dargs):
    """
    Constructor for a Stream.

    Parameters
    ----------
    The parameters should be iterables that will be chained together. If
    they're not iterables, the stream will be an endless repeat of the
    given elements. If any parameter is a generator and its contents is
    used elsewhere, you should use the "tee" (Stream method or itertools
    function) before.

    Notes
    -----
    All operations that works on the elements will work with this iterator
    in a element-wise fashion (like Numpy 1D arrays). When the stream
    sizes differ, the resulting stream have the size of the shortest
    operand.

    Examples
    --------
    A finite sequence:

    >>> x = Stream([1,2,3]) + Stream([8,5]) # Finite constructor
    >>> x
    <audiolazy.lazy_stream.Stream object at 0x...>
    >>> tuple(x)
    (9, 7)

    But be careful:

    >>> x = Stream(1,2,3) + Stream(8,5) # Periodic constructor
    >>> x
    <audiolazy.lazy_stream.Stream object at 0x...>
    >>> x.take(15) # Don't try "tuple" or "list": this Stream is endless!
    [9, 7, 11, 6, 10, 8, 9, 7, 11, 6, 10, 8, 9, 7, 11]

    """
    if len(dargs) == 0:
      raise TypeError("Missing argument(s)")

    elif len(dargs) == 1:
      if isinstance(dargs[0], Iterable):
        self._data = iter(dargs[0])
      else:
        self._data = it.repeat(dargs[0])

    else:
      if all(isinstance(arg, Iterable) for arg in dargs):
        self._data = it.chain(*dargs)
      elif not any(isinstance(arg, Iterable) for arg in dargs):
        self._data = it.cycle(dargs)
      else:
        raise TypeError("Input with both iterables and non-iterables")

  def __iter__(self):
    """ Returns the Stream contents iterator. """
    return self._data

  def __nonzero__(self):
    """
    Boolean value of a stream, called by the bool() built-in and by "if"
    tests. As boolean operators "and", "or" and "not" couldn't be overloaded,
    any trial to cast an instance of this class to a boolean should be seen
    as a mistake.
    """
    raise TypeError("Streams can't be used as booleans.\n"
                    "If you need a boolean stream, try using bitwise "
                    "operators & and | instead of 'and' and 'or'. If using "
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
    iterators (the ones automatically used in for loops) are slightly faster.
    Use iter() builtin if you need that, instead, or perhaps the blocks
    method.
    If there are less than n samples in the stream, raises StopIteration,
    and at most n-1 last samples can be lost.
    """
    if n is None:
      return next(self._data)
    return constructor(next(self._data) for _ in xrange(n))

  def tee(self):
    """
    Returns a "T" (tee) copy of the given stream, allowing the calling
    stream to continue being used.
    """
    a, b = it.tee(self._data) # 2 generators, not thread-safe
    self._data = a
    return Stream(b)

  # Copy is just another useful common name for "tee"
  copy = tee

  def __getattr__(self, name):
    """
    Returns a Stream of attributes or methods, got in an elementwise fashion.
    """
    if name == "next":
      raise NotImplementedError("Streams are iterable, not iterators")
    return Stream(getattr(a, name) for a in self._data)

  def __call__(self, *args, **kwargs):
    """
    Returns the results from calling elementwise (where each element is
    assumed to be callable), with the same arguments.
    """
    return Stream(a(*args, **kwargs) for a in self._data)

  def append(self, *other):
    """
    Append self with other stream(s). Chaining this way has the behaviour:

      ``self = Stream(self, *others)``

    """
    self._data = it.chain(self._data, Stream(*other)._data)
    return self

  def map(self, func):
    """
    A lazy way to apply the given function to each element in the stream.
    Useful for type casting, like:

    >>> from audiolazy import count
    >>> count().take(5)
    [0, 1, 2, 3, 4]
    >>> my_stream = count().map(float)
    >>> my_stream.take(5) # A float counter
    [0.0, 1.0, 2.0, 3.0, 4.0]

    """
    self._data = it.imap(func, self._data)
    return self

  def filter(self, func):
    """
    A lazy way to skip elements in the stream that gives False for the given
    function.
    """
    self._data = it.ifilter(func, self._data)
    return self

  @classmethod
  def register_ignored_class(cls, ignore):
    cls.__ignored_classes__ += (ignore,)


def avoid_stream(cls):
  """
  Decorator to a class whose instances should avoid casting to a Stream when
  used with operators applied to them.
  """
  Stream.register_ignored_class(cls)
  return cls


def tostream(func):
  """
  Decorator to convert the function output into a Stream. Useful for
  generator functions.
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

  Examples
  --------

  >>> cs = ControlStream(7)
  >>> data = Stream(1, 3) # [1, 3, 1, 3, 1, 3, ...] endless iterable
  >>> res = data + cs
  >>> res.take(5)
  [8, 10, 8, 10, 8]
  >>> cs.value = 9
  >>> res.take(5)
  [12, 10, 12, 10, 12]

  """
  def __init__(self, value):
    self.value = value

    def data_generator():
      while True:
        yield self.value

    super(ControlStream, self).__init__(data_generator())


class MemoryLeakWarning(Warning):
  """ A warning to be used when a memory leak is detected. """


class StreamTeeHub(Stream):
  """
  A Stream that returns a different iterator each time it is used.

  See Also
  --------
  thub :
    Auto-copy "tee hub" and helpful constructor alternative for this class.

  """
  def __init__(self, data, n):
    super(StreamTeeHub, self).__init__(data)
    iter_self = super(StreamTeeHub, self).__iter__()
    self._iters = list(it.tee(iter_self, n))

  def __iter__(self):
    try:
      return self._iters.pop()
    except IndexError:
      raise IndexError("StreamTeeHub has no more copies left to use.")

  def __del__(self):
    if self._iters != []:
      warn("StreamTeeHub requesting {0} more copies than "
           "needed".format(len(self._iters)), MemoryLeakWarning)


def thub(data, n):
  """
  Tee or "T" hub auto-copier to help working with Stream instances as well as
  with numbers.

  Parameters
  ----------
  data :
    Input to be copied. Can be anything.
  n :
    Number of copies.

  Returns
  -------
  A StreamTeeHub instance, if input data is iterable.
  The data itself, otherwise.

  Examples
  --------

  >>> def sub_sum(x, y):
  ...     x = thub(x, 2) # Casts to StreamTeeHub, when needed
  ...     y = thub(y, 2)
  ...     return (x - y) / (x + y) # Return type might be number or Stream

  With numbers:

  >>> sub_sum(1, 1)
  0

  Combining number with iterable:

  >>> sub_sum(3., [1, 2, 3])
  <audiolazy.lazy_stream.Stream object at 0x...>
  >>> list(sub_sum(3., [1, 2, 3]))
  [0.5, 0.2, 0.0]

  Both iterables (the Stream input behaves like an endless [6, 1, 6, 1, ...]):

  >>> list(sub_sum([4., 3., 2., 1.], [1, 2, 3]))
  [0.6, 0.2, -0.2]
  >>> list(sub_sum([4., 3., 2., 1.], Stream(6, 1)))
  [-0.2, 0.5, -0.5, 0.0]

  This function can also be used as a an alternative to the Stream
  constructor when your function has only one parameter, to avoid casting
  when that's not needed:

  >>> func = lambda x: 250 * thub(x, 1)
  >>> func(1)
  250
  >>> func([2] * 10)
  <audiolazy.lazy_stream.Stream object at 0x...>
  >>> func([2] * 10).take(5)
  [500, 500, 500, 500, 500]

  """
  return StreamTeeHub(data, n) if isinstance(data, Iterable) else data
