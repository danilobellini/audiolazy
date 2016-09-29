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
Stream class definition module
"""

import itertools as it
from collections import Iterable, deque
from functools import wraps
import warnings
from math import isinf

# Audiolazy internal imports
from .lazy_misc import blocks, rint
from .lazy_compat import meta, xrange, xmap, xfilter, NEXT_NAME
from .lazy_core import AbstractOperatorOverloaderMeta
from .lazy_math import inf

__all__ = ["StreamMeta", "Stream", "avoid_stream", "tostream",
           "ControlStream", "MemoryLeakWarning", "StreamTeeHub", "thub",
           "Streamix"]


class StreamMeta(AbstractOperatorOverloaderMeta):
  """
  Stream metaclass. This class overloads all operators to the Stream class,
  but cmp/rcmp (deprecated), ternary pow (could be called with Stream.map) as
  well as divmod (same as pow, but this will result in a Stream of tuples).
  """
  def __binary__(cls, op):
    op_func = op.func
    def dunder(self, other):
      if isinstance(other, cls.__ignored_classes__):
        return NotImplemented
      if isinstance(other, Iterable):
        return Stream(xmap(op_func, iter(self), iter(other)))
      return Stream(xmap(lambda a: op_func(a, other), iter(self)))
    return dunder

  def __rbinary__(cls, op):
    op_func = op.func
    def dunder(self, other):
      if isinstance(other, cls.__ignored_classes__):
        return NotImplemented
      if isinstance(other, Iterable):
        return Stream(xmap(op_func, iter(other), iter(self)))
      return Stream(xmap(lambda a: op_func(other, a), iter(self)))
    return dunder

  def __unary__(cls, op):
    op_func = op.func
    def dunder(self):
      return Stream(xmap(op_func, iter(self)))
    return dunder


class Stream(meta(Iterable, metaclass=StreamMeta)):
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
  TypeError: unsupported operand type(s) for *: 'int' and ...

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
  >>> w = itertools.takewhile(lambda pair: pair[0] < 10, enumerate(z))
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
  __ignored_classes__ = tuple()

  def __init__(self, *dargs):
    """
    Constructor for a Stream.

    Parameters
    ----------
    *dargs:
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

  def __bool__(self):
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

  __nonzero__ = __bool__ # For Python 2.x compatibility

  def blocks(self, *args, **kwargs):
    """
    Interface to apply audiolazy.blocks directly in a stream, returning
    another stream. Use keyword args.
    """
    return Stream(blocks(iter(self), *args, **kwargs))

  def take(self, n=None, constructor=list):
    """
    Returns a container with the n first elements from the Stream, or less if
    there aren't enough. Use this without args if you need only one element
    outside a list.

    Parameters
    ----------
    n :
      Number of elements to be taken. Defaults to None.
      Rounded when it's a float, and this can be ``inf`` for taking all.
    constructor :
      Container constructor function that can receie a generator as input.
      Defaults to ``list``.

    Returns
    -------
    The first ``n`` elements of the Stream sequence, created by the given
    constructor unless ``n == None``, which means returns the next element
    from the sequence outside any container.
    If ``n`` is None, this can raise StopIteration due to lack of data in
    the Stream. When ``n`` is a number, there's no such exception.

    Examples
    --------
    >>> Stream(5).take(3) # Three elements
    [5, 5, 5]
    >>> Stream(1.2, 2, 3).take() # One element, outside a container
    1.2
    >>> Stream(1.2, 2, 3).take(1) # With n = 1 argument, it'll be in a list
    [1.2]
    >>> Stream(1.2, 2, 3).take(1, constructor=tuple) # Why not a tuple?
    (1.2,)
    >>> Stream([1, 2]).take(3) # More than the Stream size, n is integer
    [1, 2]
    >>> Stream([]).take() # More than the Stream size, n is None
    Traceback (most recent call last):
      ...
    StopIteration

    Taking rounded float quantities and "up to infinity" elements
    (don't try using ``inf`` with endless Stream instances):

    >>> Stream([4, 3, 2, 3, 2]).take(3.4)
    [4, 3, 2]
    >>> Stream([4, 3, 2, 3, 2]).take(3.6)
    [4, 3, 2, 3]
    >>> Stream([4, 3, 2, 3, 2]).take(inf)
    [4, 3, 2, 3, 2]

    See Also
    --------
    Stream.peek :
      Returns the n first elements from the Stream, without removing them.

    Note
    ----
    You should avoid using take() as if this would be an iterator. Streams
    are iterables that can be easily part of a "for" loop, and their
    iterators (the ones automatically used in for loops) are slightly faster.
    Use iter() builtin if you need that, instead, or perhaps the blocks
    method.

    """
    if n is None:
      return next(self._data)
    if isinf(n) and n > 0:
      return constructor(self._data)
    if isinstance(n, float):
      n = rint(n) if n > 0 else 0 # So this works with -inf and nan
    return constructor(next(self._data) for _ in xrange(n))

  def copy(self):
    """
    Returns a "T" (tee) copy of the given stream, allowing the calling
    stream to continue being used.
    """
    a, b = it.tee(self._data) # 2 generators, not thread-safe
    self._data = a
    return Stream(b)

  def peek(self, n=None, constructor=list):
    """
    Sees/peeks the next few items in the Stream, without removing them.

    Besides that this functions keeps the Stream items, it's the same to the
    ``Stream.take()`` method.

    See Also
    --------
    Stream.take :
      Returns the n first elements from the Stream, removing them.

    Note
    ----
    When applied in a StreamTeeHub, this method doesn't consume a copy.
    Data evaluation is done only once, i.e., after peeking the data is simply
    stored to be yielded again when asked for.

    """
    return self.copy().take(n=n, constructor=constructor)

  def skip(self, n):
    """
    Throws away the first ``n`` values from the Stream.

    Note
    ----
    Performs the evaluation lazily, i.e., the values are thrown away only
    after requesting the next value.

    """
    def skipper(data):
      for _ in xrange(int(round(n))):
        next(data)
      for el in data:
        yield el

    self._data = skipper(self._data)
    return self

  def limit(self, n):
    """
    Enforces the Stream to finish after ``n`` items.
    """
    data = self._data
    self._data = (next(data) for _ in xrange(int(round(n))))
    return self

  def __getattr__(self, name):
    """
    Returns a Stream of attributes or methods, got in an elementwise fashion.
    """
    if name == NEXT_NAME:
      raise AttributeError("Streams are iterable, not iterators")
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
    self._data = xmap(func, self._data)
    return self

  def filter(self, func):
    """
    A lazy way to skip elements in the stream that gives False for the given
    function.
    """
    self._data = xfilter(func, self._data)
    return self

  @classmethod
  def register_ignored_class(cls, ignore):
    cls.__ignored_classes__ += (ignore,)

  def __abs__(self):
    return self.map(abs)


def avoid_stream(cls):
  """
  Decorator to a class whose instances should avoid casting to a Stream when
  used with operators applied to them.
  """
  Stream.register_ignored_class(cls)
  return cls


def tostream(func, module_name=None):
  """
  Decorator to convert the function output into a Stream. Useful for
  generator functions.

  Note
  ----
  Always use the ``module_name`` input when "decorating" a function that was
  defined in other module.

  """
  @wraps(func)
  def new_func(*args, **kwargs):
    return Stream(func(*args, **kwargs))
  if module_name is not None:
    new_func.__module__ = module_name
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
    if self._iters:
      msg_fmt = "StreamTeeHub requesting {0} more copies than needed"
      msg = msg_fmt.format(len(self._iters))
      warnings.warn(MemoryLeakWarning(msg))
      self._iters[:] = [] # Avoid many warnings for many calls to __del__

  def take(self, *args, **kwargs):
    """
    Fake function just to avoid using inherited Stream.take implicitly.

    Warning
    -------
    You shouldn't need to call this method directly.
    If you need a Stream instance to work progressively changing it, try:

    >>> data = thub([1, 2, 3], 2) # A StreamTeeHub instance
    >>> first_copy = Stream(data)
    >>> first_copy.take(2)
    [1, 2]
    >>> list(data) # Gets the second copy
    [1, 2, 3]
    >>> first_copy.take()
    3

    If you just want to see the first few values, try
    ``self.peek(*args, **kwargs)`` instead.

    >>> data = thub((9, -1, 0, 4), 2) # StreamTeeHub instance
    >>> data.peek()
    9
    >>> data.peek(3)
    [9, -1, 0]
    >>> list(data) # First copy
    [9, -1, 0, 4]
    >>> data.peek(1)
    [9]
    >>> second_copy = Stream(data)
    >>> second_copy.peek(2)
    [9, -1]
    >>> data.peek() # There's no third copy
    Traceback (most recent call last):
        ...
    IndexError: StreamTeeHub has no more copies left to use.

    If you want to consume from every StreamTeeHub copy, you probably
    should change your code before calling the ``thub()``,
    but you still might use:

    >>> data = thub(Stream(1, 2, 3), 2)
    >>> Stream.take(data, n=2)
    [1, 2]
    >>> Stream(data).take() # First copy
    3
    >>> Stream(data).take(1) # Second copy
    [3]
    >>> Stream(data)
    Traceback (most recent call last):
        ...
    IndexError: StreamTeeHub has no more copies left to use.

    """
    raise AttributeError("Use peek or cast to Stream.")

  def copy(self):
    """
    Returns a new "T" (tee) copy of this StreamTeeHub without consuming
    any of the copies done with the constructor.
    """
    if self._iters:
      a, b = it.tee(self._iters[0])
      self._iters[0] = a
      return Stream(b)
    iter(self) # Try to consume (so it'll raise the same error as usual)

  limit = wraps(Stream.limit)(lambda self, n: Stream(self).limit(n))
  skip = wraps(Stream.skip)(lambda self, n: Stream(self).skip(n))
  append = wraps(Stream.append)( lambda self, *other:
                                   Stream(self).append(*other) )
  map = wraps(Stream.map)(lambda self, func: Stream(self).map(func))
  filter = wraps(Stream.filter)(lambda self, func: Stream(self).filter(func))


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
  ...   x = thub(x, 2) # Casts to StreamTeeHub, when needed
  ...   y = thub(y, 2)
  ...   return (x - y) / (x + y) # Return type might be number or Stream

  With numbers:

  >>> sub_sum(1, 1.)
  0.0

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


class Streamix(Stream):
  """
  Stream mixer of iterables.

  Examples
  --------

  With integer iterables:

  >>> s1 = [-1, 1, 3, 2]
  >>> s2 = Stream([4, 4, 4])
  >>> s3 = tuple([-3, -5, -7, -5, -7, -1])
  >>> smix = Streamix(zero=0) # Default zero is 0.0, changed to keep integers
  >>> smix.add(0, s1) # 1st number = delta time (in samples) from last added
  >>> smix.add(2, s2)
  >>> smix.add(0, s3)
  >>> smix
  <audiolazy.lazy_stream.Streamix object at ...>
  >>> list(smix)
  [-1, 1, 4, 1, -3, -5, -7, -1]

  With time constants:

  >>> from audiolazy import sHz, line
  >>> s, Hz = sHz(10) # You probably will use 44100 or something alike, not 10
  >>> sdata = list(line(2 * s, 1, -1, finish=True))
  >>> smix = Streamix()
  >>> smix.add(0.0 * s, sdata)
  >>> smix.add(0.5 * s, sdata)
  >>> smix.add(1.0 * s, sdata)
  >>> result = [round(sm, 2) for sm in smix]
  >>> len(result)
  35
  >>> 0.5 * s # Let's see how many samples this is
  5.0
  >>> result[:7]
  [1.0, 0.89, 0.79, 0.68, 0.58, 1.47, 1.26]
  >>> result[10:17]
  [0.42, 0.21, 0.0, -0.21, -0.42, 0.37, 0.05]
  >>> result[-1]
  -1.0

  See Also
  --------
  ControlStream :
    Stream (iterable with operators)
  sHz :
    Time in seconds (s) and frequency in hertz (Hz) constants from sample
    rate in samples/second.

  """
  def __init__(self, keep=False, zero=0.):
    self._not_playing = deque() # Tuples (integer delta, iterable)
    self._playing = []
    self.keep = keep

    def data_generator():
      count = 0.5
      to_remove = []

      while True:
        # Find if there's anything new to start "playing"
        while self._not_playing and (count >= self._not_playing[0][0]):
          delta, newdata = self._not_playing.popleft()
          self._playing.append(newdata)
          count -= delta # Delta might be float (less error propagation)

        # Sum the data to be played, seeing if something finished
        data = zero
        for snd in self._playing:
          try:
            data += next(snd)
          except StopIteration:
            to_remove.append(snd)

        # Remove finished
        if to_remove:
          for snd in to_remove:
            self._playing.remove(snd)
          to_remove = []

        # Tests whether there were any data (finite Streamix had finished?)
        if not (self.keep or self._playing or self._not_playing):
          break # Stops the iterator

        # Finish iteration
        yield data
        count += 1.

    super(Streamix, self).__init__(data_generator())

  def add(self, delta, data):
    """
    Adds (enqueues) an iterable event to the mixer.

    Parameters
    ----------
    delta :
      Time in samples since last added event. This can be zero and can be
      float. Use "s" object from sHz for time conversion.
    data :
      Iterable (e.g. a list, a tuple, a Stream) to be "played" by the mixer at
      the given time delta.

    See Also
    --------
    sHz :
      Time in seconds (s) and frequency in hertz (Hz) constants from sample
      rate in samples/second.

    """
    if delta < 0:
      raise ValueError("Delta time should be always positive")
    self._not_playing.append((delta, iter(data)))
