# -*- coding: utf-8 -*-
"""
Simple audio/stream synthesis module

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

Created on Wed Jul 18 2012
danilo [dot] bellini [at] gmail [dot] com
"""

from math import sin, pi, ceil
import collections
import random
import itertools as it

# Audiolazy internal imports
from .lazy_stream import Stream, tostream, AbstractOperatorOverloaderMeta
from .lazy_itertools import cycle
from .lazy_filters import comb

__all__ = ["modulo_counter", "line", "fadein", "fadeout", "attack", "ones",
           "zeros", "zeroes", "adsr", "white_noise", "TableLookupMeta",
           "TableLookup", "DEFAULT_TABLE_SIZE", "sin_table", "saw_table",
           "sinusoid", "impulse", "karplus_strong"]


@tostream
def modulo_counter(start=0., modulo=256., step=1.):
  """
  Creates a lazy endless counter stream with the given modulo, i.e., its
  values ranges from 0. to the given "modulo", somewhat equivalent to:\n
    Stream(itertools.count(start, step)) % modulo\n
  Yet the given step can be an iterable, and doen't create unneeded big
  ints. All inputs can be float. Input order remembers slice/range inputs.
  All inputs can also be iterables. If any of them is an iterable, the end
  of this counter happen when there's no more data in one of those inputs.
  to continue iteration.
  """
  if isinstance(start, collections.Iterable):
    lastp = 0.
    c = 0.
    if isinstance(step, collections.Iterable):
      if isinstance(modulo, collections.Iterable):
        for p, m, s in it.izip(start, modulo, step):
          c += p - lastp
          c %= m
          yield c
          c += s
          lastp = p
      else:
        for p, s in it.izip(start, step):
          c += p - lastp
          c %= modulo
          yield c
          c += s
          lastp = p
    else:
      if isinstance(modulo, collections.Iterable):
        for p, m in it.izip(start, modulo):
          c += p - lastp
          c %= m
          yield c
          c += step
          lastp = p
      else: # Only start is iterable. This should be optimized!
        if step == 0:
          for p in start:
            yield p % modulo
        else:
          steps = int(modulo / step)
          if steps > 1:
            n = 0
            for p in start:
              c += p - lastp
              yield (c + n * step) % modulo
              lastp = p
              n += 1
              if n == steps:
                n = 0
                c = (c + steps * step) % modulo
          else:
            for p in start:
              c += p - lastp
              c %= modulo
              yield c
              c += step
              lastp = p
  else:
    c = start
    if isinstance(step, collections.Iterable):
      if isinstance(modulo, collections.Iterable):
        for m, s in it.izip(modulo, step):
          c %= m
          yield c
          c += s
      else: # Only step is iterable. This should be optimized!
        for s in step:
          c %= modulo
          yield c
          c += s
    else:
      if isinstance(modulo, collections.Iterable):
        for m in modulo:
          c %= m
          yield c
          c += step
      else: # None is iterable
        if step == 0:
          c = start % modulo
          while True:
            yield c
        else:
          steps = int(modulo / step)
          if steps > 1:
            n = 0
            while True:
              yield (c + n * step) % modulo
              n += 1
              if n == steps:
                n = 0
                c = (c + steps * step) % modulo
          else:
            while True:
              c %= modulo
              yield c
              c += step


@tostream
def line(dur, begin=0., end=1., finish=False):
  """
  Finite Stream with a straight line, could be used as fade in/out effects.

  Duration is given in number of samples. Use the sHz function to help with
  durations in seconds. Amplitudes should be float numbers between -1 and 1.\n
  The last sample won't have "end" amplitude unless finish is True, i.e.,
  without explicitly saying "finish=True", the "end" input works like a
  "stop" range parameter, although it can [should] be a float. This is so
  to help concatenating several lines.\n
  Using line(<inputs>).append([end]) you can finish the line with one extra
  sample without worrying with the "finish" input.
  """
  m = (end - begin) / (dur - (1. if finish else 0.))
  for sample in xrange(int(dur + .5)):
    yield begin + sample * m


def fadein(dur):
  """
  Linear fading in.

  Parameters
  ----------
  dur :
    Duration, in number of samples.

  Returns
  -------
  Stream instance yielding a line from zero to one.
  """
  return line(dur)


def fadeout(dur):
  """
  Linear fading out. Multiply by this one at end to finish and avoid clicks.

  Parameters
  ----------
  dur :
    Duration, in number of samples.

  Returns
  -------
  Stream instance yielding the line. The starting amplitude is is 1.0.
  """
  return line(dur, 1., 0.)


def attack(a, d, s):
  """
  Linear ADS fading attack stream generator, useful to be multiplied with a
  given stream.

  Parameters
  ----------
  a :
    "Attack" time, in number of samples.
  d :
    "Decay" time, in number of samples.
  s :
    "Sustain" amplitude level (should be based on attack amplitude).
    The sustain can be a Stream, if desired.

  Returns
  -------
  Stream instance yielding an endless envelope, or a finite envelope if the
  sustain input is a finite Stream. The attack amplitude is is 1.0.

  """
  # Configure sustain possibilities
  if isinstance(s, collections.Iterable):
    it_s = iter(s)
    s = it_s.next()
  else:
    it_s = None

  # Attack and decay lines
  m_a = 1. / a
  m_d = (s - 1.) / d
  len_a = int(a + .5)
  len_d = int(d + .5)
  for sample in xrange(len_a):
    yield sample * m_a
  for sample in xrange(len_d):
    yield 1. + sample * m_d

  # Sustain!
  if it_s is None:
    while True:
      yield s
  else:
    for s in it_s:
      yield s


@tostream
def ones(dur=None):
  """
  Ones stream generator.
  You may multiply your endless stream by this to enforce an end to it.

  Parameters
  ----------
  dur :
    Duration, in number of samples; endless if not given.

  Returns
  -------
  Stream that repeats "1.0" during a given time duration (if any) or
  endlessly.

  """
  if dur is None:
    while True:
      yield 1.0
  for x in xrange(int(.5 + dur)):
    yield 1.0


@tostream
def zeros(dur=None):
  """
  Zeros/zeroes stream generator.
  You may sum your endless stream by this to enforce an end to it.

  Parameters
  ----------
  dur :
    Duration, in number of samples; endless if not given.

  Returns
  -------
  Stream that repeats "0.0" during a given time duration (if any) or
  endlessly.

  """
  if dur is None:
    while True:
      yield 0.0
  for x in xrange(int(.5 + dur)):
    yield 0.0

zeroes = zeros


@tostream
def adsr(dur, a, d, s, r):
  """
  Linear ADSR envelope.

  Parameters
  ----------
  dur :
    Duration, in number of samples, including the release time.
  a :
    "Attack" time, in number of samples.
  d :
    "Decay" time, in number of samples.
  s :
    "Sustain" amplitude level (should be based on attack amplitude).
  r :
    "Release" time, in number of samples.

  Returns
  -------
  Stream instance yielding a finite ADSR envelope, starting and finishing with
  0.0, having peak value of 1.0.

  """
  m_a = 1. / a
  m_d = (s - 1.) / d
  m_r = - s * 1. / r
  len_a = int(a + .5)
  len_d = int(d + .5)
  len_r = int(r + .5)
  len_s = int(dur + .5) - len_a - len_d - len_r
  for sample in xrange(len_a):
    yield sample * m_a
  for sample in xrange(len_d):
    yield 1. + sample * m_d
  for sample in xrange(len_s):
    yield s
  for sample in xrange(len_r):
    yield s + sample * m_r


@tostream
def white_noise(dur=None):
  """
  White noise stream generator.

  Parameters
  ----------
  dur :
    Duration, in number of samples; endless if not given.

  Returns
  -------
  Stream yielding random numbers between -1 and 1.

  """
  if dur is None:
    while True:
      yield random.uniform(-1.,1.)
  for x in xrange(int(.5 + dur)):
    yield random.uniform(-1.,1.)


class TableLookupMeta(AbstractOperatorOverloaderMeta):
  """
  Table lookup metaclass. This class overloads all operators to the
  TableLookup class, applying them to the table contents, elementwise.
  Table length and number of cycles should be equal for this to work.
  """
  __operators__ = ("add radd sub rsub mul rmul pow rpow div rdiv mod rmod "
                   "truediv rtruediv floordiv rfloordiv "
                   "pos neg lshift rshift rlshift rrshift "
                   "and rand or ror xor rxor invert ")

  def __binary__(cls, op_func):
    def dunder(self, other):
      if isinstance(other, TableLookup):
        if self.cycles != other.cycles:
          raise ValueError("Incompatible number of cycles")
        if len(self) != len(other):
          raise ValueError("Incompatible sizes")
        zip_tables = zip(self.table, other.table)
        new_table = [op_func(data1, data2) for data1, data2 in zip_tables]
        return TableLookup(new_table, self.cycles)
      if isinstance(other, (int, float, complex)):
        new_table = [op_func(data, other) for data in self.table]
        return TableLookup(new_table, self.cycles)
      raise NotImplementedError("Unknown action do be done")
    return dunder

  def __rbinary__(cls, op_func):
    def dunder(self, other):
      if isinstance(other, (int, float, complex)):
        new_table = [op_func(other, data) for data in self.table]
        return TableLookup(new_table, self.cycles)
      raise NotImplementedError("Unknown action do be done")
    return dunder

  def __unary__(cls, op_func):
    def dunder(self):
      new_table = [op_func(data) for data in self.table]
      return TableLookup(new_table, self.cycles)
    return dunder


class TableLookup(object):
  """
  Table lookup synthesis class, also allowing multi-cycle tables as input.
  """
  __metaclass__ = TableLookupMeta

  def __init__(self, table, cycles=1):
    """
    Inits a table lookup. The given table should be a sequence, like a list.
    The cycles input should have the number of cycles in table for frequency
    calculation afterwards.
    """
    self.table = table
    self.cycles = cycles

  @property
  def table(self):
    return self._table

  @table.setter
  def table(self, value):
    self._table = value
    self._len = len(value)

  def __len__(self):
    return self._len

  def __call__(self, freq, phase=0.):
    """
    Returns a wavetable lookup synthesis endless stream. Play it with the
    given frequency and starting phase. Phase is given in rads, and frequency
    in rad/sample. Accepts streams of numbers, as well as numbers, for both
    frequency and phase inputs.
    """
    total_length = len(self)
    total_len_float = float(total_length)
    cycle_length = total_len_float / (self.cycles * 2 * pi)
    step = cycle_length * freq
    part = cycle_length * phase
    tbl_iter = modulo_counter(part, total_len_float, step)
    tbl = self.table
    #return Stream(tbl[int(idx)] for idx in tbl_iter)
    return Stream(tbl[int(idx)] * (1. - (idx - int(idx))) +
                  tbl[int(ceil(idx)) - total_length] * (idx - int(idx))
                  for idx in tbl_iter)

  def __getitem__(self, idx):
    """
    Gets an item from the table from its index, which can possibly be a float.
    The data is linearly interpolated.
    """
    total_length = len(self)
    tbl = self.table
    return tbl[int(idx) % total_length] * (1. - (idx - int(idx))) + \
           tbl[int(ceil(idx)) % total_length] * (idx - int(idx))

  def __eq__(self, other):
    if isinstance(other, TableLookup):
      return (self.cycles == other.cycles) and (self.table == other.table)
    return False

  def __ne__(self, other):
    return not self == other

  def harmonize(self, harmonics_dict):
    """
    Returns a "harmonized" table lookup instance by using a "harmonics"
    dictionary with {partial: amplitude} terms, where all "partial" keys have
    to be integers.
    """
    data = sum(cycle(self.table[::partial+1]) * amplitude
               for partial, amplitude in harmonics_dict.iteritems())
    return TableLookup(data.take(len(self)), cycles=self.cycles)

  def normalize(self):
    """
    Returns a new table with values ranging from -1 to 1, reaching at least
    one of these, unless there's no data.
    """
    max_abs = max(self.table, key=abs)
    if max_abs == 0:
      raise ValueError("Can't normalize zeros")
    return self / max_abs


# Create the instance for each default table
DEFAULT_TABLE_SIZE = 2**16
sin_table = TableLookup([sin(x * 2 * pi / DEFAULT_TABLE_SIZE)
                         for x in xrange(DEFAULT_TABLE_SIZE)])
saw_table = TableLookup([x * (1./(DEFAULT_TABLE_SIZE-1))
                         for x in xrange(DEFAULT_TABLE_SIZE)])


@tostream
def sinusoid(freq, phase=0.):
  """
  Sinusoid based on the optimized math.sin
  """
  # When at 44100 samples / sec, 5 seconds of this leads to an error of 8e-14
  # peak to peak. That's fairly enough.
  for n in modulo_counter(start=phase, modulo=2 * pi, step=freq):
    yield sin(n)


@tostream
def impulse(dur=None):
  """
  Impulse stream generator.

  Parameters
  ----------
  dur :
    Duration, in number of samples; endless if not given.

  Returns
  -------
  Stream that repeats "0.0" during a given time duration (if any) or
  endlessly, but starts with one (and only one) "1.0".

  """
  if dur is None:
    yield 1.
    while True:
      yield 0.
  elif dur >= .5:
    num_samples = int(dur - .5)
    yield 1.
    for x in xrange(num_samples):
      yield 0.


def karplus_strong(freq, alpha=.99, memory=white_noise):
  return comb(2 * pi / freq, alpha).linearize()(zeros(), memory=memory)