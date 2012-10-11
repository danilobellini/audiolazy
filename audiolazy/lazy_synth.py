#!/usr/bin/env python
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

from math import sin, pi
import collections
import random

# Audiolazy internal imports
from .lazy_stream import Stream, tostream, AbstractOperatorOverloaderMeta
from .lazy_itertools import cycle

# Tables for periodic table-lookup wave synthesis
DEFAULT_TABLE_SIZE = 2**16
DEFAULT_TABLES = {"sinusoid": [sin(x * 2 * pi / DEFAULT_TABLE_SIZE)
                               for x in xrange(DEFAULT_TABLE_SIZE)],
                  "sawtooth": [x * (1./(DEFAULT_TABLE_SIZE-1))
                               for x in xrange(DEFAULT_TABLE_SIZE)],

                 }


@tostream
def modulo_counter(start=0., modulo=15., step=1.):
  """
  Creates a lazy endless counter stream with the given modulo, i.e., its
  values ranges from 0. to the given "modulo", somewhat equivalent to:\n
    Stream(itertools.count(start, step)) % modulo\n
  Yet the given step can be an iterable, and doen't create unneeded big
  ints. All inputs can be float. Input order remembers slice/range inputs.
  The start can also be a stream (or iterable). If the start or step
  is an iterable, the end of this counter happen when there's no more
  data in start/step to continue iteration.
  """
  if isinstance(start, collections.Iterable):
    # Updates "step" to have the iterable contents from start.
    start_stream = Stream(start)
    start = start_stream.take() # Now start is a number
    given_start = Stream([start], start_stream.copy())
    step = step + (start_stream - given_start) # Start's derivative
  c = start % modulo # c is the internal counter value
  if isinstance(step, collections.Iterable):
    step = Stream(step)
    yield c # From now, it have to be faster.
    for s in step:
      c += s
      c %= modulo
      yield c
  else:
    while True:
      yield c # From now, it have to be faster.
      c += step
      c %= modulo


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
  Linear fading in from zero to one.
  """
  return line(dur)


def fadeout(dur):
  """
  Linear fading out. Multiply by this one to finish and avoid clicks.
  The starting amplitude is is 1.0.
  """
  return line(dur, 1., 0.)


def attack(a, d, s):
  """
  Linear ADS fading attack endless stream, useful to be multiplied
  with a given stream. Parameters:\n
    a = "Attack" time, in samples
    d = "Decay" time, in samples
    s = "Sustain" amplitude level (should be based on attack amplitude)\n
  The attack amplitude is is 1.0. The sustain can be a Stream, if desired.
  """
  return line(a).append(line(d, 1.0, s)).append(s)


@tostream
def ones(self, dur):
  """
  Finite stream that repeats "1.0" for the given time duration, in sample
  You may multiply your endless stream by this to enforce an end to it.
  """
  for x in xrange(int(.5 + dur)):
    yield 1.0


@tostream
def zeros(self, dur):
  """
  Stream that repeats "0.0" during a given time duration.
  You may multiply your endless stream by this to enforce an end to it.
  """
  for x in xrange(int(.5 + dur)):
    yield 0.0

zeroes = zeros


def adsr(self, dur, a, d, s, r):
  """
  Linear ADSR envelope for a fixed duration. The inputs are keywords
  "a", "d", "s", "r", see attack and fadeout for more information.
  The given total duration includes the release time.
  """
  return attack(a, d, s) * Stream(ones(dur - r), self.line(r, 1., 0.))


@tostream
def white_noise(self):
  """
  White noise endless stream, ranging from -1 to 1.
  """
  while True:
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

  def binary_dunder(cls, op_func):
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

  def reverse_binary_dunder(cls, op_func):
    def dunder(self, other):
      if isinstance(other, (int, float, complex)):
        new_table = [op_func(other, data) for data in self.table]
        return TableLookup(new_table, self.cycles)
      raise NotImplementedError("Unknown action do be done")
    return dunder

  def unary_dunder(cls, op_func):
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
    Returns a wavetable lookup synthesis endless stream. Play it with the given frequency and starting
    phase. Phase is given in rads, and frequency in rad/s. Accepts streams of
    numbers as well as numbers as frequency and phase inputs.
    """
    total_length = len(self)
    total_len_float = float(total_length)
    cycle_length = total_len_float / (self.cycles * 2 * pi)
    step = cycle_length * freq
    part = cycle_length * phase
    tbl_iter = modulo_counter(part + .5, total_len_float + .5, step)
    return Stream(self.table[int(idx) % total_length] for idx in tbl_iter)

  def __eq__(self, other):
    if isinstance(other, TableLookup):
      return (self.cycles == other.cycles) and (self.table == other.table)
    return False

  def __ne__(self, other):
    return not self == other

  def harmonize(self, harmonics_dict):
    """
    Return a "harmonized" table lookup instance by using a "harmonics"
    dictionary with {partial: amplitude} terms, where all "partial" keys have
    to be integers.
    """
    data = sum(cycle(self.table[::partial+1]) * amplitude
               for partial, amplitude in harmonics_dict.iteritems())
    return TableLookup(data.take(len(self)), cycles=self.cycles)

  def normalized(self):
    """
    Return a new table with values ranging from -1 to 1, reaching at least
    one of these, unless there's no data.
    """
    max_abs = max(self.table, key=abs)
    if max_abs == 0:
      raise ValueError("Can't normalize zeros")
    return self / max_abs

# Create the instance for each default table
for table_name in DEFAULT_TABLES:
  locals()[table_name] = TableLookup(DEFAULT_TABLES[table_name])
