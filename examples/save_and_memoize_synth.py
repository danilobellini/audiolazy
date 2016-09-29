#!/usr/bin/env python
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
Random synthesis with saving and memoization
"""

from __future__ import division
from audiolazy import (sHz, octaves, chain, adsr, gauss_noise, sin_table, pi,
                       sinusoid, lag2freq, Streamix, zeros, clip, lowpass,
                       TableLookup, line, inf, xrange, thub, chunks)
from random import choice, uniform, randint
from functools import wraps, reduce
from contextlib import closing
import operator, wave


#
# Helper functions
#
def memoize(func):
  """
  Decorator for unerasable memoization based on function arguments, for
  functions without keyword arguments.
  """
  class Memoizer(dict):
    def __missing__(self, args):
      val = func(*args)
      self[args] = val
      return val
  memory = Memoizer()
  @wraps(func)
  def wrapper(*args):
    return memory[args]
  return wrapper


def save_to_16bit_wave_file(fname, sig, rate):
  """
  Save a given signal ``sig`` to file ``fname`` as a 16-bit one-channel wave
  with the given ``rate`` sample rate.
  """
  with closing(wave.open(fname, "wb")) as wave_file:
    wave_file.setnchannels(1)
    wave_file.setsampwidth(2)
    wave_file.setframerate(rate)
    for chunk in chunks((clip(sig) * 2 ** 15).map(int), dfmt="h", padval=0):
      wave_file.writeframes(chunk)


#
# AudioLazy Initialization
#
rate = 44100
s, Hz = sHz(rate)
ms = 1e-3 * s

# Frequencies (always in Hz here)
freq_base = 440
freq_min = 100
freq_max = 8000
ratios = [1/1, 8/7, 7/6, 3/2, 49/32, 7/4] # 2/1 is the next octave
concat = lambda iterables: reduce(operator.concat, iterables, [])
oct_partial = lambda freq: octaves(freq, fmin = freq_min, fmax = freq_max)
freqs = concat(oct_partial(freq_base * ratio) for ratio in ratios)


#
# Audio synthesis models
#
def freq_gen():
  """
  Endless frequency generator (in rad/sample).
  """
  while True:
    yield choice(freqs) * Hz


def new_note_track(env, synth):
  """
  Audio track with the frequencies.

  Parameters
  ----------
  env:
    Envelope Stream (which imposes the duration).
  synth:
    One-argument function that receives a frequency (in rad/sample) and
    returns a Stream instance (a synthesized note).

  Returns
  -------
  Endless Stream instance that joins synthesized notes.

  """
  list_env = list(env)
  return chain.from_iterable(synth(freq) * list_env for freq in freq_gen())


@memoize
def unpitched_high(dur, idx):
  """
  Non-harmonic treble/higher frequency sound as a list (due to memoization).

  Parameters
  ----------
  dur:
    Duration, in samples.
  idx:
    Zero or one (integer), for a small difference to the sound played.

  Returns
  -------
  A list with the synthesized note.

  """
  first_dur, a, d, r, gain = [
    (30 * ms, 10 * ms, 8 * ms, 10 * ms, .4),
    (60 * ms, 20 * ms, 8 * ms, 20 * ms, .5)
  ][idx]
  env = chain(adsr(first_dur, a=a, d=d, s=.2, r=r),
              adsr(dur - first_dur,
                   a=10 * ms, d=30 * ms, s=.2, r=dur - 50 * ms))
  result = gauss_noise(dur) * env * gain
  return list(result)


# Values used by the unpitched low synth
harmonics = dict(enumerate([3] * 4 + [2] * 4 + [1] * 10))
low_table = sin_table.harmonize(harmonics).normalize()


@memoize
def unpitched_low(dur, idx):
  """
  Non-harmonic bass/lower frequency sound as a list (due to memoization).

  Parameters
  ----------
  dur:
    Duration, in samples.
  idx:
    Zero or one (integer), for a small difference to the sound played.

  Returns
  -------
  A list with the synthesized note.

  """
  env = sinusoid(lag2freq(dur * 2)).limit(dur) ** 2
  freq = 40 + 20 * sinusoid(1000 * Hz, phase=uniform(-pi, pi)) # Hz
  result = (low_table(freq * Hz) + low_table(freq * 1.1 * Hz)) * env * .5
  return list(result)


def geometric_delay(sig, dur, copies, pamp=.5):
  """
  Delay effect by copying data (with Streamix).

  Parameters
  ----------
  sig:
    Input signal (an iterable).
  dur:
    Duration, in samples.
  copies:
    Number of times the signal will be replayed in the given duration. The
    signal is played copies + 1 times.
  pamp:
    The relative remaining amplitude fraction for the next played Stream,
    based on the idea that total amplitude should sum to 1. Defaults to 0.5.

  """
  out = Streamix()
  sig = thub(sig, copies + 1)
  out.add(0, sig * pamp) # Original
  remain = 1 - pamp
  for unused in xrange(copies):
    gain = remain * pamp
    out.add(dur / copies, sig * gain)
    remain -= gain
  return out


#
# Audio mixture
#
tracks = 3 # besides unpitched track
dur_note = 120 * ms
dur_perc = 100 * ms
smix = Streamix()

# Pitched tracks based on a 1:2 triangular wave
table = TableLookup(line(100, -1, 1).append(line(200, 1, -1)).take(inf))
for track in xrange(tracks):
  env = adsr(dur_note, a=20 * ms, d=10 * ms, s=.8, r=30 * ms) / 1.7 / tracks
  smix.add(0, geometric_delay(new_note_track(env, table), 80 * ms, 2))

# Unpitched tracks
pfuncs = [unpitched_low] * 4 + [unpitched_high]
snd = chain.from_iterable(choice(pfuncs)(dur_perc, randint(0, 1))
                          for unused in zeros())
smix.add(0, geometric_delay(snd * (1 - 1/1.7), 20 * ms, 1))


#
# Finishes (save in a wave file)
#
data = lowpass(5000 * Hz)(smix).limit(180 * s)
fname = "audiolazy_save_and_memoize_synth.wav"
save_to_16bit_wave_file(fname, data, rate)
