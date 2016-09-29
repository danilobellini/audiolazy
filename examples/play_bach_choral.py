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
Random Bach Choral playing example (needs Music21 corpus)

This example uses a personalized synth based on the Karplus-Strong model.
You can also get the synth models and effects from the ode_to_joy.py example
and adapt them to get used here.
"""

from __future__ import unicode_literals, print_function
from music21 import corpus
from music21.expressions import Fermata
import audiolazy as lz
import random, operator, sys, time
from functools import reduce


def ks_synth(freq):
  """
  Synthesize the given frequency into a Stream by using a model based on
  Karplus-Strong.
  """
  ks_mem = (sum(lz.sinusoid(x * freq) for x in [1, 3, 9]) +
            lz.white_noise() + lz.Stream(-1, 1)) / 5
  return lz.karplus_strong(freq, memory=ks_mem)


def get_random_choral(log=True):
  """ Gets a choral from the J. S. Bach chorals corpus (in Music21). """
  choral_file = corpus.getBachChorales()[random.randint(0, 399)]
  choral = corpus.parse(choral_file)
  if log:
    print("Chosen choral:", choral.metadata.title)
  return choral


def m21_to_stream(score, synth=ks_synth, beat=90, fdur=2., pad_dur=.5,
                  rate=lz.DEFAULT_SAMPLE_RATE):
  """
  Converts Music21 data to a Stream object.

  Parameters
  ----------
  score :
    A Music21 data, usually a music21.stream.Score instance.
  synth :
    A function that receives a frequency as input and should yield a Stream
    instance with the note being played.
  beat :
    The BPM (beats per minute) value to be used in playing.
  fdur :
    Relative duration of a fermata. For example, 1.0 ignores the fermata, and
    2.0 (default) doubles its duration.
  pad_dur :
    Duration in seconds, but not multiplied by ``s``, to be used as a
    zero-padding ending event (avoids clicks at the end when playing).
  rate :
    The sample rate, given in samples per second.

  """
  # Configuration
  s, Hz = lz.sHz(rate)
  step = 60. / beat * s

  # Creates a score from the music21 data
  score = reduce(operator.concat,
                 [[(pitch.frequency * Hz, # Note
                    note.offset * step, # Starting time
                    note.quarterLength * step, # Duration
                    Fermata in note.expressions) for pitch in note.pitches]
                                                 for note in score.flat.notes]
                )

  # Mix all notes into song
  song = lz.Streamix()
  last_start = 0
  for freq, start, dur, has_fermata in score:
    delta = start - last_start
    if has_fermata:
      delta *= 2
    song.add(delta, synth(freq).limit(dur))
    last_start = start

  # Zero-padding and finishing
  song.add(dur + pad_dur * s, lz.Stream([]))
  return song


# Play the song!
if __name__ == "__main__":
  api = next(arg for arg in sys.argv[1:] + [None] if arg != "loop")
  lz.chunks.size = 1 if api == "jack" else 16
  rate = 44100
  while True:
    with lz.AudioIO(True, api=api) as player:
      player.play(m21_to_stream(get_random_choral(), rate=rate), rate=rate)
    if not "loop" in sys.argv[1:]:
      break
    time.sleep(3)
