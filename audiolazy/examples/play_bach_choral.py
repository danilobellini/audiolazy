#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of AudioLazy, the signal processing Python package.
# Copyright (C) 2012-2013 Danilo de Jesus da Silva Bellini
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
# Created on Mon Jan 28 2013
# danilo [dot] bellini [at] gmail [dot] com
"""
Random Bach Choral playing example (needs Music21 corpus)
"""

from __future__ import unicode_literals, print_function
from music21 import corpus
from music21.expressions import Fermata
import audiolazy as lz
import random
import operator
from functools import reduce

def ks_mem(freq):
  """ Alternative memory for Karplus-Strong """
  return (sum(lz.sinusoid(x * freq) for x in [1, 3, 9]) +
          lz.white_noise() + lz.Stream(-1, 1)) / 5

# Configuration
rate = 44100
s, Hz = lz.sHz(rate)
ms = 1e-3 * s

beat = 90 # bpm
step = 60. / beat * s

# Open the choral file
choral_file = corpus.getBachChorales()[random.randint(0, 399)]
choral = corpus.parse(choral_file)
print("Playing", choral.metadata.title)

# Creates the score from the music21 data
score = reduce(operator.concat,
               [[(pitch.frequency * Hz, # Note
                  note.offset * step, # Starting time
                  note.quarterLength * step, # Duration
                  Fermata in note.expressions) for pitch in note.pitches]
                                               for note in choral.flat.notes]
              )

# Mix all notes into song
song = lz.Streamix()
last_start = 0
for freq, start, dur, has_fermata in score:
  delta = start - last_start
  if has_fermata:
    delta *= 2
  song.add(delta, lz.karplus_strong(freq, memory=ks_mem(freq)) * lz.ones(dur))
  last_start = start

# Play the song!
with lz.AudioIO(True) as player:
  song = song.append(lz.zeros(.5 * s)) # To avoid a click at the end
  player.play(song, rate=rate)
