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
Example based on the Shepard tone
"""

from __future__ import division
from audiolazy import (sHz, Streamix, log2, line, window, sinusoid, AudioIO,
                       chunks)
import sys

# Basic initialization
rate = 44100
s, Hz = sHz(rate)
kHz = 1e3 * Hz

# Some parameters
table_len = 8192
min_freq = 20 * Hz
max_freq = 10 * kHz
duration = 60 * s

# "Track-by-track" partials configuration
noctaves = abs(log2(max_freq/min_freq))
octave_duration = duration / noctaves
smix = Streamix()
data = [] # Global: keeps one parcial "track" for all uses (but the first)

# Inits "data"
def partial():
  smix.add(octave_duration, partial_cached()) # Next track/partial event
  # Octave-based frequency values sequence
  scale = 2 ** line(duration, finish=True)
  partial_freq = (scale - 1) * (max_freq - min_freq) + min_freq
  # Envelope to "hide" the partial beginning/ending
  env = [k ** 2 for k in window.hamming(int(round(duration)))]
  # The generator, properly:
  for el in env * sinusoid(partial_freq) / noctaves:
    data.append(el)
    yield el

# Replicator ("track" data generator)
def partial_cached():
  smix.add(octave_duration, partial_cached()) # Next track/partial event
  for el in data:
    yield el

# Play!
smix.add(0, partial()) # Starts the mixing with the first track/partial
api = sys.argv[1] if sys.argv[1:] else None # Choose API via command-line
chunks.size = 1 if api == "jack" else 16
with AudioIO(True, api=api) as player:
  player.play(smix, rate=rate)
