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
Voiced "ah-eh-ee-oh-oo" based on resonators at formant frequencies
"""

from __future__ import unicode_literals, print_function

from audiolazy import (sHz, maverage, rint, AudioIO, ControlStream,
                       CascadeFilter, resonator, saw_table, chunks)
from time import sleep
import sys

# Script input, change this with symbols from the table below
vowels = "aɛiɒu"

# Formant table from in http://en.wikipedia.org/wiki/Formant
formants = {
  "i": [240, 2400],
  "y": [235, 2100],
  "e": [390, 2300],
  "ø": [370, 1900],
  "ɛ": [610, 1900],
  "œ": [585, 1710],
  "a": [850, 1610],
  "æ": [820, 1530],
  "ɑ": [750, 940],
  "ɒ": [700, 760],
  "ʌ": [600, 1170],
  "ɔ": [500, 700],
  "ɤ": [460, 1310],
  "o": [360, 640],
  "ɯ": [300, 1390],
  "u": [250, 595],
}


# Initialization
rate = 44100
s, Hz = sHz(rate)
inertia_dur = .5 * s
inertia_filter = maverage(rint(inertia_dur))

api = sys.argv[1] if sys.argv[1:] else None # Choose API via command-line
chunks.size = 1 if api == "jack" else 16

with AudioIO(api=api) as player:
  first_coeffs = formants[vowels[0]]

  # These are signals to be changed during the synthesis
  f1 = ControlStream(first_coeffs[0] * Hz)
  f2 = ControlStream(first_coeffs[1] * Hz)
  gain = ControlStream(0) # For fading in

  # Creates the playing signal
  filt = CascadeFilter([
    resonator.z_exp(inertia_filter(f1).skip(inertia_dur), 400 * Hz),
    resonator.z_exp(inertia_filter(f2).skip(inertia_dur), 2000 * Hz),
  ])
  sig = filt((saw_table)(100 * Hz)) * inertia_filter(gain)

  th = player.play(sig)
  for vowel in vowels:
    coeffs = formants[vowel]
    print("Now playing: ", vowel)
    f1.value = coeffs[0] * Hz
    f2.value = coeffs[1] * Hz
    gain.value = 1 # Fade in the first vowel, changes nothing afterwards
    sleep(2)

  # Fade out
  gain.value = 0
  sleep(inertia_dur / s + .2) # Divide by s because here it's already
                              # expecting a value in seconds, and we don't
                              # want ot give a value in a time-squaed unit
                              # like s ** 2
