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
# Created on Tue Sep 10 18:02:32 2013
# danilo [dot] bellini [at] gmail [dot] com
"""
Voiced "ah-eh-ee-oh-oo" based on resonators at formant frequencies
"""

from __future__ import unicode_literals

from audiolazy import (sHz, maverage, rint, AudioIO, ControlStream, pi,
                       CascadeFilter, resonator, saw_table)
from time import sleep

# Initialization
rate = 44100
s, Hz = sHz(rate)

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

inertia_filter = maverage(rint(.5 * s))

with AudioIO() as player:
  f1, f2 = ControlStream(0), ControlStream(pi)
  gain = ControlStream(0)
  filt = CascadeFilter([
    resonator.z_exp(inertia_filter(f1), 400 * Hz),
    resonator.z_exp(inertia_filter(f2), 2000 * Hz),
  ])
  sig = filt((saw_table)(100 * Hz)) * inertia_filter(gain)

  player.play(sig)

  vowels = "aɛiɒu"
  for vowel in vowels:
    coeffs = formants[vowel]
    print "Now playing: ", vowel
    f1.value = coeffs[0] * Hz
    f2.value = coeffs[1] * Hz
    gain.value = 1
    sleep(2)
