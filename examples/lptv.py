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
LPTV (Linear Periodically Time Variant) filter example (a.k.a. PLTV)
"""

from audiolazy import sHz, sinusoid, Stream, AudioIO, z, pi, chunks
import time, sys

# Basic initialization
rate = 44100
s, Hz = sHz(rate)

# Some time-variant coefficients
cycle_a1 = [.1, .2, .1, 0, -.1, -.2, -.1, 0]
cycle_a2 = [.1, 0, -.1, 0, 0]
a1 = Stream(*cycle_a1)
a2 = Stream(*cycle_a2) * 2
b1 = sinusoid(18 * Hz) # Sine phase
b2 = sinusoid(freq=7 * Hz, phase=pi/2) # Cosine phase

# The filter
filt = (1 + b1 * z ** -1 + b2 * z ** -2 + .7 * z ** -5)
filt /= (1 - a1 * z ** -1 - a2 * z ** -2 - .1 * z ** -3)

# A really simple input
input_data = sinusoid(220 * Hz)

# Let's play it!
api = sys.argv[1] if sys.argv[1:] else None # Choose API via command-line
chunks.size = 1 if api == "jack" else 16
with AudioIO(api=api) as player:
  th = player.play(input_data, rate=rate)
  time.sleep(1) # Wait a sec
  th.stop()
  time.sleep(1) # One sec "paused"
  player.play(filt(input_data), rate=rate) # It's nice with rate/2 here =)
  time.sleep(3) # Play the "filtered" input (3 secs)

# Quiz!
#
# Question 1: What's the filter "cycle" duration?
# Hint: Who cares?
#
# Question 2: Does the filter need to be periodic?
# Hint: Import white_noise and try to put this before defining the filt:
#   a1 *= white_noise()
#   a2 *= white_noise()
#
# Question 3: Does the input need to be periodic?
# Hint: Import comb and white_noise. Now try to use this as the input:
#   .9 * sinusoid(220 * Hz) + .01 * comb(200, .9)(white_noise())
