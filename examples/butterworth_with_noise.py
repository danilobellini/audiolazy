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
Two butterworth filters with Scipy applied to white noise

This example is based on the experiment number 34 from

  Demonstrations to accompany Bregman’s Auditory Scene Analysis

by Albert S. Bregman and Pierre A. Ahad.

This experiment shows the audio is perceived as pitched differently on the
100ms glimpses when the context changes, although they are physically
identical, i.e., the glimpses are indeed perceptually segregated from the
background noise.

Noise ranges are from [0; 2kHz] and [2kHz; 4kHz] and durations are 100ms and
400ms instead of the values declared in the original text. IIR filters are
being used instead of FIR ones to get the noise, and the fact that originally
there's no silence between the higher and lower pitch contexts, but the core
idea of the experiment remains the same.
"""

from audiolazy import (sHz, dB10, ZFilter, pi, ControlStream, white_noise,
                       chunks, AudioIO, xrange, z)
from scipy.signal import butter, buttord
import numpy as np
from time import sleep
import sys

rate = 44100
s, Hz = sHz(rate)
kHz = 1e3 * Hz
tol = 100 * Hz
freq = 2 * kHz

wp = freq - tol # Bandpass frequency in rad/sample (from zero)
ws = freq + tol # Bandstop frequency in rad/sample (up to Nyquist frequency)
order, new_wp_divpi = buttord(wp/pi, ws/pi, gpass=dB10(.6), gstop=dB10(.4))
ssfilt = butter(order, new_wp_divpi, btype="lowpass")
filt_low = ZFilter(ssfilt[0].tolist(), ssfilt[1].tolist())

## That can be done without scipy using the equation directly:
#filt_low = ((2.90e-4 + 1.16e-3 * z ** -1 + 1.74e-3 * z ** -2
#                     + 1.16e-3 * z ** -3 + 2.90e-4 * z ** -4) /
#            (1       - 3.26    * z ** -1 + 4.04    * z ** -2
#                     - 2.25    * z ** -3 +  .474   * z ** -4))

wp = np.array([freq + tol, 2 * freq - tol]) # Bandpass range in rad/sample
ws = np.array([freq - tol, 2 * freq + tol]) # Bandstop range in rad/sample
order, new_wp_divpi = buttord(wp/pi, ws/pi, gpass=dB10(.6), gstop=dB10(.4))
ssfilt = butter(order, new_wp_divpi, btype="bandpass")
filt_high = ZFilter(ssfilt[0].tolist(), ssfilt[1].tolist())

## Likewise, using the equation directly this one would be:
#filt_high = ((2.13e-3 * (1 - z ** -6) - 6.39e-3 * (z ** -2 - z ** -4)) /
#             (1 - 4.99173 * z ** -1 + 10.7810 * z ** -2 - 12.8597 * z ** -3
#                + 8.93092 * z ** -4 - 3.42634 * z ** -5 + .569237 * z ** -6))

gain_low = ControlStream(0)
gain_high = ControlStream(0)

low = filt_low(white_noise())
high = filt_high(white_noise())
low /= 2 * max(low.take(2000))
high /= 2 * max(high.take(2000))

api = sys.argv[1] if sys.argv[1:] else None # Choose API via command-line
chunks.size = 1 if api == "jack" else 16
with AudioIO(api=api) as player:
  player.play(low * gain_low + high * gain_high)
  gain_low.value = 1
  while True:
    gain_high.value = 0
    sleep(1)
    for unused in xrange(5): # Keeps low playing
      sleep(.1)
      gain_high.value = 0
      sleep(.4)
      gain_high.value = 1

    gain_low.value = 0
    sleep(1)
    for unused in xrange(5): # Keeps high playing
      sleep(.1)
      gain_low.value = 0
      sleep(.4)
      gain_low.value = 1
