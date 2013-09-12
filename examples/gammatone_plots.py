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
# Created on Sun Oct 07 2012
# danilo [dot] bellini [at] gmail [dot] com
"""
Gammatone frequency and impulse response plots example
"""

from __future__ import division
from audiolazy import (erb, gammatone, gammatone_erb_constants, sHz, impulse,
                       dB20)
from numpy import linspace, ceil
from matplotlib import pyplot as plt

# Initialization info
rate = 44100
s, Hz = sHz(rate)
ms = 1e-3 * s
plot_freq_time = {80.: 60 * ms,
                  100.: 50 * ms,
                  200.: 40 * ms,
                  500.: 25 * ms,
                  800.: 20 * ms,
                  1000.: 15 * ms}
freq = linspace(0.1, 2 * max(freq for freq in plot_freq_time), 100)

fig1 = plt.figure("Frequency response", figsize=(16, 9), dpi=60)
fig2 = plt.figure("Impulse response", figsize=(16, 9), dpi=60)

# Plotting loop
for idx, (fc, endtime) in enumerate(sorted(plot_freq_time.items()), 1):
  # Configuration for the given frequency
  num_samples = int(round(endtime))
  time_scale = linspace(0, num_samples / ms, num_samples)
  bw = gammatone_erb_constants(4)[0] * erb(fc * Hz, Hz)

  # Subplot configuration
  plt.figure(1)
  plt.subplot(2, ceil(len(plot_freq_time) / 2), idx)
  plt.title("Frequency response - {0} Hz".format(fc))
  plt.xlabel("Frequency (Hz)")
  plt.ylabel("Gain (dB)")

  plt.figure(2)
  plt.subplot(2, ceil(len(plot_freq_time) / 2), idx)
  plt.title("Impulse response - {0} Hz".format(fc))
  plt.xlabel("Time (ms)")
  plt.ylabel("Amplitude")

  # Plots each filter frequency and impulse response
  for gt, config in zip(gammatone, ["b-", "g--", "r-.", "k:"]):
    filt = gt(fc * Hz, bw)

    plt.figure(1)
    plt.plot(freq, dB20(filt.freq_response(freq * Hz)), config,
             label=gt.__name__)

    plt.figure(2)
    plt.plot(time_scale, filt(impulse()).take(num_samples), config,
             label=gt.__name__)

# Finish
for graph in fig1.axes + fig2.axes:
  graph.grid()
  graph.legend(loc="best")

fig1.tight_layout()
fig2.tight_layout()

plt.show()