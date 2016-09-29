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
Butterworth filter from SciPy as a ZFilter instance, with plots

One resonator (first order filter) is used for comparison with the
butterworth from the example (third order filter). Both has zeros at
1 (DC level) and -1 (Nyquist).
"""

from __future__ import print_function
from audiolazy import sHz, ZFilter, dB10, resonator, pi
from scipy.signal import butter, buttord
import pylab

# Example
rate = 44100
s, Hz = sHz(rate)
wp = pylab.array([100 * Hz, 240 * Hz]) # Bandpass range in rad/sample
ws = pylab.array([80 * Hz, 260 * Hz]) # Bandstop range in rad/sample

# Let's use wp/pi since SciPy defaults freq from 0 to 1 (Nyquist frequency)
order, new_wp_divpi = buttord(wp/pi, ws/pi, gpass=dB10(.6), gstop=dB10(.4))
ssfilt = butter(order, new_wp_divpi, btype="bandpass")
filt_butter = ZFilter(ssfilt[0].tolist(), ssfilt[1].tolist())

# Some debug information
new_wp = new_wp_divpi * pi
print("Butterworth filter order:", order) # Should be 3
print("Bandpass ~3dB range (in Hz):", new_wp / Hz)

# Resonator using only the frequency and bandwidth from the Butterworth filter
freq = new_wp.mean()
bw = new_wp[1] - new_wp[0]
filt_reson = resonator.z_exp(freq, bw)

# Plots with MatPlotLib
kwargs = {
  "min_freq": 10 * Hz,
  "max_freq": 800 * Hz,
  "rate": rate, # Ensure frequency unit in plot is Hz
}
filt_butter.plot(pylab.figure("From scipy.signal.butter"), **kwargs)
filt_reson.plot(pylab.figure("From audiolazy.resonator.z_exp"), **kwargs)
filt_butter.zplot(pylab.figure("Zeros/Poles from scipy.signal.butter"))
filt_reson.zplot(pylab.figure("Zeros/Poles from audiolazy.resonator.z_exp"))
pylab.ioff()
pylab.show()
