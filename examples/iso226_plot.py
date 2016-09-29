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
Plots ISO/FDIS 226:2003 equal loudness contour curves

This is based on figure A.1 of ISO226, and needs Scipy and Matplotlib
"""

from __future__ import division

from audiolazy import exp, line, ln, phon2dB, xrange
import pylab

title = "ISO226 equal loudness curves"
freqs = list(exp(line(2048, ln(20), ln(12500), finish=True)))
pylab.figure(title, figsize=[8, 4.5], dpi=120)

# Plots threshold
freq2dB_threshold = phon2dB.iso226(None) # Threshold
pylab.plot(freqs, freq2dB_threshold(freqs), color="blue", linestyle="--")
pylab.text(300, 5, "Hearing threshold", fontsize=8,
           horizontalalignment="right")

# Plots 20 to 80 phons
for loudness in xrange(20, 81, 10): # in phons
  freq2dB = phon2dB.iso226(loudness)
  pylab.plot(freqs, freq2dB(freqs), color="black")
  pylab.text(850, loudness + 2, "%d phon" % loudness, fontsize=8,
             horizontalalignment="center")

# Plots 90 phons
freq2dB_90phon = phon2dB.iso226(90)
freqs4k1 = list(exp(line(2048, ln(20), ln(4100), finish=True)))
pylab.plot(freqs4k1, freq2dB_90phon(freqs4k1), color="black")
pylab.text(850, 92, "90 phon", fontsize=8, horizontalalignment="center")

# Plots 10 and 100 phons
freq2dB_10phon = phon2dB.iso226(10)
freq2dB_100phon = phon2dB.iso226(100)
freqs1k = list(exp(line(1024, ln(20), ln(1000), finish=True)))
pylab.plot(freqs, freq2dB_10phon(freqs), color="green", linestyle=":")
pylab.plot(freqs1k, freq2dB_100phon(freqs1k), color="green", linestyle=":")
pylab.text(850, 12, "10 phon", fontsize=8, horizontalalignment="center")
pylab.text(850, 102, "100 phon", fontsize=8, horizontalalignment="center")

# Plot axis config
pylab.axis(xmin=16, xmax=16000, ymin=-10, ymax=130)
pylab.xscale("log")
pylab.yticks(list(xrange(-10, 131, 10)))
xticks_values = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
pylab.xticks(xticks_values, xticks_values)
pylab.grid() # The grid follows the ticks

# Plot labels
pylab.title(title)
pylab.xlabel("Frequency (Hz)")
pylab.ylabel("Sound Pressure (dB)")

# Finish
pylab.tight_layout()
pylab.ioff()
pylab.show()
