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
LPC plot with DFT, showing two formants (magnitude peaks)
"""

from audiolazy import sHz, sin_table, str2freq, lpc
import pylab

rate = 22050
s, Hz = sHz(rate)
size = 512
table = sin_table.harmonize({1: 1, 2: 5, 3: 3, 4: 2, 6: 9, 8: 1}).normalize()

data = table(str2freq("Bb3") * Hz).take(size)
filt = lpc(data, order=14) # Analysis filter
gain = 1e-2 # Gain just for alignment with DFT

# Plots the synthesis filter
# - If blk is given, plots the block DFT together with the filter
# - If rate is given, shows the frequency range in Hz
(gain / filt).plot(blk=data, rate=rate, samples=1024, unwrap=False)
pylab.ioff()
pylab.show()
