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
Realtime STFT effect to "roll" the magnitude spectrum while keeping the phase
"""

from audiolazy import *
import numpy as np
import sys

@stft(size=2048, hop=682, wnd=window.hann, ola_wnd=window.hann)
def roll_mag(spectrum):
  mag = abs(spectrum)
  phases = np.angle(spectrum)
  return np.roll(mag, 16) * np.exp(1j * phases)

api = sys.argv[1] if sys.argv[1:] else None
chunks.size = 1 if api == "jack" else 16
with AudioIO(True, api=api) as pr:
  pr.play(roll_mag(pr.record()))
