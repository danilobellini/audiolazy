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
Realtime STFT effect to robotize a voice (or anything else)

This is done by removing (zeroing) the phases, which means a single spectrum
block processing function that keeps the magnitudes and removes the phase, a
function a.k.a. "abs", the absolute value. The initial zero-phasing isn't
needed at all since the phases are going to be removed, so the "before" step
can be safely removed.
"""

from audiolazy import window, stft, chunks, AudioIO
import sys

wnd = window.hann
robotize = stft(abs, size=1024, hop=441, before=None, wnd=wnd, ola_wnd=wnd)

api = sys.argv[1] if sys.argv[1:] else None
chunks.size = 1 if api == "jack" else 16
with AudioIO(True, api=api) as pr:
  pr.play(robotize(pr.record()))
