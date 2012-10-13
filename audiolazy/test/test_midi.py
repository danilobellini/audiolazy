#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing module for the lazy_midi module

Copyright (C) 2012 Danilo de Jesus da Silva Bellini

This file is part of AudioLazy, the signal processing Python package.

AudioLazy is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

Created on Tue Jul 31 2012
danilo [dot] bellini [at] gmail [dot] com
"""

# Audiolazy internal imports
from ..lazy_midi import MIDI_A4, FREQ_A4, SEMITONE_RATIO, midi2freq
from ..lazy_misc import almost_eq


def test_midi2freq():
  assert almost_eq(midi2freq(MIDI_A4), FREQ_A4)
  assert almost_eq(midi2freq(MIDI_A4+12), FREQ_A4*2)
  assert almost_eq(midi2freq(MIDI_A4+24), FREQ_A4*4)
  assert almost_eq(midi2freq(MIDI_A4-12), FREQ_A4*.5)
  assert almost_eq(midi2freq(MIDI_A4-24), FREQ_A4*.25)
  assert almost_eq(midi2freq(MIDI_A4+1), FREQ_A4*SEMITONE_RATIO)
  assert almost_eq(midi2freq(MIDI_A4+2), FREQ_A4*SEMITONE_RATIO**2)
  assert almost_eq(midi2freq(MIDI_A4-1), FREQ_A4/SEMITONE_RATIO)
  assert almost_eq(midi2freq(MIDI_A4-13), FREQ_A4*.5/SEMITONE_RATIO)
