#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIDI representation data & note-frequency relationship

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

Created on Wed Jul 18 2012
danilo [dot] bellini [at] gmail [dot] com
"""

# Useful constants
MIDI_A4 = 69   # MIDI Pitch number
FREQ_A4 = 440. # Hz
SEMITONE_RATIO = 2.**(1./12.) # Ascending


def midi2freq(midi_number):
  """ Given a MIDI pitch number, returns its frequency in Hz. """
  return FREQ_A4 * 2 ** ((midi_number - MIDI_A4) * (1./12.))
