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

import itertools as it

# Audiolazy internal imports
from .lazy_misc import elementwise

__all__ = ["MIDI_A4", "FREQ_A4", "SEMITONE_RATIO", "midi2freq", "str2midi",
           "str2freq"]

# Useful constants
MIDI_A4 = 69   # MIDI Pitch number
FREQ_A4 = 440. # Hz
SEMITONE_RATIO = 2.**(1./12.) # Ascending


@elementwise("midi_number", 0)
def midi2freq(midi_number):
  """ Given a MIDI pitch number, returns its frequency in Hz. """
  return FREQ_A4 * 2 ** ((midi_number - MIDI_A4) * (1./12.))


@elementwise("note_string", 0)
def str2midi(note_string):
  data = note_string.strip().lower()
  name2delta = {"c": -9, "d": -7, "e": -5, "f": -4, "g": -2, "a": 0, "b": 2}
  accident2delta = {"b": -1, "#": 1, "x": 2}
  accidents = list(it.takewhile(lambda el: el in accident2delta, data[1:]))
  octave_delta = int(data[len(accidents) + 1:]) - 4
  return (MIDI_A4 +
          name2delta[data[0]] + # Name
          sum(accident2delta[ac] for ac in accidents) + # Accident
          12 * octave_delta # Octave
         )


def str2freq(note_string):
  return midi2freq(str2midi(note_string))
