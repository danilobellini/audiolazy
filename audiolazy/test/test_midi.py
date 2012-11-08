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

import pytest
p = pytest.mark.parametrize

# Audiolazy internal imports
from ..lazy_midi import MIDI_A4, FREQ_A4, SEMITONE_RATIO, midi2freq, str2midi
from ..lazy_misc import almost_eq


class TestMIDI2Freq(object):
  table = [(MIDI_A4, FREQ_A4),
           (MIDI_A4 + 12, FREQ_A4 * 2),
           (MIDI_A4 + 24, FREQ_A4 * 4),
           (MIDI_A4 - 12, FREQ_A4 * .5),
           (MIDI_A4 - 24, FREQ_A4 * .25),
           (MIDI_A4 + 1, FREQ_A4 * SEMITONE_RATIO),
           (MIDI_A4 + 2, FREQ_A4 * SEMITONE_RATIO ** 2),
           (MIDI_A4 - 1, FREQ_A4 / SEMITONE_RATIO),
           (MIDI_A4 - 13, FREQ_A4 * .5 / SEMITONE_RATIO),
           (MIDI_A4 - 3, FREQ_A4 / SEMITONE_RATIO ** 3),
           (MIDI_A4 - 11, FREQ_A4 * SEMITONE_RATIO / 2),
          ]

  @p(("note", "freq"), table)
  def test_single_note(self, note, freq):
    assert almost_eq(midi2freq(note), freq)

  @p("data_type", [tuple, list])
  def test_note_list_tuple(self, data_type):
    notes, freqs = zip(*self.table)
    assert almost_eq(midi2freq(data_type(notes)), data_type(freqs))


class TestStr2MIDI(object):
  table = [("A4", MIDI_A4),
           ("A5", MIDI_A4 + 12),
           ("A3", MIDI_A4 - 12),
           ("Bb4", MIDI_A4 + 1),
           ("B4", MIDI_A4 + 2),
           ("C5", MIDI_A4 + 3),
           ("C#5", MIDI_A4 + 4),
           ("Db3", MIDI_A4 - 20),
          ]

  @p(("name", "note"), table)
  def test_single_name(self, name, note):
    assert str2midi(name) == note
    assert str2midi(name.lower()) == note
    assert str2midi(name.upper()) == note
    assert str2midi("  " + name + " ") == note

  @p("data_type", [tuple, list])
  def test_name_list_tuple(self, data_type):
    names, notes = zip(*self.table)
    assert str2midi(data_type(names)) == data_type(notes)
