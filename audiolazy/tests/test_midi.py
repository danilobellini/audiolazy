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
Testing module for the lazy_midi module
"""

import pytest
p = pytest.mark.parametrize

from random import random

# Audiolazy internal imports
from ..lazy_midi import (MIDI_A4, FREQ_A4, SEMITONE_RATIO, midi2freq,
                         str2midi, freq2midi, midi2str)
from ..lazy_misc import almost_eq
from ..lazy_compat import xzip
from ..lazy_math import inf, nan, isinf, isnan


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
    notes, freqs = xzip(*self.table)
    assert almost_eq(midi2freq(data_type(notes)), data_type(freqs))

  invalid_table = [
    (inf, lambda x: isinf(x) and x > 0),
    (-inf, lambda x: x == 0),
    (nan, isnan),
  ]
  @p(("note", "func_result"), invalid_table)
  def test_invalid_inputs(self, note, func_result):
    assert func_result(midi2freq(note))


class TestFreq2MIDI(object):
  @p(("note", "freq"), TestMIDI2Freq.table)
  def test_single_note(self, note, freq):
    assert almost_eq(freq2midi(freq), note)

  invalid_table = [
    (inf, lambda x: isinf(x) and x > 0),
    (0, lambda x: isinf(x) and x < 0),
    (-1, isnan),
    (-inf, isnan),
    (nan, isnan),
  ]
  @p(("freq", "func_result"), invalid_table)
  def test_invalid_inputs(self, freq, func_result):
    assert func_result(freq2midi(freq))


class TestStr2MIDI(object):
  table = [("A4", MIDI_A4),
           ("A5", MIDI_A4 + 12),
           ("A3", MIDI_A4 - 12),
           ("Bb4", MIDI_A4 + 1),
           ("B4", MIDI_A4 + 2), # TestMIDI2Str.test_name_with_errors:
           ("C5", MIDI_A4 + 3), # These "go beyond" octave by a small amount
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
    names, notes = xzip(*self.table)
    assert str2midi(data_type(names)) == data_type(notes)

  def test_interrogation_input(self):
    assert isnan(str2midi("?"))


class TestMIDI2Str(object):
  @p(("name", "note"), TestStr2MIDI.table)
  def test_single_name(self, name, note):
    assert midi2str(note, sharp="#" in name) == name

  @p(("name", "note"), TestStr2MIDI.table)
  def test_name_with_errors(self, name, note):
    error = round(random() / 3 + .1, 3) # Minimum is greater than tolerance

    full_name = name + "+{}%".format("%.1f" % (error * 100))
    assert midi2str(note + error, sharp="#" in name) == full_name

    full_name = name + "-{}%".format("%.1f" % (error * 100))
    assert midi2str(note - error, sharp="#" in name) == full_name

  @p("note", [inf, -inf, nan])
  def test_interrogation_output(self, note):
    assert midi2str(note) == "?"
