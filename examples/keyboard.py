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
Musical keyboard synth example with a QWERTY keyboard
"""

from audiolazy import (str2midi, midi2freq, saw_table, sHz, Streamix, Stream,
                       line, AudioIO, chunks)
import sys
try:
  import tkinter
except ImportError:
  import Tkinter as tkinter

keys = "awsedftgyhujkolp;" # Chromatic scale
first_note = str2midi("C3")

pairs = list(enumerate(keys.upper(), first_note + 12)) + \
        list(enumerate(keys, first_note))
notes = {k: midi2freq(idx) for idx, k in pairs}
synth = saw_table

txt = """
Press keys

W E   T Y U   O P
A S D F G H J K L ;

The above should be
seen as piano keys.

Using lower/upper
letters changes the
octave.
"""

tk = tkinter.Tk()
tk.title("Keyboard Example")
lbl = tkinter.Label(tk, text=txt, font=("Mono", 30))
lbl.pack(expand=True, fill=tkinter.BOTH)

rate = 44100
s, Hz = sHz(rate)
ms = 1e-3 * s
attack = 30 * ms
release = 50 * ms
level = .2 # Highest amplitude value per note

smix = Streamix(True)
cstreams = {}

class ChangeableStream(Stream):
  """
  Stream that can be changed after being used if the limit/append methods are
  called while playing. It uses an iterator that keep taking samples from the
  Stream instead of an iterator to the internal data itself.
  """
  def __iter__(self):
    while True:
      yield next(self._data)

has_after = None

def on_key_down(evt):
  # Ignores key up if it came together with a key down (debounce)
  global has_after
  if has_after:
    tk.after_cancel(has_after)
    has_after = None

  ch = evt.char
  if not ch in cstreams and ch in notes:
    # Prepares the synth
    freq = notes[ch]
    cs = ChangeableStream(level)
    env = line(attack, 0, level).append(cs)
    snd = env * synth(freq * Hz)

    # Mix it, storing the ChangeableStream to be changed afterwards
    cstreams[ch] = cs
    smix.add(0, snd)

def on_key_up(evt):
  global has_after
  has_after = tk.after_idle(on_key_up_process, evt)

def on_key_up_process(evt):
  ch = evt.char
  if ch in cstreams:
    cstreams[ch].limit(0).append(line(release, level, 0))
    del cstreams[ch]

tk.bind("<KeyPress>", on_key_down)
tk.bind("<KeyRelease>", on_key_up)

api = sys.argv[1] if sys.argv[1:] else None # Choose API via command-line
chunks.size = 1 if api == "jack" else 16

with AudioIO(api=api) as player:
  player.play(smix, rate=rate)
  tk.mainloop()
