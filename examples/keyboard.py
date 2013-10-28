#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of AudioLazy, the signal processing Python package.
# Copyright (C) 2012-2013 Danilo de Jesus da Silva Bellini
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
#
# Created on Wed Oct 16 13:44:10 2013
# danilo [dot] bellini [at] gmail [dot] com
"""
Musical keyboard synth example with a QWERTY keyboard
"""

from audiolazy import *
try:
  import tkinter
except ImportError:
  import Tkinter as tkinter

keys = "awsedftgyhujkolp;" # Chromatic scale
first_note = str2midi("C3")

pairs = list(enumerate(keys.upper(), 12)) + list(enumerate(keys))
notes = {k: midi2freq(first_note + idx) for idx, k in pairs}
threads = {}
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

with AudioIO() as player:

  def on_key_down(evt):
    ch = evt.char
    if not ch in threads and ch in notes:
      freq = notes[ch]
      snd = line(50 * ms, 0, .2).append(.2) * synth(freq * Hz)
      threads[ch] = player.play(snd, rate=rate)

  def on_key_up(evt):
    ch = evt.char
    if ch in threads:
      threads[ch].stop()
      del threads[ch]

  tk.bind("<KeyPress>", on_key_down)
  tk.bind("<KeyRelease>", on_key_up)

  tk.mainloop()
