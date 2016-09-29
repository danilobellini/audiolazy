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
Pitch follower via zero-crossing rate with Tkinter GUI
"""

# ------------------------
# AudioLazy pitch follower
# ------------------------
import sys
from audiolazy import (tostream, zcross, lag2freq, AudioIO, freq2str, sHz,
                       lowpass, envelope, pi, maverage, Stream, thub, chunks)

def limiter(sig, threshold=.1, size=256, env=envelope.rms, cutoff=pi/2048):
  sig = thub(sig, 2)
  return sig * Stream( 1. if el <= threshold else threshold / el
                       for el in maverage(size)(env(sig, cutoff=cutoff)) )

@tostream
def zcross_pitch(sig, size=2048, hop=None):
  for blk in zcross(sig, hysteresis=.01).blocks(size=size, hop=hop):
    crossings = sum(blk)
    yield 0. if crossings == 0 else lag2freq(2. * size / crossings)


def pitch_from_mic(upd_time_in_ms):
  rate = 44100
  s, Hz = sHz(rate)

  api = sys.argv[1] if sys.argv[1:] else None # Choose API via command-line
  chunks.size = 1 if api == "jack" else 16

  with AudioIO(api=api) as recorder:
    snd = recorder.record(rate=rate)
    sndlow = lowpass(400 * Hz)(limiter(snd, cutoff=20 * Hz))
    hop = int(upd_time_in_ms * 1e-3 * s)
    for pitch in freq2str(zcross_pitch(sndlow, size=2*hop, hop=hop) / Hz):
      yield pitch


# ----------------
# GUI with tkinter
# ----------------
if __name__ == "__main__":
  try:
    import tkinter
  except ImportError:
    import Tkinter as tkinter
  import threading
  import re

  # Window (Tk init), text label and button
  tk = tkinter.Tk()
  tk.title(__doc__.strip().splitlines()[0])
  lbldata = tkinter.StringVar(tk)
  lbltext = tkinter.Label(tk, textvariable=lbldata, font=("Purisa", 72),
                          width=10)
  lbltext.pack(expand=True, fill=tkinter.BOTH)
  btnclose = tkinter.Button(tk, text="Close", command=tk.destroy,
                            default="active")
  btnclose.pack(fill=tkinter.X)

  # Needed data
  regex_note = re.compile(r"^([A-Gb#]*-?[0-9]*)([?+-]?)(.*?%?)$")
  upd_time_in_ms = 200

  # Update functions for each thread
  def upd_value(): # Recording thread
    pitches = iter(pitch_from_mic(upd_time_in_ms))
    while not tk.should_finish:
      tk.value = next(pitches)

  def upd_timer(): # GUI mainloop thread
    lbldata.set("\n".join(regex_note.findall(tk.value)[0]))
    tk.after(upd_time_in_ms, upd_timer)

  # Multi-thread management initialization
  tk.should_finish = False
  tk.value = freq2str(0) # Starting value
  lbldata.set(tk.value)
  tk.upd_thread = threading.Thread(target=upd_value)

  # Go
  tk.upd_thread.start()
  tk.after_idle(upd_timer)
  tk.mainloop()
  tk.should_finish = True
  tk.upd_thread.join()
