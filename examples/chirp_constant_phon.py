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
Constant phon (ISO226) sinusoid glissando/chirp/glide

Plays a chirp from ``fstart`` to ``fend``, but a fade in/out is
done on each of these two frequencies, before/after the chirp is played
and with the same duration each, using the ``total_duration`` and
``chirp_duration`` values.

Obviously the actual sound depends on the hardware, and a constant phon curve
needs a hardware dB SPL calibration for the needed frequency range.

Besides AudioLazy, this example needs Scipy and Pyaudio.
"""
from audiolazy import *
if PYTHON2:
  input = raw_input

rate = 44100 # samples/s
fstart, fend = 16, 20000 # Hz
intensity = 50 # phons
chirp_duration = 5 # seconds
total_duration = 9 # seconds

assert total_duration > chirp_duration

def finalize(zeros_dur):
  print("Finished!")
  for el in zeros(zeros_dur):
    yield el

def dB2magnitude(logpower):
  return 10 ** (logpower / 20)

s, Hz = sHz(rate)
freq2dB = phon2dB.iso226(intensity)

freq = thub(2 ** line(chirp_duration * s, log2(fstart), log2(fend)), 2)
gain = thub(dB2magnitude(freq2dB(freq)), 2)
maxgain = max(gain)

unclick_dur = rint((total_duration - chirp_duration) * s / 2)
gstart = line(unclick_dur, 0, dB2magnitude(freq2dB(fstart)) / maxgain)
gend = line(unclick_dur, dB2magnitude(freq2dB(fend)) / maxgain, 0)

sfreq = chain(repeat(fstart, unclick_dur), freq, repeat(fend, unclick_dur))
sgain = chain(gstart, gain / maxgain, gend)

snd = sinusoid(sfreq * Hz) * sgain

with AudioIO(True) as player:
  refgain = dB2magnitude(freq2dB(1e3)) / maxgain
  th = player.play(sinusoid(1e3 * Hz) * refgain)
  input("Playing the 1 kHz reference tone. You should calibrate the output "
        "to get {0} dB SPL and press enter to continue.".format(intensity))
  th.stop()
  print("Playing the chirp!")
  player.play(chain(snd, finalize(.5 * s)), rate=rate)
