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
Playing Ode to Joy with a "score" written in code.

You can change the synth, the music, the effects, etc.. There are some
comments suggesting some modifications to this script.
"""
from audiolazy import *
import sys

# Initialization
rate = 44100
s, Hz = sHz(rate)
ms = 1e-3 * s
beat = 120 # bpm
quarter_dur = 60 * s / beat


def delay(sig):
  """ Simple feedforward delay effect """
  smix = Streamix()
  sig = thub(sig, 3) # Auto-copy 3 times (remove this line if using feedback)
  smix.add(0, sig)
  # To get a feedback delay, use "smix.copy()" below instead of both "sig"
  smix.add(280 * ms, .1 * sig) # You can also try other constants
  smix.add(220 * ms, .1 * sig)
  return smix
  # When using the feedback delay proposed by the comments, you can use:
  #return smix.limit((1 + sum(dur for n, dur in notes)) * quarter_dur)
  # or something alike (e.g. ensuring that duration outside of this
  # function), helping you to avoid an endless signal.


def note2snd(pitch, quarters):
  """
  Creates an audio Stream object for a single note.

  Parameters
  ----------
  pitch :
    Pitch note like ``"A4"``, as a string, or ``None`` for a rest.
  quarters :
    Duration in quarters (see ``quarter_dur``).
  """
  dur = quarters * quarter_dur
  if pitch is None:
    return zeros(dur)
  freq = str2freq(pitch) * Hz
  return synth(freq, dur)


def synth(freq, dur):
  """
  Synth based on the Karplus-Strong model.

  Parameters
  ----------
  freq :
    Frequency, given in rad/sample.
  dur :
    Duration, given in samples.

  See Also
  --------
  sHz :
    Create constants ``s`` and ``Hz`` for converting "rad/sample" and
    "samples" to/from "seconds" and "hertz" using expressions like "440 * Hz".
  """
  return karplus_strong(freq, tau=800*ms).limit(dur)


## Uncomment these lines to use a "8-bits"-like synth
#square_table = TableLookup([-1] * 512 + [1] * 512)
#adsr_params = dict(a=30*ms, d=150*ms, s=.6, r=100*ms)
#def synth(freq, dur, model=square_table):
#  """ Table-lookup synth with an ADSR envelope. """
#  # Why not trying "sinusoid" and "saw_table" instead of the "square_table"?
#  return model(freq) * adsr(dur, **adsr_params)


## Uncomment these lines to get a more "noisy" synth
#sin_cube_table = sin_table ** 3
#def synth(freq, dur):
#  env = adsr(dur, a=dur/3, d=dur/4, s=.1, r=5 * dur / 12)
#  env1 = env.copy() * white_noise(low=.9, high=1)
#  env2 = env * white_noise(low=.9, high=1)
#  sig1 = saw_table(gauss_noise(mu=freq, sigma=freq * .03)) * env1
#  sig2 = sin_cube_table(gauss_noise(mu=freq, sigma=freq * .03)) * env2
#  return .4 * sig1 + .6 * sig2


# Musical "score"
notes = [
  ("D4", 1), ("D4", 1), ("Eb4", 1), ("F4", 1),
  ("F4", 1), ("Eb4", 1), ("D4", 1), ("C4", 1),
  ("Bb3", 1), ("Bb3", 1), ("C4", 1), ("D4", 1),
  ("D4", 1.5), ("C4", .5), ("C4", 1.5), (None, .5),

  ("D4", 1), ("D4", 1), ("Eb4", 1), ("F4", 1),
  ("F4", 1), ("Eb4", 1), ("D4", 1), ("C4", 1),
  ("Bb3", 1), ("Bb3", 1), ("C4", 1), ("D4", 1),
  ("C4", 1.5), ("Bb3", .5), ("Bb3", 1.5), (None, .5),
]


# Creates the music (lazily)
# See play_bach_choral.py for a polyphonic (and safer) way to achieve this
music = chain.from_iterable(starmap(note2snd, notes))
#music = atan(15 * music) # Uncomment this to apply a simple dirtortion effect
music = delay(.9 * music) # Uncomment this to apply a simple delay effect


# Play it!
music.append(zeros(.5 * s)) # Avoids an ending "click"
api = sys.argv[1] if sys.argv[1:] else None # Choose API via command-line
chunks.size = 1 if api == "jack" else 16
with AudioIO(True, api=api) as player:
  player.play(music, rate=rate)
