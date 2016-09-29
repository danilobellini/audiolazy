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
FM synthesis benchmarking
"""

from __future__ import unicode_literals, print_function
from timeit import timeit
import sys


# ===================
# Some initialization
# ===================
num_tests = 30
is_pypy = any(name.startswith("pypy") for name in dir(sys))
if is_pypy:
  print("PyPy detected!")
  print()
  numpy_name = "numpypy"
else:
  numpy_name = "numpy"


# ======================
# AudioLazy benchmarking
# ======================
kws = {}
kws["setup"] = """
from audiolazy import sHz, adsr, sinusoid
from math import pi
"""
kws["number"] = num_tests
kws["stmt"] = """
s, Hz = sHz(44100)
ms = 1e-3 * s
env = adsr(dur=5*s, a=20*ms, d=30*ms, s=.8, r=50*ms)
sin_data = sinusoid(freq=440*Hz,
                    phase=sinusoid(220*Hz) * pi)
result = sum(env * sin_data)
"""

print("=== AudioLazy benchmarking ===")
print("Trials:", kws["number"])
print()
print("Setup code:")
print(kws["setup"])
print()
print("Benchmark code (also executed once as 'setup'/'training'):")
kws["setup"] += kws["stmt"] # Helpful for PyPy
print(kws["stmt"])
print()
print("Mean time (milliseconds):")
print(timeit(**kws) * 1e3 / num_tests)
print("==============================")
print()


# ==================
# Numpy benchmarking
# ==================
kws_np = {}
kws_np["setup"] = "import {0} as np".format(numpy_name)
kws_np["number"] = num_tests
kws_np["stmt"] = """
rate = 44100
dur = 5 * rate
sustain_level = .8
# The np.linspace isn't in numpypy yet; it uses float64
attack = np.linspace(0., 1., num=np.round(20e-3 * rate), endpoint=False)
decay = np.linspace(1., sustain_level, num=np.round(30e-3 * rate),
                    endpoint=False)
release = np.linspace(sustain_level, 0., num=np.round(50e-3 * rate),
                      endpoint=False)
sustain_dur = dur - len(attack) - len(decay) - len(release)
sustain = sustain_level * np.ones(sustain_dur)
env = np.hstack([attack, decay, sustain, release])
freq220 = 220 * 2 * np.pi / rate
freq440 = 440 * 2 * np.pi / rate
phase220 = np.arange(dur, dtype=np.float64) * freq220
phase440 = np.arange(dur, dtype=np.float64) * freq440
sin_data = np.sin(phase440 + np.sin(phase220) * np.pi)
result = np.sum(env * sin_data)
"""

# Alternative for numpypy (since it don't have "linspace" nor "hstack")
stmt_npp = """
rate = 44100
dur = 5 * rate
sustain_level = .8
len_attack = int(round(20e-3 * rate))
attack = np.arange(len_attack, dtype=np.float64) / len_attack
len_decay = int(round(30e-3 * rate))
decay = (np.arange(len_decay - 1, -1, -1, dtype=np.float64
                  ) / len_decay) * (1 - sustain_level) + sustain_level
len_release = int(round(50e-3 * rate))
release = (np.arange(len_release - 1, -1, -1, dtype=np.float64
                    ) / len_release) * sustain_level
env = np.ndarray(dur, dtype=np.float64)
env[:len_attack] = attack
env[len_attack:len_attack+len_decay] = decay
env[len_attack+len_decay:dur-len_release] = sustain_level
env[dur-len_release:dur] = release
freq220 = 220 * 2 * np.pi / rate
freq440 = 440 * 2 * np.pi / rate
phase220 = np.arange(dur, dtype=np.float64) * freq220
phase440 = np.arange(dur, dtype=np.float64) * freq440
sin_data = np.sin(phase440 + np.sin(phase220) * np.pi)
result = np.sum(env * sin_data)
"""

try:
  if is_pypy:
    import numpypy as np
  else:
    import numpy as np
except ImportError:
  print("Numpy not found. Finished benchmarking!")
else:
  if is_pypy:
    kws_np["stmt"] = stmt_npp

  print("Numpy found!")
  print()
  print("=== Numpy benchmarking ===")
  print("Trials:", kws_np["number"])
  print()
  print("Setup code:")
  print(kws_np["setup"])
  print()
  print("Benchmark code (also executed once as 'setup'/'training'):")
  kws_np["setup"] += kws_np["stmt"] # Helpful for PyPy
  print(kws_np["stmt"])
  print()
  print("Mean time (milliseconds):")
  print(timeit(**kws_np) * 1e3 / num_tests)
  print("==========================")
