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
Testing module for the lazy_synth module by using numpy as an oracle
"""

import pytest
p = pytest.mark.parametrize

import numpy as np
from math import pi

# Audiolazy internal imports
from ..lazy_misc import almost_eq, sHz
from ..lazy_synth import adsr, sinusoid


def test_adsr():
  rate = 44100
  dur = 3 * rate
  sustain_level = .8
  attack = np.linspace(0., 1., num=np.round(20e-3 * rate), endpoint=False)
  decay = np.linspace(1., sustain_level, num=np.round(30e-3 * rate),
                      endpoint=False)
  release = np.linspace(sustain_level, 0., num=np.round(50e-3 * rate),
                        endpoint=False)
  sustain_dur = dur - len(attack) - len(decay) - len(release)
  sustain = sustain_level * np.ones(sustain_dur)
  env = np.hstack([attack, decay, sustain, release])

  s, Hz = sHz(rate)
  ms = 1e-3 * s
  assert almost_eq(env, adsr(dur=3*s, a=20*ms, d=30*ms, s=.8, r=50*ms))


def test_sinusoid():
  rate = 44100
  dur = 3 * rate

  freq220 = 220 * (2 * np.pi / rate)
  freq440 = 440 * (2 * np.pi / rate)
  phase220 = np.arange(dur, dtype=np.float64) * freq220
  phase440 = np.arange(dur, dtype=np.float64) * freq440
  sin_data = np.sin(phase440 + np.sin(phase220) * np.pi)

  s, Hz = sHz(rate)
  assert almost_eq.diff(sin_data,
                        sinusoid(freq=440*Hz,
                                 phase=sinusoid(220*Hz) * pi
                                ).take(int(3 * s)),
                        max_diff=1e-8
                       )
