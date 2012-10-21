#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing module for the lazy_auditory module

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

Created on Sun Sep 09 2012
danilo [dot] bellini [at] gmail [dot] com
"""

import pytest
p = pytest.mark.parametrize

# Audiolazy internal imports
from ..lazy_auditory import erb
from ..lazy_misc import almost_eq

class TestERB(object):

  @p(("freq", "bandwidth"),
     [(1000, 132.639),
      (3000, 348.517),
     ])
  def test_glasberg_moore_slaney_example(self, freq, bandwidth):
    assert almost_eq(erb["gm90"](freq), bandwidth)
