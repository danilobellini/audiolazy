#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing module for the lazy_filters module by using scipy as an oracle

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

Created on Sat Oct 06 2012
danilo [dot] bellini [at] gmail [dot] com
"""

import pytest
p = pytest.mark.parametrize

from scipy import signal as ss

# Audiolazy internal imports
from ..lazy_filters import LTIFreq
from ..lazy_misc import almost_eq


@p("a", [[1.], [3.], [1., 3.], [15., -17.2], [-18., 9.8, 0., 14.3]])
@p("b", [[1.], [-1.], [1., 0., -1.], [1., 3.]])
@p("data", [range(5), range(5, 0, -1), [7, 22, -5], [8., 3., 15.]])
def test_lfilter(a, b, data):
  filt = LTIFreq(b, a)
  expected = ss.lfilter(b, a, data).tolist()
  assert almost_eq(filt(data), expected)