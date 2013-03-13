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
# Created on Fri Nov 02 2012
# danilo [dot] bellini [at] gmail [dot] com
"""
Window functions plot example
"""

from matplotlib import pyplot as plt
from audiolazy import window

M = 256

for func in window:
  plt.plot(func(M), label=func.__name__)
plt.legend(loc="best")
plt.axis(xmin=-5, xmax=M + 5 - 1, ymin=-.05, ymax=1.05)
plt.title("AudioLazy windows for size of {M} samples".format(M=M))
plt.show()
