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
Calculate "pi" using the Madhava-Gregory-Leibniz series and Machin formula
"""

from __future__ import division, print_function
from audiolazy import Stream, thub, count, z, pi # For comparison

def mgl_seq(x):
  """
  Sequence whose sum is the Madhava-Gregory-Leibniz series.

    [x,  -x^3/3, x^5/5, -x^7/7, x^9/9, -x^11/11, ...]

  Returns
  -------
    An endless sequence that has the property
    ``atan(x) = sum(mgl_seq(x))``.
    Usually you would use the ``atan()`` function, not this one.

  """
  odd_numbers = thub(count(start=1, step=2), 2)
  return Stream(1, -1) * x ** odd_numbers / odd_numbers


def atan_mgl(x, n=10):
  """
  Finds the arctan using the Madhava-Gregory-Leibniz series.
  """
  acc = 1 / (1 - z ** -1) # Accumulator filter
  return acc(mgl_seq(x)).skip(n-1).take()


if __name__ == "__main__":
  print("Reference (for comparison):", repr(pi))
  print()

  print("Machin formula (fast)")
  pi_machin = 4 * (4 * atan_mgl(1/5) - atan_mgl(1/239))
  print("Found:", repr(pi_machin))
  print("Error:", repr(abs(pi - pi_machin)))
  print()

  print("Madhava-Gregory-Leibniz series for 45 degrees (slow)")
  pi_mgl_series = 4 * atan_mgl(1, n=1e6) # Sums 1,000,000 items...slow...
  print("Found:", repr(pi_mgl_series))
  print("Error:", repr(abs(pi - pi_mgl_series)))
  print()
