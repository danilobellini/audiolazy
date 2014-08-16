#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of AudioLazy, the signal processing Python package.
# Copyright (C) 2012-2014 Danilo de Jesus da Silva Bellini
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
# Created on Mon Aug 11 02:10:58 2014
# danilo [dot] bellini [at] gmail [dot] com
"""
Digital filter design for the AudioLazy highpass.z filter, with Sympy.

Equation completely defined by:

- Single zero at 1, so gain at the DC level is zero (-inf dB)
- Single pole at R, where R needs to be real since output should be real
- Max gain at pi rad/sample (Nyquist frequency) should be 1 (0 dB)
- R should be defined by the 50% power cutoff frequency given in rad/sample
- Filter should be stable (-1 < R < 1)
"""
from __future__ import division, print_function, unicode_literals
from functools import reduce
from sympy import *
init_printing(use_unicode=True)


def fcompose(*funcs):
  return lambda data: reduce(lambda d, p: p(d), funcs, data)


G = Symbol("G", positive=True) # Gain (linear)
R = Symbol("R", real=True)     # "Radius" (pole magnitude)
w = Symbol("omega", real=True) # Frequency (rad/s) usually in [0;pi]
z = Symbol("z")                # Z-Transform complex variable


# Build the filter
print("## Filter highpass.z, a single zero and single pole highpass")
print()
filt_str = "G * (1 - z ** -1) / (1 - R * z ** -1)"
print("H(z) = " + filt_str) # Avoids printing as "1/z"
filt = sympify(filt_str, locals())
print()

# Finds the power magnitude equation for the filter
freq_resp = filt.subs(z, exp(I * w))
frr, fri = freq_resp.as_real_imag()
power_resp = fcompose(expand_complex, cancel, trigsimp)(frr ** 2 + fri ** 2)
pprint(Eq(Symbol("Power"), power_resp))
print()

# Finds the gain G value given the max value of 1 at the Nyquist frequency.
# As exp(I*pi) is -1, we can use freq_resp instead of power_resp.
Gsolutions = factor(solve(Eq(freq_resp.subs(w, pi), 1), G))
assert len(Gsolutions) == 1
pprint(Eq(G, Gsolutions[0]))
print()

# Finds the unconstrained R values for a given cutoff frequency
power_resp_no_G = power_resp.subs(G, Gsolutions[0])
half_power_eq = Eq(power_resp_no_G, Rational(1, 2))
Rsolutions = solve(half_power_eq, R)

# Constraining -1 < R < 1 when w = pi/4 (although the constraint is general)
Rsolutions_stable = [el for el in Rsolutions if -1 < el.subs(w, pi/4) < 1]
assert len(Rsolutions_stable) == 1

# Constraining w to the [0;pi] range, so |sin(w)| = sin(w)
Rsolution = Rsolutions_stable[0].subs(abs(sin(w)), sin(w))
pprint(Eq(R, Rsolution))
