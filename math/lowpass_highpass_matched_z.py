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
Digital filter design for the AudioLazy lowpass and highpass filters
strategies ``pole_exp`` and ``z_exp`` from analog design via the
matching Z-transform technique, using Sympy.

This script just prints the coefficient values, the filter equations and
power gain for the analog (Laplace transform), digital (Z-Transform) and
mirrored digital filters. About the analog filter, there's some constraints
used to find the coefficients:

- The g "gain" value is found from the max gain of 1 (0 dB) imposed at a
  specific frequency (zero for lowpass, symbolic "infinite" for highpass).
- The p value is found from the 50% power cutoff frequency given in rad/s.
- The single pole at p should be real to ensure a real output.
- The Laplace equation pole should be negative to ensure stability.

Note that this matching procedure is an approximation. For precise values for
the coefficients, you should look for a design technique that works directly
with digital filters, or perhaps a numerical approach.
"""
from __future__ import division, print_function, unicode_literals
from sympy import (Symbol, init_printing, S, sympify, exp, cancel, pprint, Eq,
                   factor, solve, I, pi, oo, limit)
init_printing(use_unicode=True)


# Symbols used
g = Symbol("g", positive=True)     # Analog gain (linear)
p = Symbol("p", real=True)         # Laplace pole
f = Symbol("Omega", positive=True) # Frequency in rad/s (analog)
s = Symbol("s")                    # Laplace Transform complex variable

rate = Symbol("rate", positive=True) # Rate in samples/s

G = Symbol("G", positive=True)     # Digital gain (linear)
R = Symbol("R", real=True)         # Digital pole ("radius")
w = Symbol("omega", real=True)     # Frequency (rad/sample) usually in [0;pi]
z = Symbol("z")                    # Z-Transform complex variable


for max_gain_freq in [0, oo]: # Freq whose gain is max and equal to 1 (0 dB).
  has_zero = max_gain_freq != 0

  # Build some useful strings from the parameters
  if has_zero:                    # See the "Matching Z-Transform" comment
    afilt_str = "g * s / (s - p)" # for more details on the filt_str values
    filt_str = "G * (1 - z ** -1) / (1 - R * z ** -1)" # Single zero at exp(0)
    strategy = "z_exp"
    prefix, mprefix = ["high", "low"]
  else:
    afilt_str = "g / (s - p)"
    filt_str = "G / (1 - R * z ** -1)"
    strategy = "pole_exp"
    prefix, mprefix = ["low", "high"]
  filt_name = prefix + "pass." + strategy
  mfilt_name = mprefix + "pass." + strategy

  # Output header
  xtra_descr = "and single zero " if has_zero else ""
  msg = "## Laplace single pole {}{}pass filter ##".format(xtra_descr, prefix)
  msg_detail = "#" * len(msg)
  print(msg_detail, msg, msg_detail, sep="\n")

  # Creates the analog filter sympy object
  print("\n  ** Analog design (Laplace Transform) **\n")
  print("H(s) = " + afilt_str)
  afilt = sympify(afilt_str, dict(g=g, p=p, s=s))
  print()

  # Finds the power magnitude equation for the filter
  freq_resp = afilt.subs(s, I * f)
  frr, fri = freq_resp.as_real_imag()
  power_resp = cancel(frr ** 2 + fri ** 2)
  pprint(Eq(Symbol("Power"), power_resp))
  print()

  # Finds the g value given the max gain value of 1 at the DC frequency. As
  # I*0 is zero and I*s cancels the imaginary unit at the limit when s -> oo
  # (as both numerator and denominator becomes purely complex numbers),
  # we can use freq_resp (without "abs") instead of power_resp.
  gsolutions = factor(solve(Eq(limit(freq_resp, f, max_gain_freq), 1), g))
  assert len(gsolutions) == 1
  pprint(Eq(g, gsolutions[0]))
  print()

  # Finds the p value for a given cutoff frequency, imposing stability (p < 0)
  power_resp_no_g = power_resp.subs(g, gsolutions[0])
  half_power_eq = Eq(power_resp_no_g, S.Half)
  psolutions_stable = [el for el in solve(half_power_eq, p) if el < 0]
  assert len(psolutions_stable) == 1
  psolution = psolutions_stable[0]
  pprint(Eq(p, psolution))

  # Creates the digital filter sympy object
  print("\n  ** Digital design (Z-Transform) for {} **\n".format(filt_name))
  print("H(z) = " + filt_str)
  filt = sympify(filt_str, dict(G=G, R=R, z=z))
  print()

  # Matching Z-Transform
  # for each zero/pole in both Laplace and Z-Transform equations,
  #   z_zp = exp(s_zp / rate)
  # where z_zp and s_zp are a single zero/pole for such equations
  Rsolution = exp(psolution / rate).subs(f, w * rate)
  pprint(Eq(R, Rsolution))
  print()

  # Finds the G value that fits with the Laplace filter for a single
  # frequency: the max_gain_freq (and its matched Z-Transform value)
  gain_eq = Eq(filt.subs(z, 1 if max_gain_freq == 0 else -1),
               limit(afilt, s, max_gain_freq).subs(g, gsolutions[0]))
  Gsolutions = factor(solve(gain_eq, G))
  assert len(Gsolutions) == 1
  pprint(Eq(G, Gsolutions[0]))

  # Mirroring the Z-Transform linear frequency response, so:
  #   z_mirror = -z
  #   w_mirror = pi - w
  print("\n  ** Mirroring lowpass.pole_exp to get the {} **\n"
        .format(mfilt_name))
  mfilt_str = filt_str.replace(" - ", " + ")
  print("H(z) = " + mfilt_str)
  mfilt = sympify(mfilt_str, dict(G=G, R=R, z=z))
  assert filt.subs(z, -z) == mfilt
  print()
  pprint(Eq(R, Rsolution.subs(w, pi - w)))
  print()
  pprint(Eq(G, Gsolutions[0])) # This was kept
  print()