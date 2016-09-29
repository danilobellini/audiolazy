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
strategies ``pole`` and ``z``, using Sympy.

The single pole at R (or -R) should be real to ensure a real output.
"""
from __future__ import division, print_function, unicode_literals
from functools import reduce
from sympy import (Symbol, preorder_traversal, C, init_printing, S, sympify,
                   exp, expand_complex, cancel, trigsimp, pprint, Eq, factor,
                   solve, I, expand, pi, sin, fraction, pretty, tan)
from collections import OrderedDict
init_printing(use_unicode=True)


def fcompose(*funcs):
  return lambda data: reduce(lambda d, p: p(d), funcs, data)

def has_sqrt(sympy_obj):
  return any(el.func is C.Pow and el.args[-1] is S.Half
             for el in preorder_traversal(sympy_obj))


G = Symbol("G", positive=True) # Gain (linear)
R = Symbol("R", real=True)     # Pole "radius"
w = Symbol("omega", real=True) # Frequency (rad/sample) usually in [0;pi]
z = Symbol("z")                # Z-Transform complex variable


def design_z_filter_single_pole(filt_str, max_gain_freq):
  """
  Finds the coefficients for a simple lowpass/highpass filter.

  This function just prints the coefficient values, besides the given
  filter equation and its power gain. There's 3 constraints used to find the
  coefficients:

  1. The G value is defined by the max gain of 1 (0 dB) imposed at a
     specific frequency
  2. The R value is defined by the 50% power cutoff frequency given in
     rad/sample.
  3. Filter should be stable (-1 < R < 1)

  Parameters
  ----------
  filt_str :
    Filter equation as a string using the G, R, w and z values.
  max_gain_freq :
    A value of zero (DC) or pi (Nyquist) to ensure the max gain as 1 (0 dB).

  Note
  ----
  The R value is evaluated only at pi/4 rad/sample to find whether -1 < R < 1,
  and the max gain is assumed to be either 0 or pi, using other values might
  fail.
  """
  print("H(z) = " + filt_str) # Avoids printing as "1/z"
  filt = sympify(filt_str, dict(G=G, R=R, w=w, z=z))
  print()

  # Finds the power magnitude equation for the filter
  freq_resp = filt.subs(z, exp(I * w))
  frr, fri = freq_resp.as_real_imag()
  power_resp = fcompose(expand_complex, cancel, trigsimp)(frr ** 2 + fri ** 2)
  pprint(Eq(Symbol("Power"), power_resp))
  print()

  # Finds the G value given the max gain value of 1 at the DC or Nyquist
  # frequency. As exp(I*pi) is -1 and exp(I*0) is 1, we can use freq_resp
  # (without "abs") instead of power_resp.
  Gsolutions = factor(solve(Eq(freq_resp.subs(w, max_gain_freq), 1), G))
  assert len(Gsolutions) == 1
  pprint(Eq(G, Gsolutions[0]))
  print()

  # Finds the unconstrained R values for a given cutoff frequency
  power_resp_no_G = power_resp.subs(G, Gsolutions[0])
  half_power_eq = Eq(power_resp_no_G, S.Half)
  Rsolutions = solve(half_power_eq, R)

  # Constraining -1 < R < 1 when w = pi/4 (although the constraint is general)
  Rsolutions_stable = [el for el in Rsolutions if -1 < el.subs(w, pi/4) < 1]
  assert len(Rsolutions_stable) == 1

  # Constraining w to the [0;pi] range, so |sin(w)| = sin(w)
  Rsolution = Rsolutions_stable[0].subs(abs(sin(w)), sin(w))
  pprint(Eq(R, Rsolution))

  # More information about the pole (or -pole)
  print("\n  ** Alternative way to write R **\n")
  if has_sqrt(Rsolution):
    x = Symbol("x") # A helper symbol
    xval = sum(el for el in Rsolution.args if not has_sqrt(el))
    pprint(Eq(x, xval))
    print()
    pprint(Eq(R, expand(Rsolution.subs(xval, x))))
  else:
    # That's also what would be found in a bilinear transform with prewarping
    pprint(Eq(R, Rsolution.rewrite(tan).cancel())) # Not so nice numerically

    # See whether the R denominator can be zeroed
    for root in solve(fraction(Rsolution)[1], w):
      if 0 <= root <= pi:
        power_resp_r = fcompose(expand, cancel)(power_resp_no_G.subs(w, root))
        Rsolutions_r = solve(Eq(power_resp_r, S.Half), R)
        assert len(Rsolutions_r) == 1
        print("\nDenominator is zero for this value of " + pretty(w))
        pprint(Eq(w, root))
        pprint(Eq(R, Rsolutions_r[0]))


filters_data = OrderedDict([
  ("lowpass.pole", # No zeros (constant numerator)
   "G / (1 - R * z ** -1)"),
  ("highpass.pole", # No zeros (constant numerator)
   "G / (1 + R * z ** -1)"),
  ("highpass.z", # Single zero at 1, so gain at the DC level is zero (-inf dB)
   "G * (1 - z ** -1) / (1 - R * z ** -1)"),
  ("lowpass.z", # Single zero at -1, so gain=0 (-inf dB) at the Nyquist freq
   "G * (1 + z ** -1) / (1 + R * z ** -1)"),
])

if __name__ == "__main__":
  for name, filt_str in filters_data.items():
    ftype, fstrategy = name.split(".")
    descr = ("single zero and " if fstrategy == "z" else "") + "single pole"
    msg = "## Filter {name} ({descr} {ftype}) ##".format(**locals())
    msg_detail = "#" * len(msg)
    print(msg_detail, msg, msg_detail, "", sep="\n")
    max_gain_freq = 0 if ftype == "lowpass" else pi
    design_z_filter_single_pole(filt_str, max_gain_freq=max_gain_freq)
    print("\n\n" + " --//-- " * 8 + "\n\n")
