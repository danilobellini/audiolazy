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
Digital filter design for the simplest lowpass and highpass filters using the
bilinear transformation method with prewarping, using Sympy. The results
matches the exact AudioLazy filter strategies ``highpass.z`` and
``lowpass.z``.
"""
from __future__ import division, print_function, unicode_literals
from audiolazy import Stream
from sympy import (Symbol, init_printing, sympify, exp, pprint, Eq,
                   factor, solve, I, sin, tan, pretty, together, radsimp)
init_printing(use_unicode=True)


def print_header(msg):
  msg_full = " ".join(["##", msg, "##"])
  msg_detail = "#" * len(msg_full)
  print(msg_detail, msg_full, msg_detail, sep="\n")

def taylor(f, n=2, **kwargs):
  """
  Taylor/Mclaurin polynomial aproximation for the given function.
  The ``n`` (default 2) is the amount of aproximation terms for ``f``. Other
  arguments are keyword-only and will be passed to the ``f.series`` method.
  """
  return sum(Stream(f.series(n=None, **kwargs)).limit(n))


# Symbols used
p = Symbol("p", real=True)         # Laplace pole
f = Symbol("Omega", positive=True) # Frequency in rad/s (analog)
s = Symbol("s")                    # Laplace Transform complex variable

rate = Symbol("rate", positive=True) # Rate in samples/s

G = Symbol("G", positive=True)     # Digital gain (linear)
R = Symbol("R", real=True)         # Digital pole ("radius")
w = Symbol("omega", real=True)     # Frequency (rad/sample) usually in [0;pi]
z = Symbol("z")                    # Z-Transform complex variable

zinv = Symbol("z^-1")              # z ** -1


# Bilinear transform equation
print_header("Bilinear transformation method")
print("\nLaplace and Z Transforms are related by:")
pprint(Eq(z, exp(s / rate)))

print("\nBilinear transform approximation (no prewarping):")
z_num = exp( s / (2 * rate))
z_den = exp(-s / (2 * rate))
assert z_num / z_den == exp(s / rate)
z_bilinear = together(taylor(z_num, x=s, x0=0) / taylor(z_den, x=s, x0=0))
pprint(Eq(z, z_bilinear))

print("\nWhich also means:")
s_bilinear = solve(Eq(z, z_bilinear), s)[0]
pprint(Eq(s, radsimp(s_bilinear.subs(z, 1 / zinv))))

print("\nPrewarping H(z) = H(s) at a frequency " +
      pretty(w) + " (rad/sample) to " +
      pretty(f) + " (rad/s):")
pprint(Eq(z, exp(I * w)))
pprint(Eq(s, I * f))
f_prewarped = (s_bilinear / I).subs(z, exp(I * w)).rewrite(sin) \
                                                  .rewrite(tan).cancel()
pprint(Eq(f, f_prewarped))


# Lowpass/highpass filters with prewarped bilinear transform equation
T = tan(w / 2)
for name, afilt_str in [("high", "s / (s - p)"),
                        ("low", "-p / (s - p)")]:
  print()
  print_header("Laplace {0}pass filter (matches {0}pass.z)".format(name))
  print("\nFilter equations:")
  print("H(s) = " + afilt_str)
  afilt = sympify(afilt_str, dict(p=-f, s=s))
  pprint(Eq(p, -f)) # Proof is given in lowpass_highpass_matched_z.py
  print("where " + pretty(f) + " is the cut-off frequency in rad/s.")

  print("\nBilinear transformation (prewarping at the cut-off frequency):")
  filt = afilt.subs({f: f_prewarped,
                     s: s_bilinear,
                     z: 1 / zinv}).cancel().collect(zinv)
  pprint(Eq(Symbol("H(z)"), (filt)))
  print("where " + pretty(w) + " is the cut-off frequency in rad/sample.")

  print("\nThe single pole found is:")
  pole = 1 / solve(filt.as_numer_denom()[1], zinv)[0]
  pprint(Eq(Symbol("pole"), pole))

  print("\nSo we can assume ...")
  R_subs = -pole if name == "low" else pole
  RT_eq = Eq(R, R_subs)
  pprint(RT_eq)

  print("\n... and get a simpler equation:")
  pprint(Eq(Symbol("H(z)"), factor(filt.subs(T, solve(RT_eq, T)[0]))))
