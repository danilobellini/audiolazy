# -*- coding: utf-8 -*-
"""
Peripheral auditory modeling module

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

Created on Fri Sep 21 2012
danilo [dot] bellini [at] gmail [dot] com
"""

# Audiolazy internal imports
from .lazy_core import StrategyDict
from .lazy_misc import elementwise
from .lazy_filters import z, CascadeFilter, ZFilter, resonator
from .lazy_math import pi, exp, cos, sin, sqrt, factorial
from .lazy_stream import thub

__all__ = ["erb", "gammatone", "gammatone_erb_constants"]


erb = StrategyDict("erb")
gammatone = StrategyDict("gammatone")


@erb.strategy("gm90", "glasberg_moore_90", "glasberg_moore")
@elementwise("freq", 0)
def erb(freq, Hz=None):
  """
  ERB model from:

    ``B. R. Glasberg and B. C. J. Moore, "Derivation of auditory filter
    shapes from notched-noise data". Hearing Research, vol. 47, 1990, pp.
    103-108.``

  Parameters
  ----------
  freq :
    Frequency, in rad/sample if second parameter is given, in Hz otherwise.
  Hz :
    Frequency conversion "Hz" from sHz function, i.e., ``sHz(rate)[1]``.
    If this value is not given, both input and output will be in Hz.

  Returns
  -------
  Frequency range size, in rad/sample if second parameter is given, in Hz
  otherwise.

  """
  if Hz is None:
    if freq < 7: # Perhaps user tried something up to 2 * pi
      raise ValueError("Frequency out of range.")
    Hz = 1
  fHz = freq / Hz
  result = 24.7 * (4.37e-3 * fHz + 1.)
  return result * Hz


@erb.strategy("mg83", "moore_glasberg_83")
@elementwise("freq", 0)
def erb(freq, Hz=None):
  """
  ERB model from:

    ``B. C. J. Moore and B. R. Glasberg, "Suggested formulae for calculating
    auditory filter bandwidths and excitation patterns". J. Acoust. Soc.
    Am., 74, 1983, pp. 750-753.``

  Parameters
  ----------
  freq :
    Frequency, in rad/sample if second parameter is given, in Hz otherwise.
  Hz :
    Frequency conversion "Hz" from sHz function, i.e., ``sHz(rate)[1]``.
    If this value is not given, both input and output will be in Hz.

  Returns
  -------
  Frequency range size, in rad/sample if second parameter is given, in Hz
  otherwise.

  """
  if Hz is None:
    if freq < 7: # Perhaps user tried something up to 2 * pi
      raise ValueError("Frequency out of range.")
    Hz = 1
  fHz = freq / Hz
  result = 6.23e-6 * fHz ** 2 + 93.39e-3 * fHz + 28.52
  return result * Hz


def gammatone_erb_constants(n):
  """
  Constants for using the real bandwidth in the gammatone filter, given its
  order. Returns a pair ``(x, y) = (1/a_n, c_n)``, based on equations from:

    ``Holdsworth, J.; Patterson, R.; Nimmo-Smith I.; Rice, P. Implementing a
    GammaTone Filter Bank. In: SVOS Final Report, Annex C, Part A: The
    Auditory Filter Bank. 1988.``

  First returned value is a bandwidth compensation for direct use in the
  gammatone formula:

    >>> x, y = gammatone_erb_constants(4)
    >>> central_frequency = 1000
    >>> round(x, 3)
    1.019
    >>> bandwidth = x * erb["moore_glasberg_83"](central_frequency)
    >>> round(bandwidth, 2)
    130.52

  Second returned value helps us find the ``3 dB`` bandwidth as:

    >>> x, y = gammatone_erb_constants(4)
    >>> central_frequency = 1000
    >>> bandwidth3dB = x * y * erb["moore_glasberg_83"](central_frequency)
    >>> round(bandwidth3dB, 2)
    113.55

  """
  tnt = 2 * n - 2
  return (factorial(n - 1) ** 2 / (pi * factorial(tnt) * 2 ** -tnt),
          2 * (2 ** (1. / n) - 1) ** .5
         )


@gammatone.strategy("sampled")
def gammatone(freq, bandwidth, phase=0, eta=4):
  """
  Gammatone filter based on a sampled impulse response.

    ``n ** (eta - 1) * exp(-bandwidth * n) * cos(freq * n + phase)``

  Parameters
  ----------
  freq :
    Frequency, in rad/sample.
  bandwidth :
    Frequency range size, in rad/sample. See gammatone_erb_constants for
    more information about how you can find this.
  phase :
    Phase, in radians. Defaults to zero (cosine).
  eta :
    Gammatone filter order. Defaults to 4.

  Returns
  -------
  A CascadeFilter object with ZFilter filters, each of them a pole-conjugated
  IIR filter model.
  Gain is normalized to have peak with 0 dB (1.0 amplitude).
  The total number of poles is twice the value of eta (conjugated pairs), one
  pair for each ZFilter.

  """
  assert eta >= 1

  A = exp(-bandwidth)
  numerator = cos(phase) - A * cos(freq - phase) * z ** -1
  denominator = 1 - 2 * A * cos(freq) * z ** -1 + A ** 2 * z ** -2
  filt = (numerator / denominator).diff(n=eta-1, mul_after=-z)

  # Filter is done, but the denominator might have some numeric loss
  f0 = ZFilter(filt.numpoly) / denominator
  f0 /= abs(f0.freq_response(freq)) # Max gain == 1.0 (0 dB)
  fn = 1 / denominator
  fn /= abs(fn.freq_response(freq))
  return CascadeFilter([f0] + [fn] * (eta - 1))


@gammatone.strategy("slaney")
def gammatone(freq, bandwidth):
  """
  Gammatone filter based on Malcolm Slaney's IIR cascading filter model
  described in:

    ``Slaney, M. "An Efficient Implementation of the Patterson-Holdsworth
    Auditory Filter Bank", Apple Computer Technical Report #35, 1993.``

  Parameters
  ----------
  freq :
    Frequency, in rad/sample.
  bandwidth :
    Frequency range size, in rad/sample. See gammatone_erb_constants for
    more information about how you can find this.

  Returns
  -------
  A CascadeFilter object with ZFilter filters, each of them a pole-conjugated
  IIR filter model.
  Gain is normalized to have peak with 0 dB (1.0 amplitude).
  The total number of poles is twice the value of eta (conjugated pairs), one
  pair for each ZFilter.

  """
  A = exp(-bandwidth)
  cosw = cos(freq)
  sinw = sin(freq)
  sig = [1., -1.]
  coeff = [cosw + s1 * (sqrt(2) + s2) * sinw for s1 in sig for s2 in sig]
  numerator = [1 - A * c * z ** -1 for c in coeff]
  denominator = 1 - 2 * A * cosw * z ** -1 + A ** 2 * z ** -2

  filt = CascadeFilter(num / denominator for num in numerator)
  return CascadeFilter(f / abs(f.freq_response(freq)) for f in filt)


@gammatone.strategy("klapuri")
def gammatone(freq, bandwidth):
  """
  Gammatone filter based on Anssi Klapuri's IIR cascading filter model
  described in:

    ``A. Klapuri, "Multipich Analysis of Polyphonic Music and Speech Signals
    Using an Auditory Model". IEEE Transactions on Audio, Speech and Language
    Processing, vol. 16, no. 2, 2008, pp. 255-266.``

  Parameters
  ----------
  freq :
    Frequency, in rad/sample.
  bandwidth :
    Frequency range size, in rad/sample. See gammatone_erb_constants for
    more information about how you can find this.

  Returns
  -------
  A CascadeFilter object with ZFilter filters, each of them a pole-conjugated
  IIR filter model.
  Gain is normalized to have peak with 0 dB (1.0 amplitude).
  The total number of poles is twice the value of eta (conjugated pairs), one
  pair for each ZFilter.

  """
  bw = thub(bandwidth, 1)
  bw2 = thub(bw * 2, 4)
  freq = thub(freq, 4)
  resons = [resonator.z_exp, resonator.poles_exp] * 2
  return CascadeFilter(reson(freq, bw2) for reson in resons)
