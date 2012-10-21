#!/usr/bin/env python
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

from math import pi, exp, cos, sin, sqrt
import operator

# Audiolazy internal imports
from .lazy_core import StrategyDict
from .lazy_misc import elementwise, factorial
from .lazy_filters import z


erb = StrategyDict("erb")
gammatone = StrategyDict("gammatone")


@erb.strategy("gm90", "glasberg_moore_90", "glasberg_moore")
@elementwise("freq", 0)
def erb(freq, Hz=1):
  """
  ERB model from:

    ``B. R. Glasberg and B. C. J. Moore, "Derivation of auditory filter
    shapes from notched-noise data". Hearing Research, vol. 47, 1990, pp.
    103-108.``

  Parameters
  ----------
  freq :
    Frequency, in rad/s if second parameter is given, in Hz otherwise.
  Hz :
    Frequency conversion "Hz" from sHz function, i.e., ``sHz(rate)[1]``.
    If this value is not given, both input and output will be in Hz.

  Returns
  -------
  Frequency range size, in rad/s if second parameter is given, in Hz otherwise.
  """
  fHz = freq / Hz
  result = 24.7 * (4.37e-3 * fHz + 1.)
  return result * Hz


@erb.strategy("mg83", "moore_glasberg_83")
@elementwise("freq", 0)
def erb(freq, Hz=1):
  """
  ERB model from:

    ``B. C. J. Moore and B. R. Glasberg, "Suggested formulae for calculating
    auditory filter bandwidths and excitation patterns". J. Acoust. Soc.
    Am., 74, 1983, pp. 750-753.``

  Parameters
  ----------
  freq :
    Frequency, in rad/s if second parameter is given, in Hz otherwise.
  Hz :
    Frequency conversion "Hz" from sHz function, i.e., ``sHz(rate)[1]``.
    If this value is not given, both input and output will be in Hz.

  Returns
  -------
  Frequency range size, in rad/s if second parameter is given, in Hz otherwise.
  """
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

    ``t ** (eta - 1) * exp(-bandwidth) * cos(freq * t + phase)``

  Parameters
  ----------
  freq :
    Frequency, in rad/s.
  bandwidth :
    Frequency range size, in rad/s. See gammatone_erb_constants for
    more information about how you can find this.
  phase :
    Phase, in radians. Defaults to zero (cosine).
  eta :
    Gammatone filter order. Defaults to 4.

  Returns
  -------
  A LTIFreq filter object, that can be seem as an IIR filter model.
  Gain is normalized to have peak with 0 dB (1.0 amplitude).
  The number of poles is twice the value of eta (conjugated pairs).
  """
  A = exp(-bandwidth)
  numerator = cos(phase) - A * cos(freq - phase) * z ** -1
  denominator = 1 - 2 * A * cos(freq) * z ** -1 + A ** 2 * z ** -2
  filt = (numerator / denominator).diff(n=eta-1, mul_after=-z)
  return filt / abs(filt.freq_response(freq)) # Max gain == 1.0 (0 dB)


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
    Frequency, in rad/s.
  bandwidth :
    Frequency range size, in rad/s. See gammatone_erb_constants for
    more information about how you can find this.

  Returns
  -------
  A LTIFreq filter object, that can be seem as an IIR filter model.
  Gain is normalized to have peak with 0 dB (1.0 amplitude).
  The number of poles is twice the value of eta (conjugated pairs).
  """
  A = exp(-bandwidth)
  cosw = cos(freq)
  sinw = sin(freq)
  sig = [1., -1.]
  coeff = [cosw + s1 * (sqrt(2) + s2) * sinw for s1 in sig for s2 in sig]
  numerator = [1 - A * c * z ** -1 for c in coeff]
  denominator = 1 - 2 * A * cosw * z ** -1 + A ** 2 * z ** -2
  filt = reduce(operator.mul, (num / denominator for num in numerator))
  return filt / abs(filt.freq_response(freq)) # Max gain == 1.0 (0 dB)


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
    Frequency, in rad/s.
  bandwidth :
    Frequency range size, in rad/s. See gammatone_erb_constants for
    more information about how you can find this.

  Returns
  -------
  A LTIFreq filter object, that can be seem as an IIR filter model.
  Gain is normalized to have peak with 0 dB (1.0 amplitude).
  The number of poles is twice the value of eta (conjugated pairs).
  """
  A = exp(-bandwidth)
  cosw = cos(freq)

  Asqr = A ** 2
  costheta1 = cosw * (1 + Asqr) / (2 * A)
  costheta2 = cosw * (2 * A) / (1 + Asqr)
  ro1 = .5 * (1 - Asqr)
  ro2 = (1 - Asqr) * (1 - costheta2 ** 2) ** .5

  H1 = ro1 * (1 - z ** -2) / (1 - 2 * A * costheta1 * z ** -1
                                + Asqr * z ** -2)
  H2 = ro2                 / (1 - 2 * A * costheta2 * z ** -1
                                + Asqr * z ** -2)

  return (H1 * H2) ** 2
