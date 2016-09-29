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
Peripheral auditory modeling module
"""

import math

# Audiolazy internal imports
from .lazy_core import StrategyDict
from .lazy_misc import elementwise
from .lazy_filters import z, CascadeFilter, ZFilter, resonator
from .lazy_math import pi, exp, cos, sin, sqrt, factorial
from .lazy_stream import thub
from .lazy_compat import xzip
from .lazy_text import format_docstring

__all__ = ["erb", "gammatone", "gammatone_erb_constants", "phon2dB"]


erb = StrategyDict("erb")
erb._doc_template = """
  Equivalent Rectangular Model (ERB) from {authors} ({year}).

  This is a model for a single filter bandwidth for auditory filter modeling,
  taken from:
  {__doc__}
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

@erb.strategy("gm90", "glasberg_moore_90", "glasberg_moore")
@elementwise("freq", 0)
@format_docstring(erb._doc_template, authors="Glasberg and Moore", year=1990)
def erb(freq, Hz=None):
  """
    ``B. R. Glasberg and B. C. J. Moore, "Derivation of auditory filter
    shapes from notched-noise data". Hearing Research, vol. 47, 1990, pp.
    103-108.``
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
@format_docstring(erb._doc_template, authors="Moore and Glasberg", year=1983)
def erb(freq, Hz=None):
  """
    ``B. C. J. Moore and B. R. Glasberg, "Suggested formulae for calculating
    auditory filter bandwidths and excitation patterns". J. Acoust. Soc.
    Am., 74, 1983, pp. 750-753.``
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
  order. Returns a pair :math:`(x, y) = (1/a_n, c_n)`.

  Based on equations from:

    ``Holdsworth, J.; Patterson, R.; Nimmo-Smith, I.; Rice, P. Implementing a
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


gammatone = StrategyDict("gammatone")
gammatone._doc_template = """
  Gammatone filter based on {model}.

  Model is described in:
  {__doc__}
  Parameters
  ----------
  freq :
    Frequency, in rad/sample.
  bandwidth :
    Frequency range size, in rad/sample. See ``gammatone_erb_constants`` for
    more information about how you can find this.
  {extra_params}
  Returns
  -------
  A CascadeFilter object with ZFilter filters, each of them a pole-conjugated
  IIR filter model. Gain is normalized to have peak with 0 dB (1.0 amplitude).
  The total number of poles is twice the value of eta (conjugated pairs), one
  pair for each ZFilter.
"""


@gammatone.strategy("sampled")
@format_docstring(gammatone._doc_template, model="a sampled impulse response",
  extra_params="\n  ".join([
    "phase :", "  Phase, in radians. Defaults to zero (cosine)."
    "eta :", "  Gammatone filter order. Defaults to 4.", "" # Skip a line
  ]),
)
def gammatone(freq, bandwidth, phase=0, eta=4):
  """
    ``Bellini, D. J. S. "AudioLazy: Processamento digital de sinais
    expressivo e em tempo real", IME-USP, Mastership Thesis, 2013.``

  This implementation have the impulse response (for each sample ``n``,
  keeping the input parameter names):

  .. math::

    n^{{eta - 1}} e^{{- bandwidth \cdot n}} \cos(freq \cdot n + phase)
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
@format_docstring(gammatone._doc_template, extra_params="",
                  model="Malcolm Slaney's IIR cascading filter model")
def gammatone(freq, bandwidth):
  """
    ``Slaney, M. "An Efficient Implementation of the Patterson-Holdsworth
    Auditory Filter Bank", Apple Computer Technical Report #35, 1993.``
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
@format_docstring(gammatone._doc_template, extra_params="",
                  model="Anssi Klapuri's IIR cascading filter model")
def gammatone(freq, bandwidth):
  """
    ``A. Klapuri, "Multipich Analysis of Polyphonic Music and Speech Signals
    Using an Auditory Model". IEEE Transactions on Audio, Speech and Language
    Processing, vol. 16, no. 2, 2008, pp. 255-266.``
  """
  bw = thub(bandwidth, 1)
  bw2 = thub(bw * 2, 4)
  freq = thub(freq, 4)
  resons = [resonator.z_exp, resonator.poles_exp] * 2
  return CascadeFilter(reson(freq, bw2) for reson in resons)


phon2dB = StrategyDict("phon2dB")


@phon2dB.strategy("iso226", "iso226_2003", "iso_fdis_226_2003")
def phon2dB(loudness=None):
  """
  Loudness in phons to Sound Pressure Level (SPL) in dB using the
  ISO/FDIS 226:2003 model.

  This function needs Scipy, as ``scipy.interpolate.UnivariateSpline``
  objects are used as interpolators.

  Parameters
  ----------
  loudness :
    The loudness value in phons to be converted, or None (default) to get
    the threshold of hearing.

  Returns
  -------
  A callable that returns the SPL dB value for each given frequency in hertz.

  Note
  ----
  See ``phon2dB.iso226.schema`` and ``phon2dB.iso226.table`` to know the
  original frequency used for the result. The result for any other value is
  an interpolation (spline). Don't trust on values nor lower nor higher than
  the frequency limits there (20Hz and 12.5kHz) as they're not part of
  ISO226 and no value was collected to estimate them (they're just a spline
  interpolation to reach 1000dB at -30Hz and 32kHz). Likewise, the trustful
  loudness input range is from 20 to 90 phon, as written on ISO226, although
  other values aren't found by a spline interpolation but by using the
  formula on section 4.1 of ISO226.

  Hint
  ----
  The ``phon2dB.iso226.table`` also have other useful information, such as
  the threshold values in SPL dB.

  """
  from scipy.interpolate import UnivariateSpline

  table = phon2dB.iso226.table
  schema = phon2dB.iso226.schema
  freqs = [row[schema.index("freq")] for row in table]

  if loudness is None: # Threshold levels
    spl = [row[schema.index("threshold")] for row in table]

  else: # Curve for a specific phon value
    def get_pressure_level(freq, alpha, loudness_base, threshold):
      return 10 / alpha * math.log10(
        4.47e-3 * (10 ** (.025 * loudness) - 1.14) +
        (.4 * 10 ** ((threshold + loudness_base) / 10 - 9)) ** alpha
      ) - loudness_base + 94

    spl = [get_pressure_level(**dict(xzip(schema, args))) for args in table]

  interpolator = UnivariateSpline(freqs, spl, s=0)
  interpolator_low = UnivariateSpline([-30] + freqs, [1e3] + spl, s=0)
  interpolator_high = UnivariateSpline(freqs + [32000], spl + [1e3], s=0)

  @elementwise("freq", 0)
  def freq2dB_spl(freq):
    if freq < 20:
      return interpolator_low(freq).tolist()
    if freq > 12500:
      return interpolator_high(freq).tolist()
    return interpolator(freq).tolist()
  return freq2dB_spl

# ISO226 Table 1
phon2dB.iso226.schema = ("freq", "alpha", "loudness_base", "threshold")
phon2dB.iso226.table = (
  (   20, 0.532, -31.6, 78.5),
  (   25, 0.506, -27.2, 68.7),
  ( 31.5, 0.480, -23.0, 59.5),
  (   40, 0.455, -19.1, 51.1),
  (   50, 0.432, -15.9, 44.0),
  (   63, 0.409, -13.0, 37.5),
  (   80, 0.387, -10.3, 31.5),
  (  100, 0.367,  -8.1, 26.5),
  (  125, 0.349,  -6.2, 22.1),
  (  160, 0.330,  -4.5, 17.9),
  (  200, 0.315,  -3.1, 14.4),
  (  250, 0.301,  -2.0, 11.4),
  (  315, 0.288,  -1.1,  8.6),
  (  400, 0.276,  -0.4,  6.2),
  (  500, 0.267,   0.0,  4.4),
  (  630, 0.259,   0.3,  3.0),
  (  800, 0.253,   0.5,  2.2),
  ( 1000, 0.250,   0.0,  2.4),
  ( 1250, 0.246,  -2.7,  3.5),
  ( 1600, 0.244,  -4.1,  1.7),
  ( 2000, 0.243,  -1.0, -1.3),
  ( 2500, 0.243,   1.7, -4.2),
  ( 3150, 0.243,   2.5, -6.0),
  ( 4000, 0.242,   1.2, -5.4),
  ( 5000, 0.242,  -2.1, -1.5),
  ( 6300, 0.245,  -7.1,  6.0),
  ( 8000, 0.254, -11.2, 12.6),
  (10000, 0.271, -10.7, 13.9),
  (12500, 0.301,  -3.1, 12.3),
)
