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
# Created on Sun Jul 29 2012
# danilo [dot] bellini [at] gmail [dot] com
"""
Audio analysis and block processing module
"""

from math import cos, pi
from collections import deque

# Audiolazy internal imports
from .lazy_core import StrategyDict
from .lazy_stream import tostream, thub, Stream
from .lazy_math import cexp, abs as lzabs
from .lazy_filters import lowpass, z
from .lazy_compat import xrange

__all__ = ["window", "acorr", "lag_matrix", "dft", "zcross", "envelope",
           "maverage", "clip", "unwrap", "freq_to_lag", "lag_to_freq", "amdf"]


window = StrategyDict("window")


@window.strategy("hamming")
def window(size):
  """
  Hamming window function with the given size.

  Returns
  -------
  List with the desired window samples. Max value is one (1.0).

  """

  if size == 1:
    return [1.0]
  return [.54 - .46 * cos(2 * pi * n / (size - 1))
          for n in xrange(size)]


@window.strategy("rectangular", "rect")
def window(size):
  """
  Rectangular window function with the given size.

  Returns
  -------
  List with the desired window samples. All values are ones (1.0).

  """
  return [1.0 for n in xrange(size)]


@window.strategy("bartlett")
def window(size):
  """
  Bartlett (triangular with zero-valued endpoints) window function with the
  given size.

  Returns
  -------
  List with the desired window samples. Max value is one (1.0).

  See Also
  --------
  window.triangular :
    Triangular with no zero end-point.

  """
  if size == 1:
    return [1.0]
  return [1 - 2.0 / (size - 1) * abs(n - (size - 1) / 2.0)
          for n in xrange(size)]


@window.strategy("triangular", "triangle")
def window(size):
  """
  Triangular (with no zero end-point) window function with the given size.

  Returns
  -------
  List with the desired window samples. Max value is one (1.0).

  See Also
  --------
  window.bartlett :
    Bartlett window, triangular with zero-valued end-points.

  """
  if size == 1:
    return [1.0]
  return [1 - 2.0 / (size + 1) * abs(n - (size - 1) / 2.0)
          for n in xrange(size)]


@window.strategy("hann", "hanning")
def window(size):
  """
  Hann window function with the given size.

  Returns
  -------
  List with the desired window samples. Max value is one (1.0).

  """
  if size == 1:
    return [1.0]
  return [.5 * (1 - cos(2 * pi * n / (size - 1))) for n in xrange(size)]


@window.strategy("blackman")
def window(size, alpha=.16):
  """
  Blackman window function with the given size.

  Parameters
  ----------
  size :
    Window size in samples.
  alpha :
    Blackman window alpha value. Defaults to 0.16.

  Returns
  -------
  List with the desired window samples. Max value is one (1.0).

  """
  if size == 1:
    return [1.0]
  return [alpha / 2 * cos(4 * pi * n / (size - 1))
          -.5 * cos(2 * pi * n / (size - 1)) + (1 - alpha) / 2
          for n in xrange(size)]


def acorr(blk, max_lag=None):
  """
  Calculate the autocorrelation of a given 1-D block sequence.

  Parameters
  ----------
  blk :
    An iterable with well-defined length. Don't use this function with Stream
    objects!
  max_lag :
    The size of the result, the lags you'd need. Defaults to ``len(blk) - 1``,
    since any lag beyond would result in zero.

  Returns
  -------
  A list with lags from 0 up to max_lag, where its ``i``-th element has the
  autocorrelation for a lag equals to ``i``. Be careful with negative lags!
  You should use abs(lag) indexes when working with them.

  Examples
  --------
  >>> seq = [1, 2, 3, 4, 3, 4, 2]
  >>> acorr(seq) # Default max_lag is len(seq) - 1
  [59, 52, 42, 30, 17, 8, 2]
  >>> acorr(seq, 9) # Zeros at the end
  [59, 52, 42, 30, 17, 8, 2, 0, 0, 0]
  >>> len(acorr(seq, 3)) # Resulting length is max_lag + 1
  4
  >>> acorr(seq, 3)
  [59, 52, 42, 30]

  """
  if max_lag is None:
    max_lag = len(blk) - 1
  return [sum(blk[n] * blk[n + tau] for n in xrange(len(blk) - tau))
          for tau in xrange(max_lag + 1)]


def lag_matrix(blk, max_lag=None):
  """
  Finds the lag matrix for a given 1-D block sequence.

  Parameters
  ----------
  blk :
    An iterable with well-defined length. Don't use this function with Stream
    objects!
  max_lag :
    The size of the result, the lags you'd need. Defaults to ``len(blk) - 1``,
    the maximum lag that doesn't create fully zeroed matrices.

  Returns
  -------
  The covariance matrix as a list of lists. Each cell (i, j) contains the sum
  of ``blk[n - i] * blk[n - j]`` elements for all n that allows such without
  padding the given block.

  """
  if max_lag is None:
    max_lag = len(blk) - 1
  elif max_lag >= len(blk):
    raise ValueError("Block length should be higher than order")

  return [[sum(blk[n - i] * blk[n - j] for n in xrange(max_lag, len(blk))
              ) for i in xrange(max_lag + 1)
          ] for j in xrange(max_lag + 1)]


def dft(blk, freqs, normalize=True):
  """
  Complex non-optimized Discrete Fourier Transform

  Finds the DFT for values in a given frequency list, in order, over the data
  block seen as periodic.

  Parameters
  ----------
  blk :
    An iterable with well-defined length. Don't use this function with Stream
    objects!
  freqs :
    List of frequencies to find the DFT, in rad/sample. FFT implementations
    like numpy.fft.ftt finds the coefficients for N frequencies equally
    spaced as ``line(N, 0, 2 * pi, finish=False)`` for N frequencies.
  normalize :
    If True (default), the coefficient sums are divided by ``len(blk)``,
    and the coefficient for the DC level (frequency equals to zero) is the
    mean of the block. If False, that coefficient would be the sum of the
    data in the block.

  Returns
  -------
  A list of DFT values for each frequency, in the same order that they appear
  in the freqs input.

  Note
  ----
  This isn't a FFT implementation, and performs :math:`O(M . N)` float
  pointing operations, with :math:`M` and :math:`N` equals to the length of
  the inputs. This function can find the DFT for any specific frequency, with
  no need for zero padding or finding all frequencies in a linearly spaced
  band grid with N frequency bins at once.

  """
  dft_data = (sum(xn * cexp(-1j * n * f) for n, xn in enumerate(blk))
                                         for f in freqs)
  if normalize:
    lblk = len(blk)
    return [v / lblk for v in dft_data]
  return list(dft_data)


@tostream
def zcross(seq, hysteresis=0, first_sign=0):
  """
  Zero-crossing stream.

  Parameters
  ----------
  seq :
    Any iterable to be used as input for the zero crossing analysis
  hysteresis :
    Crossing exactly zero might happen many times too fast due to high
    frequency oscilations near zero. To avoid this, you can make two
    threshold limits for the zero crossing detection: ``hysteresis`` and
    ``-hysteresis``. Defaults to zero (0), which means no hysteresis and only
    one threshold.
  first_sign :
    Optional argument with the sign memory from past. Gets the sig from any
    signed number. Defaults to zero (0), which means "any", and the first sign
    will be the first one found in data.

  Returns
  -------
  A Stream instance that outputs 1 for each crossing detected, 0 otherwise.

  """
  neg_hyst = -hysteresis
  seq_iter = iter(seq)

  # Gets the first sign
  if first_sign == 0:
    last_sign = 0
    for el in seq_iter:
      yield 0
      if (el > hysteresis) or (el < neg_hyst): # Ignores hysteresis region
        last_sign = -1 if el < 0 else 1 # Define the first sign
        break
  else:
    last_sign = -1 if first_sign < 0 else 1

  # Finds the full zero-crossing sequence
  for el in seq_iter: # Keep the same iterator (needed for non-generators)
    if el * last_sign < neg_hyst:
      last_sign = -1 if el < 0 else 1
      yield 1
    else:
      yield 0


envelope = StrategyDict("envelope")


@envelope.strategy("rms")
def envelope(sig, cutoff=pi/512):
  """
  Envelope non-linear filter.

  This strategy finds a RMS by passing the squared data through a low pass
  filter and taking its square root afterwards.

  Parameters
  ----------
  sig :
    The signal to be filtered.
  cutoff :
    Lowpass filter cutoff frequency, in rad/sample. Defaults to ``pi/512``.

  Returns
  -------
  A Stream instance with the envelope, without any decimation.

  See Also
  --------
  maverage :
    Moving average linear filter.

  """
  return lowpass(cutoff)(thub(sig, 1) ** 2) ** .5


@envelope.strategy("abs")
def envelope(sig, cutoff=pi/512):
  """
  Envelope non-linear filter.

  This strategy make an ideal half wave rectification (get the absolute value
  of each signal) and pass the resulting data through a low pass filter.

  Parameters
  ----------
  sig :
    The signal to be filtered.
  cutoff :
    Lowpass filter cutoff frequency, in rad/sample. Defaults to ``pi/512``.

  Returns
  -------
  A Stream instance with the envelope, without any decimation.

  See Also
  --------
  maverage :
    Moving average linear filter.

  """
  return lowpass(cutoff)(lzabs(thub(sig, 1)))


@envelope.strategy("squared")
def envelope(sig, cutoff=pi/512):
  """
  Squared envelope non-linear filter.

  This strategy squares the input, and apply a low pass filter afterwards.

  Parameters
  ----------
  sig :
    The signal to be filtered.
  cutoff :
    Lowpass filter cutoff frequency, in rad/sample. Defaults to ``pi/512``.

  Returns
  -------
  A Stream instance with the envelope, without any decimation.

  See Also
  --------
  maverage :
    Moving average linear filter.

  """
  return lowpass(cutoff)(thub(sig, 1) ** 2)


maverage = StrategyDict("maverage")


@maverage.strategy("deque")
def maverage(size):
  """
  Moving average

  This is the only strategy that uses a ``collections.deque`` object
  instead of a ZFilter instance. Fast, but without extra capabilites such
  as a frequency response plotting method.

  Parameters
  ----------
  size :
    Data block window size. Should be an integer.

  Returns
  -------
  A callable that accepts two parameters: a signal ``sig`` and the starting
  memory element ``zero`` that behaves like the ``LinearFilter.__call__``
  arguments. The output from that callable is a Stream instance, and has
  no decimation applied.

  See Also
  --------
  envelope :
    Signal envelope (time domain) strategies.

  """
  size_inv = 1. / size

  @tostream
  def maverage_filter(sig, zero=0.):
    data = deque((zero * size_inv for _ in xrange(size)), maxlen=size)
    mean_value = zero
    for el in sig:
      mean_value -= data.popleft()
      new_value = el * size_inv
      data.append(new_value)
      mean_value += new_value
      yield mean_value

  return maverage_filter


@maverage.strategy("recursive", "feedback")
def maverage(size):
  """
  Moving average

  Linear filter implementation as a recursive / feedback ZFilter.

  Parameters
  ----------
  size :
    Data block window size. Should be an integer.

  Returns
  -------
  A ZFilter instance with the feedback filter.

  See Also
  --------
  envelope :
    Signal envelope (time domain) strategies.

  """
  return (1. / size) * (1 - z ** -size) / (1 - z ** -1)


@maverage.strategy("fir")
def maverage(size):
  """
  Moving average

  Linear filter implementation as a FIR ZFilter.

  Parameters
  ----------
  size :
    Data block window size. Should be an integer.

  Returns
  -------
  A ZFilter instance with the FIR filter.

  See Also
  --------
  envelope :
    Signal envelope (time domain) strategies.

  """
  return sum((1. / size) * z ** -i for i in xrange(size))


def clip(sig, low=-1., high=1.):
  """
  Clips the signal up to both a lower and a higher limit.

  Parameters
  ----------
  sig :
    The signal to be clipped, be it a Stream instance, a list or any iterable.
  low, high :
    Lower and higher clipping limit, "saturating" the input to them. Defaults
    to -1.0 and 1.0, respectively. These can be None when needed one-sided
    clipping. When both limits are set to None, the output will be a Stream
    that yields exactly the ``sig`` input data.

  Returns
  -------
  Clipped signal as a Stream instance.

  """
  if low is None:
    if high is None:
      return Stream(sig)
    return Stream(el if el < high else high for el in sig)
  if high is None:
    return Stream(el if el > low else low for el in sig)
  if high < low:
    raise ValueError("Higher clipping limit is smaller than lower one")
  return Stream(high if el > high else
                (low if el < low else el) for el in sig)


@tostream
def unwrap(sig, max_delta=pi, step=2*pi):
  """
  Parametrized signal unwrapping.

  Parameters
  ----------
  sig :
    An iterable seen as an input signal.
  max_delta :
    Maximum value of :math:`\Delta = sig_i - sig_{i-1}` to keep output
    without another minimizing step change. Defaults to :math:`\pi`.
  step :
    The change in order to minimize the delta is an integer multiple of this
    value. Defaults to :math:`2 . \pi`.

  Returns
  -------
  The signal unwrapped as a Stream, minimizing the step difference when any
  adjacency step in the input signal is higher than ``max_delta`` by
  summing/subtracting ``step``.

  """
  idata = iter(sig)
  d0 = next(idata)
  yield d0
  delta = d0 - d0 # Get the zero (e.g., integer, float) from data
  for d1 in idata:
    d_diff = d1 - d0
    if abs(d_diff) > max_delta:
      delta += - d_diff + min((d_diff) % step,
                              (d_diff) % -step, key=lambda x: abs(x))
    yield d1 + delta
    d0 = d1


def freq_to_lag(x):
  """
  Converts between frequency (rad/sample) and lag (number of samples).

  """
  return 2 * pi / x

lag_to_freq = freq_to_lag


def amdf(lag, size):
  """
  Average Magnitude Difference Function non-linear filter for a given
  size and a fixed lag.

  Parameters
  ----------
  lag :
    Time lag, in samples. See ``freq_to_lag`` if needs conversion from
    frequency values.
  size :
    Moving average size.

  Returns
  -------
  A callable that accepts two parameters: a signal ``sig`` and the starting
  memory element ``zero`` that behaves like the ``LinearFilter.__call__``
  arguments. The output from that callable is a Stream instance, and has
  no decimation applied.

  See Also
  --------
  freq_to_lag :
    Frequency to lag and lag to frequency converter.

  """
  filt = (1 - z ** -lag).linearize()

  @tostream
  def amdf_filter(sig, zero=0.):
    return maverage(size)(lzabs(filt(sig, zero=zero)), zero=zero)

  return amdf_filter
