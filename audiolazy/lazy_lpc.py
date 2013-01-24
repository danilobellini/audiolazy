# -*- coding: utf-8 -*-
"""
Linear Prediction Coding (LPC) module

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

Created on Wed Jul 18 2012
danilo [dot] bellini [at] gmail [dot] com
"""

from __future__ import division
from itertools import izip
import operator

# Audiolazy internal imports
from .lazy_stream import Stream
from .lazy_filters import ZFilter, z
from .lazy_math import phase
from .lazy_core import StrategyDict
from .lazy_misc import blocks

__all__ = ["lpc", "ParCorError", "acorr", "levinson_durbin", "parcor",
           "parcor_stable", "lsf", "lsf_stable"]


lpc = StrategyDict("lpc")

class ParCorError(ZeroDivisionError):
  """
  Error when trying to find the partial correlation coefficients
  (reflection coefficients) and there's no way to find them.
  """

def acorr(blk, max_lag=None):
  """
  Calculate the autocorrelation of a given block.

  Parameters
  ----------
  blk :
    An iterable with well-defined length. Don't use this function with Stream
    objects!
  max_lag :
    The size of the result, the lags you'd need. Defaults to ``len(blk) - 1``,
    since any lag beyond would be zero.

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


def toeplitz(vect):
  """
  Find the toeplitz matrix as a list of lists given its first line/column.
  """
  return [[vect[abs(i-j)] for i in xrange(len(vect))]
                          for j in xrange(len(vect))]


def levinson_durbin(acdata, order=None):
  """
  Solve the Yule-Walker linear system of equations:

    ``R * a = r``

  where ``R`` is a simmetric Toeplitz matrix where each element are lags from
  the given autocorrelation list. ``R`` and ``r`` are defined:

    ``R[i][j] = acdata[abs(j - i)]``
    ``r = acdata[1 : order + 1]``

  Parameters
  ----------
  acdata :
    Autocorrelation lag list, commonly the ``acorr`` function output.
  order :
    The order of the resulting ZFilter object. Defaults to
    ``len(acdata) - 1``.

  Returns
  -------
  A FIR filter, as a ZFilter object. The mean squared error over the given
  data (variance of the white noise) is in its "error" attribute.

  See Also
  --------
  acorr:
    Calculate the autocorrelation of a given block.
  lpc :
    Calculate the Linear Prediction Coding (LPC) coefficients.
  parcor :
    Partial correlation coefficients (PARCOR), or reflection coefficients,
    relative to the lattice implementation of a filter, obtained by reversing
    the Levinson-Durbin algorithm.

  Examples
  --------
  >>> data = [2, 2, 0, 0, -1, -1, 0, 0, 1, 1]
  >>> acdata = acorr(data)
  >>> acdata
  [12, 6, 0, -3, -6, -3, 0, 2, 4, 2]
  >>> ldfilt = levinson_durbin(acorr(data), 3)
  >>> ldfilt
  1 - 0.625 * z^-1 + 0.25 * z^-2 + 0.125 * z^-3
  >>> ldfilt.error # Squared! See lpc for more information about this
  7.875

  Notes
  -----
  The Levinson-Durbin algorithm used to solve the equations uses
  ``O(order ** 2)`` floating point operations.

  """
  if order is None:
    order = len(acdata) - 1
  elif order >= len(acdata):
    acdata = Stream(acdata).append(0).take(order + 1)

  # Inner product for filters based on above statistics
  def inner(a, b): # Be careful, this depends on acdata !!!
    return sum(acdata[abs(i-j)] * ai * bj
               for i, ai in enumerate(a.numlist)
               for j, bj in enumerate(b.numlist)
              )

  A = ZFilter(1)
  try:
    for m in range(1, order + 1):
      B = ZFilter(A.numerator[::-1]) * z ** -1
      A -= inner(A, z ** -m) / inner(B, B) * B
  except ZeroDivisionError:
    raise ParCorError("Can't find next PARCOR coefficient")

  A.error = inner(A, A)
  return A


@lpc.strategy("autocor", "acorr", "autocorrelation", "auto_correlation")
def lpc(blk, order=None):
  """
  Find the Linear Prediction Coding (LPC) coefficients as a ZFilter object,
  the analysis whitening filter. This implementation uses the autocorrelation
  method, using the Levinson-Durbin algorithm or Numpy linear system solver,
  when needed.

  Parameters
  ----------
  blk :
    An iterable with well-defined length. Don't use this function with Stream
    objects!
  order :
    The order of the resulting ZFilter object. Defaults to ``len(blk) - 1``.

  Returns
  -------
  A FIR filter, as a ZFilter object. The mean squared error over the given
  block is in its "error" attribute.

  Examples
  --------
  >>> data = [-1, 0, 1, 0] * 4
  >>> len(data) # Small data
  16
  >>> filt = lpc.autocor(data, 2)
  >>> print filt # The analysis filter
  1 + 0.875 * z^-1
  >>> print filt.numerator # List of coefficients
  [1, 0.875]
  >>> print filt.error # Prediction error (squared!)
  14.125

  """
  if order < 100:
    return lpc.nautocor(blk, order)
  try:
    return lpc.kautocor(blk, order)
  except ParCorError:
    return lpc.nautocor(blk, order)


@lpc.strategy("nautocor", "nacorr", "nautocorrelation", "nauto_correlation")
def lpc(blk, order=None):
  """
  Find the Linear Prediction Coding (LPC) coefficients as a ZFilter object,
  the analysis whitening filter. This implementation uses the autocorrelation
  method, using Numpy linear system solver.

  Parameters
  ----------
  blk :
    An iterable with well-defined length. Don't use this function with Stream
    objects!
  order :
    The order of the resulting ZFilter object. Defaults to ``len(blk) - 1``.

  Returns
  -------
  A FIR filter, as a ZFilter object. The mean squared error over the given
  block is in its "error" attribute.

  Examples
  --------
  >>> data = [-1, 0, 1, 0] * 4
  >>> len(data) # Small data
  16
  >>> filt = lpc["nautocor"](data, 2)
  >>> print filt # The analysis filter
  1 + 0.875 * z^-1
  >>> print filt.numerator # List of coefficients
  [1, 0.875]
  >>> print filt.error # Prediction error (squared!)
  14.125

  """
  from numpy import array
  from numpy.linalg import solve
  acdata = acorr(blk, order)
  psi = array(acdata[1:])
  coeffs = solve(toeplitz(acdata[:-1]), -psi)
  filt = 1  + sum(ai * z ** -i for i, ai in enumerate(coeffs, 1))
  filt.error = acdata[0] + sum(psi * coeffs)
  return filt


@lpc.strategy("kautocor", "kacorr", "kautocorrelation", "kauto_correlation")
def lpc(blk, order=None):
  """
  Find the Linear Prediction Coding (LPC) coefficients as a ZFilter object,
  the analysis whitening filter. This implementation uses the autocorrelation
  method, using the Levinson-Durbin algorithm.

  Parameters
  ----------
  blk :
    An iterable with well-defined length. Don't use this function with Stream
    objects!
  order :
    The order of the resulting ZFilter object. Defaults to ``len(blk) - 1``.

  Returns
  -------
  A FIR filter, as a ZFilter object. The mean squared error over the given
  block is in its "error" attribute.

  Examples
  --------
  >>> data = [-1, 0, 1, 0] * 4
  >>> len(data) # Small data
  16
  >>> filt = lpc.kautocor(data, 2)
  >>> print filt # The analysis filter
  1 + 0.875 * z^-1
  >>> print filt.numerator # List of coefficients
  [1, 0.875]
  >>> print filt.error # Prediction error (squared!)
  14.125

  See Also
  --------
  levinson_durbin :
    Levinson-Durbin algorithm for solving Yule-Walker equations (Toeplitz
    matrix linear system).

  """
  return levinson_durbin(acorr(blk, order), order)


@lpc.strategy("covar", "cov", "covariance", "ncovar", "ncov", "ncovariance")
def lpc(blk, order=None):
  """
  Find the Linear Prediction Coding (LPC) coefficients as a ZFilter object,
  the analysis whitening filter. This implementation uses the covariance
  method, assuming a zero-mean stochastic process, using Numpy linear system
  solver.

  """
  from numpy import array
  from numpy.linalg import solve

  # Calculate the covariance for each lag pair
  if order is None:
    order = len(blk) - 1
  elif order >= len(blk):
    raise ValueError("Block length should be higher than order")
  phi = [[sum(blk[n - c1] * blk[n - c2] for n in xrange(order, len(blk))
             ) for c1 in xrange(order+1)
         ] for c2 in xrange(order+1)]
  fi = [[phi[i][j] for i in xrange(1, order + 1)]
                   for j in xrange(1, order + 1)]
  psi = array([phi[i][0] for i in xrange(1, order + 1)])
  coeffs = solve(fi, -psi)
  filt = 1  + sum(ai * z ** -i for i, ai in enumerate(coeffs, 1))
  filt.error = max(0., phi[0][0] + sum(psi * coeffs))
  return filt


@lpc.strategy("kcovar", "kcov", "kcovariance")
def lpc(blk, order=None):
  """
  Find the Linear Prediction Coding (LPC) coefficients as a ZFilter object,
  the analysis whitening filter. This implementation uses the covariance
  method, assuming a zero-mean stochastic process, by finding the reflection
  coefficients iteratively.

  """

  # Calculate the covariance for each lag pair
  if order is None:
    order = len(blk) - 1
  elif order >= len(blk):
    raise ValueError("Block length should be higher than order")
  phi = [[sum(blk[n - c1] * blk[n - c2] for n in xrange(order, len(blk))
             ) for c1 in xrange(order+1)
         ] for c2 in xrange(order+1)]

  # Inner product for filters based on above statistics
  def inner(a, b):
    return sum(phi[i][j] * ai * bj
               for i, ai in enumerate(a.numlist)
               for j, bj in enumerate(b.numlist)
              )

  A = ZFilter(1)
  eps = inner(A, A)
  B = [z ** -1]
  beta = [inner(B[0], B[0])]

  m = 1
  while True:
    try:
      k = -inner(A, z ** -m) / beta[m - 1]
    except ZeroDivisionError:
      raise ParCorError("Can't find next PARCOR coefficient")
    A += k * B[m - 1]
    eps -= k ** 2 * beta[m - 1]

    if m >= order:
      A.error = eps
      return A

    gamma = [inner(z ** -(m + 1), B[q]) / beta[q] for q in range(m)]
    B.append(z ** -(m + 1) - sum(gamma[q] * B[q] for q in range(m)))
    beta.append(inner(B[m], B[m]))
    m += 1


def parcor(fir_filt):
  """
  Find the partial correlation coefficients (PARCOR), or reflection
  coefficients, relative to the lattice implementation of a given LTI FIR
  LinearFilter with a constant denominator (i.e., without feedback).

  Parameters
  ----------
  fir_filt :
    A ZFilter object, causal, LTI and with a constant denominator.

  Returns
  -------
  A generator that results in each partial correlatino coefficient from
  iterative decomposition, reversing the Levinson-Durbin algorithm.

  Examples
  --------
  >>> filt = levinson_durbin([1, 2, 3, 4, 5, 3, 2, 1])
  >>> filt
  1 - 0.275 * z^-1 - 0.275 * z^-2 - 0.4125 * z^-3 + 1.5 * z^-4 - 0.9125 * z^-5
  - 0.275 * z^-6 - 0.275 * z^-7
  >>> filt.error
  1.9125
  >>> k_generator = parcor(filt)
  >>> k_generator
  <generator object parcor at ...>
  >>> [round(k, 7) for k in k_generator]
  [-0.275, -0.3793103, -1.4166667, -0.2, -0.25, -0.3333333, -2.0]

  See Also
  --------
  levinson_durbin :
    Levinson-Durbin algorithm for solving Yule-Walker equations (Toeplitz
    matrix linear system).

  """
  den = fir_filt.denominator
  if len(den) != 1:
    raise ValueError("Filter has feedback")
  elif den[0] != 1: # So we don't have to worry with the denominator anymore
    fir_filt /= den[0]

  while fir_filt.numpoly != 1:
    k = fir_filt.numerator[-1]
    zB = ZFilter(fir_filt.numerator[::-1])
    try:
      fir_filt = (fir_filt - k * zB) / (1 - k ** 2)
    except ZeroDivisionError:
      raise ParCorError("Can't find next PARCOR coefficient")
    fir_filt = (fir_filt - fir_filt.numpoly[0]) + 1 # Avoid rounding errors
    yield k


def parcor_stable(filt):
  """
  Tests whether the given filter is stable or not by using the partial
  correlation coefficients (reflection coefficients) of the given filter.

  Parameters
  ----------
  filt :
    A LTI filter as a LinearFilter object.

  Returns
  -------
  A boolean that is true only when all correlation coefficients are inside the
  unit circle. Critical stability (i.e., when outer coefficient has magnitude
  equals to one) is seem as an instability, and returns False.

  See Also
  --------
  parcor :
    Partial correlation coefficients generator.
  lsf_stable :
    Tests filter stability with Line Spectral Frequencies (LSF) values.

  """
  return all(abs(k) < 1 for k in parcor(ZFilter(filt.denpoly)))


def lsf(fir_filt):
  """
  Find the Line Spectral Frequencies (LSF) from a given FIR filter.

  Parameters
  ----------
  filt :
    A LTI FIR filter as a LinearFilter object.

  Returns
  -------
  A tuple with all LSFs in rad/sample, alternating from the forward prediction
  (even indexes) and backward prediction filters (odd indexes).

  """
  den = fir_filt.denominator
  if len(den) != 1:
    raise ValueError("Filter has feedback")
  elif den[0] != 1: # So we don't have to worry with the denominator anymore
    fir_filt /= den[0]

  from numpy import roots
  rev_filt = ZFilter(fir_filt.numerator[::-1]) * z ** -1
  P = fir_filt + rev_filt
  Q = fir_filt - rev_filt
  roots_p = roots(P.numerator[::-1])
  roots_q = roots(Q.numerator[::-1])
  lsf_p = sorted(phase(roots_p))
  lsf_q = sorted(phase(roots_q))
  return reduce(operator.concat, izip(lsf_p, lsf_q))


def lsf_stable(filt):
  """
  Tests whether the given filter is stable or not by using the Line Spectral
  Frequencies (LSF) of the given filter.

  Parameters
  ----------
  filt :
    A LTI filter as a LinearFilter object.

  Returns
  -------
  A boolean that is true only when the LSF values from forward and backward
  prediction filters alternates. Critical stability (both forward and backward
  filters has the same LSF value) is seem as an instability, and returns
  False.

  See Also
  --------
  lsf :
    Gets the Line Spectral Frequencies from a filter.
  parcor_stable :
    Tests filter stability with partial correlation coefficients (reflection
    coefficients).

  """
  return all(a < b for a, b in blocks(lsf(ZFilter(filt.denpoly)),
                                      size=2, hop=1))
