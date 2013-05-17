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
# Created on Wed Jul 18 2012
# danilo [dot] bellini [at] gmail [dot] com
"""
Linear Predictive Coding (LPC) module
"""

from __future__ import division
from functools import reduce
import operator

# Audiolazy internal imports
from .lazy_stream import Stream
from .lazy_filters import ZFilter, z
from .lazy_math import phase
from .lazy_core import StrategyDict
from .lazy_misc import blocks
from .lazy_compat import xrange, xzip
from .lazy_analysis import acorr, lag_matrix

__all__ = ["ParCorError", "toeplitz", "levinson_durbin", "lpc", "parcor",
           "parcor_stable", "lsf", "lsf_stable"]


class ParCorError(ZeroDivisionError):
  """
  Error when trying to find the partial correlation coefficients
  (reflection coefficients) and there's no way to find them.
  """


def toeplitz(vect):
  """
  Find the toeplitz matrix as a list of lists given its first line/column.
  """
  return [[vect[abs(i-j)] for i in xrange(len(vect))]
                          for j in xrange(len(vect))]


def levinson_durbin(acdata, order=None):
  """
  Solve the Yule-Walker linear system of equations.

  They're given by:

  .. math::

    R . a = r

  where :math:`R` is a simmetric Toeplitz matrix where each element are lags
  from the given autocorrelation list. :math:`R` and :math:`r` are defined
  (Python indexing starts with zero and slices don't include the last
  element):

  .. math::

    R[i][j] = acdata[abs(j - i)]

    r = acdata[1 : order + 1]

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
    Calculate the Linear Predictive Coding (LPC) coefficients.
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
  The Levinson-Durbin algorithm used to solve the equations needs
  :math:`O(order^2)` floating point operations.

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

  try:
    A = ZFilter(1)
    for m in xrange(1, order + 1):
      B = A(1 / z) * z ** -m
      A -= inner(A, z ** -m) / inner(B, B) * B
  except ZeroDivisionError:
    raise ParCorError("Can't find next PARCOR coefficient")

  A.error = inner(A, A)
  return A


lpc = StrategyDict("lpc")


@lpc.strategy("autocor", "acorr", "autocorrelation", "auto_correlation")
def lpc(blk, order=None):
  """
  Find the Linear Predictive Coding (LPC) coefficients as a ZFilter object,
  the analysis whitening filter. This implementation uses the autocorrelation
  method, using the Levinson-Durbin algorithm or Numpy pseudo-inverse for
  linear system solving, when needed.

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

  Hint
  ----
  See ``lpc.kautocor`` example, which should apply equally for this strategy.

  See Also
  --------
  levinson_durbin :
    Levinson-Durbin algorithm for solving Yule-Walker equations (Toeplitz
    matrix linear system).
  lpc.nautocor:
    LPC coefficients from linear system solved with Numpy pseudo-inverse.
  lpc.kautocor:
    LPC coefficients obtained with Levinson-Durbin algorithm.

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
  Find the Linear Predictive Coding (LPC) coefficients as a ZFilter object,
  the analysis whitening filter. This implementation uses the autocorrelation
  method, using numpy.linalg.pinv as a linear system solver.

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

  Hint
  ----
  See ``lpc.kautocor`` example, which should apply equally for this strategy.

  See Also
  --------
  lpc.autocor:
    LPC coefficients by using one of the autocorrelation method strategies.
  lpc.kautocor:
    LPC coefficients obtained with Levinson-Durbin algorithm.

  """
  from numpy import matrix
  from numpy.linalg import pinv
  acdata = acorr(blk, order)
  coeffs = pinv(toeplitz(acdata[:-1])) * -matrix(acdata[1:]).T
  coeffs = coeffs.T.tolist()[0]
  filt = 1  + sum(ai * z ** -i for i, ai in enumerate(coeffs, 1))
  filt.error = acdata[0] + sum(a * c for a, c in xzip(acdata[1:], coeffs))
  return filt


@lpc.strategy("kautocor", "kacorr", "kautocorrelation", "kauto_correlation")
def lpc(blk, order=None):
  """
  Find the Linear Predictive Coding (LPC) coefficients as a ZFilter object,
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
  >>> filt # The analysis filter
  1 + 0.875 * z^-2
  >>> filt.numerator # List of coefficients
  [1, 0.0, 0.875]
  >>> filt.error # Prediction error (squared!)
  1.875

  See Also
  --------
  levinson_durbin :
    Levinson-Durbin algorithm for solving Yule-Walker equations (Toeplitz
    matrix linear system).
  lpc.autocor:
    LPC coefficients by using one of the autocorrelation method strategies.
  lpc.nautocor:
    LPC coefficients from linear system solved with Numpy pseudo-inverse.

  """
  return levinson_durbin(acorr(blk, order), order)


@lpc.strategy("covar", "cov", "covariance", "ncovar", "ncov", "ncovariance")
def lpc(blk, order=None):
  """
  Find the Linear Predictive Coding (LPC) coefficients as a ZFilter object,
  the analysis whitening filter. This implementation uses the covariance
  method, assuming a zero-mean stochastic process, using numpy.linalg.pinv
  as a linear system solver.

  """
  from numpy import matrix
  from numpy.linalg import pinv

  lagm = lag_matrix(blk, order)
  phi = matrix(lagm)
  psi = phi[1:, 0]
  coeffs = pinv(phi[1:, 1:]) * -psi
  coeffs = coeffs.T.tolist()[0]
  filt = 1  + sum(ai * z ** -i for i, ai in enumerate(coeffs, 1))
  filt.error = phi[0, 0] + sum(a * c for a, c in xzip(lagm[0][1:], coeffs))
  return filt


@lpc.strategy("kcovar", "kcov", "kcovariance")
def lpc(blk, order=None):
  """
  Find the Linear Predictive Coding (LPC) coefficients as a ZFilter object,
  the analysis whitening filter. This implementation is based on the
  covariance method, assuming a zero-mean stochastic process, finding
  the coefficients iteratively and greedily like the lattice implementation
  in Levinson-Durbin algorithm, although the lag matrix found from the given
  block don't have to be toeplitz. Slow, but this strategy don't need NumPy.

  """
  # Calculate the covariance for each lag pair
  phi = lag_matrix(blk, order)
  order = len(phi) - 1

  # Inner product for filters based on above statistics
  def inner(a, b):
    return sum(phi[i][j] * ai * bj
               for i, ai in enumerate(a.numlist)
               for j, bj in enumerate(b.numlist)
              )

  A = ZFilter(1)
  B = [z ** -1]
  beta = [inner(B[0], B[0])]

  m = 1
  while True:
    try:
      k = -inner(A, z ** -m) / beta[m - 1] # Last one is really a PARCOR coeff
    except ZeroDivisionError:
      raise ZeroDivisionError("Can't find next coefficient")
    if k >= 1 or k <= -1:
      raise ValueError("Unstable filter")
    A += k * B[m - 1]

    if m >= order:
      A.error = inner(A, A)
      return A

    gamma = [inner(z ** -(m + 1), B[q]) / beta[q] for q in xrange(m)]
    B.append(z ** -(m + 1) - sum(gamma[q] * B[q] for q in xrange(m)))
    beta.append(inner(B[m], B[m]))
    m += 1


def parcor(fir_filt):
  """
  Find the partial correlation coefficients (PARCOR), or reflection
  coefficients, relative to the lattice implementation of a given LTI FIR
  LinearFilter with a constant denominator (i.e., LPC analysis filter, or
  any filter without feedback).

  Parameters
  ----------
  fir_filt :
    A ZFilter object, causal, LTI and with a constant denominator.

  Returns
  -------
  A generator that results in each partial correlation coefficient from
  iterative decomposition, reversing the Levinson-Durbin algorithm.

  Examples
  --------
  >>> filt = levinson_durbin([1, 2, 3, 4, 5, 3, 2, 1])
  >>> filt
  1 - 0.275 * z^-1 - 0.275 * z^-2 - 0.4125 * z^-3 + 1.5 * z^-4 """\
  """- 0.9125 * z^-5 - 0.275 * z^-6 - 0.275 * z^-7
  >>> round(filt.error, 4)
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

  for m in xrange(len(fir_filt.numerator) - 1, 0, -1):
    k = fir_filt.numpoly[m]
    yield k
    zB = fir_filt(1 / z) * z ** -m
    try:
      fir_filt = (fir_filt - k * zB) / (1 - k ** 2)
    except ZeroDivisionError:
      raise ParCorError("Can't find next PARCOR coefficient")
    fir_filt = (fir_filt - fir_filt.numpoly[0]) + 1 # Avoid rounding errors


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
  try:
    return all(abs(k) < 1 for k in parcor(ZFilter(filt.denpoly)))
  except ParCorError:
    return False


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
  and backward prediction filters, starting with the lowest LSF value.

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
  return reduce(operator.concat, xzip(*sorted([lsf_p, lsf_q])), tuple())


def lsf_stable(filt):
  """
  Tests whether the given filter is stable or not by using the Line Spectral
  Frequencies (LSF) of the given filter. Needs NumPy.

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
    Gets the Line Spectral Frequencies from a filter. Needs NumPy.
  parcor_stable :
    Tests filter stability with partial correlation coefficients (reflection
    coefficients).

  """
  lsf_data = lsf(ZFilter(filt.denpoly))
  return all(a < b for a, b in blocks(lsf_data, size=2, hop=1))
