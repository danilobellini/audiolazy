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
Stream filtering module
"""

from __future__ import division

import operator
from cmath import exp as complex_exp
from collections import Iterable, OrderedDict
import itertools as it
from functools import reduce

# Audiolazy internal imports
from .lazy_stream import Stream, avoid_stream, thub
from .lazy_misc import elementwise, zero_pad, sHz, almost_eq
from .lazy_text import (float_str, multiplication_formatter,
                        pair_strings_sum_formatter)
from .lazy_compat import meta, iteritems, xrange, im_func
from .lazy_poly import Poly
from .lazy_core import AbstractOperatorOverloaderMeta, StrategyDict
from .lazy_math import (exp, sin, cos, sqrt, pi, nan, dB20, phase,
                        abs as lzabs, e, inf)

__all__ = ["LinearFilterProperties", "LinearFilter", "ZFilterMeta", "ZFilter",
           "z", "FilterListMeta", "FilterList", "CascadeFilter",
           "ParallelFilter", "comb", "resonator", "lowpass", "highpass"]


class LinearFilterProperties(object):
  """
  Class with common properties in a linear filter that can be used as a mixin.

  The classes that inherits this one should implement the ``numpoly`` and
  ``denpoly`` properties, and these should return a Poly instance.

  """
  def numlist(self):
    if any(key < 0 for key, value in self.numpoly.terms()):
      raise ValueError("Non-causal filter")
    return list(self.numpoly.values())
  numerator = property(numlist)
  numlist = property(numlist)

  def denlist(self):
    if any(key < 0 for key, value in self.denpoly.terms()):
      raise ValueError("Non-causal filter")
    return list(self.denpoly.values())
  denominator = property(denlist)
  denlist = property(denlist)

  @property
  def numdict(self):
    return OrderedDict(self.numpoly.terms())

  @property
  def dendict(self):
    return OrderedDict(self.denpoly.terms())

  @property
  def numpolyz(self):
    """
    Like numpoly, the linear filter numerator (or forward coefficients) as a
    Poly instance based on ``x = z`` instead of numpoly's ``x = z ** -1``,
    useful for taking roots.

    """
    return Poly(self.numerator[::-1])

  @property
  def denpolyz(self):
    """
    Like denpoly, the linear filter denominator (or backward coefficients) as
    a Poly instance based on ``x = z`` instead of denpoly's ``x = z ** -1``,
    useful for taking roots.

    """
    return Poly(self.denominator[::-1])


def _exec_eval(data, expr):
  """
  Internal function to isolate an exec. Executes ``data`` and returns the
  ``expr`` evaluation afterwards.

  """
  ns = {}
  exec(data, ns)
  return eval(expr, ns)


@avoid_stream
class LinearFilter(LinearFilterProperties):
  """
  Base class for Linear filters, time invariant or not.
  """
  def __init__(self, numerator=None, denominator=None):
    if isinstance(numerator, LinearFilter):
      # Filter type cast
      if denominator is not None:
        numerator = operator.truediv(numerator, denominator)
      self.numpoly = numerator.numpoly
      self.denpoly = numerator.denpoly
    else:
      # Filter from coefficients
      self.numpoly = Poly(numerator)
      self.denpoly = Poly({0: 1} if denominator is None else denominator)

    # Ensure denominator has only negative powers of z (positive powers here),
    # and a not null gain constant
    power = min(key for key, value in self.denpoly.terms())
    if power != 0:
      poly_delta = Poly([0, 1]) ** -power
      self.numpoly *= poly_delta
      self.denpoly *= poly_delta

  def __iter__(self):
    yield self.numdict
    yield self.dendict

  def __hash__(self):
    return hash(tuple(self.numdict) + tuple(self.dendict))

  def __call__(self, seq, memory=None, zero=0.):
    """
    IIR, FIR and time variant linear filtering.

    Parameters
    ----------
    seq :
      Any iterable to be seem as the input stream for the filter.
    memory :
      Might be an iterable or a callable. Generally, as a iterable, the first
      needed elements from this input will be used directly as the memory
      (not the last ones!), and as a callable, it will be called with the
      size as the only positional argument, and should return an iterable.
      If ``None`` (default), memory is initialized with zeros.
    zero :
      Value to fill the memory, when needed, and to be seem as previous
      input when there's a delay. Defaults to ``0.0``.

    Returns
    -------
    A Stream that have the data from the input sequence filtered.

    """
    # Data check
    if any(key < 0 for key, value in it.chain(self.numpoly.terms(),
                                              self.denpoly.terms())
          ):
      raise ValueError("Non-causal filter")
    if isinstance(self.denpoly[0], Stream): # Variable output gain
      den = self.denpoly
      inv_gain = 1 / den[0]
      den[0] = 0
      den *= inv_gain.copy()
      den[0] = 1
      return ZFilter(self.numpoly * inv_gain, den)(seq, memory=memory,
                                                   zero=zero)
    if self.denpoly[0] == 0:
      raise ZeroDivisionError("Invalid filter gain")

    # Lengths
    la, lb = len(self.denominator), len(self.numerator)
    lm = la - 1 # Memory size

    # Convert memory input to a list with size exactly equals to lm
    if memory is None:
      memory = [zero for unused in xrange(lm)]
    else: # Get data from iterable
      if not isinstance(memory, Iterable): # Function with 1 parameter: size
        memory = memory(lm)
      tw = it.takewhile(lambda pair: pair[0] < lm,
                        enumerate(memory))
      memory = [data for idx, data in tw]
      actual_len = len(memory)
      if actual_len < lm:
        memory = list(zero_pad(memory, lm - actual_len, zero=zero))

    # Creates the expression in a string
    data_sum = []

    num_iterables = []
    for delay, coeff in iteritems(self.numdict):
      if isinstance(coeff, Iterable):
        num_iterables.append(delay)
        data_sum.append("d{idx} * next(b{idx})".format(idx=delay))
      elif coeff == 1:
        data_sum.append("d{idx}".format(idx=delay))
      elif coeff == -1:
        data_sum.append("-d{idx}".format(idx=delay))
      elif coeff != 0:
        data_sum.append("d{idx} * {value}".format(idx=delay, value=coeff))

    den_iterables = []
    for delay, coeff in iteritems(self.dendict):
      if isinstance(coeff, Iterable):
        den_iterables.append(delay)
        data_sum.append("-m{idx} * next(a{idx})".format(idx=delay))
      elif delay == 0:
        gain = coeff
      elif coeff == -1:
        data_sum.append("m{idx}".format(idx=delay))
      elif coeff == 1:
        data_sum.append("-m{idx}".format(idx=delay))
      elif coeff != 0:
        data_sum.append("-m{idx} * {value}".format(idx=delay, value=coeff))

    # Creates the generator function for this call
    if len(data_sum) == 0:
      gen_func =  ["def gen(seq):",
                   "  for unused in seq:",
                   "    yield {zero}".format(zero=zero)
                  ]
    else:
      expr = " + ".join(data_sum)
      if gain == -1:
        expr = "-({expr})".format(expr=expr)
      elif gain != 1:
        expr = "({expr}) / {gain}".format(expr=expr, gain=gain)

      arg_names = ["seq"]
      arg_names.extend("b{idx}".format(idx=idx) for idx in num_iterables)
      arg_names.extend("a{idx}".format(idx=idx) for idx in den_iterables)
      gen_func =  ["def gen({args}):".format(args=", ".join(arg_names))]
      gen_func += ["  m{idx} = {value}".format(idx=idx, value=value)
                   for idx, value in enumerate(memory, 1)]
      gen_func += ["  d{idx} = {value}".format(idx=idx, value=zero)
                   for idx in xrange(1, lb)]
      gen_func += ["  for d0 in seq:",
                   "    m0 = {expr}".format(expr=expr),
                   "    yield m0"]
      gen_func += ["    m{idx} = m{idxold}".format(idx=idx, idxold=idx - 1)
                   for idx in xrange(lm, 0, -1)]
      gen_func += ["    d{idx} = d{idxold}".format(idx=idx, idxold=idx - 1)
                   for idx in xrange(lb - 1, 0, -1)]

    # Uses the generator function to return the desired values
    gen = _exec_eval("\n".join(gen_func), "gen")
    arguments = [iter(seq)]
    arguments.extend(iter(self.numpoly[idx]) for idx in num_iterables)
    arguments.extend(iter(self.denpoly[idx]) for idx in den_iterables)
    return Stream(gen(*arguments))


  @elementwise("freq", 1)
  def freq_response(self, freq):
    """
    Frequency response for this filter.

    Parameters
    ----------
    freq :
      Frequency, in rad/sample. Can be an iterable with frequencies.

    Returns
    -------
    Complex number with the frequency response of the filter.

    See Also
    --------
    dB10 :
      Logarithmic power magnitude from data with squared magnitude.
    dB20 :
      Logarithmic power magnitude from raw complex data or data with linear
      amplitude.
    phase :
      Phase from complex data.

    """
    z_ = complex_exp(-1j * freq)
    num = self.numpoly(z_)
    den = self.denpoly(z_)
    if not isinstance(den, Stream):
      if den == 0:
        return nan
    return num / den

  def is_lti(self):
    """
    Test if this filter is LTI (Linear Time Invariant).

    Returns
    -------
    Boolean returning True if this filter is LTI, False otherwise.

    """
    return not any(isinstance(value, Iterable)
                   for delay, value in it.chain(self.numpoly.terms(),
                                                self.denpoly.terms()))

  def is_causal(self):
    """
    Causality test for this filter.

    Returns
    -------
    Boolean returning True if this filter is causal, False otherwise.

    """
    return all(delay >= 0 for delay, value in self.numpoly.terms())

  def copy(self):
    """
    Returns a filter copy.

    It'll return a LinearFilter instance (more specific class when
    subclassing) with the same terms in both numerator and denominator, but
    as a "T" (tee) copy when the coefficients are Stream instances, allowing
    maths using a filter more than once.

    """
    return type(self)(self.numpoly.copy(), self.denpoly.copy())

  def linearize(self):
    """
    Linear interpolation of fractional delay values.

    Returns
    -------
    A new linear filter, with the linearized delay values.

    Examples
    --------

    >>> filt = z ** -4.3
    >>> filt.linearize()
    0.7 * z^-4 + 0.3 * z^-5

    """
    data = []
    for poly in [self.numpoly, self.denpoly]:
      data.append({})
      new_poly = data[-1]
      for k, v in poly.terms():
        if isinstance(k, int) or (isinstance(k, float) and k.is_integer()):
          pairs = [(int(k), v)]
        else:
          left = int(k)
          right = left + 1
          weight_right = k - left
          weight_left = 1. - weight_right
          pairs = [(left, v * weight_left), (right, v * weight_right)]
        for key, value in pairs:
          if key in new_poly:
            new_poly[key] += value
          else:
            new_poly[key] = value
    return self.__class__(*data)

  def plot(self, fig=None, samples=2048, rate=None, min_freq=0., max_freq=pi,
           blk=None, unwrap=True, freq_scale="linear", mag_scale="dB"):
    """
    Plots the filter frequency response into a formatted MatPlotLib figure
    with two subplots, labels and title, including the magnitude response
    and the phase response.

    Parameters
    ----------
    fig :
      A matplotlib.figure.Figure instance. Defaults to None, which means that
      it will create a new figure.
    samples :
      Number of samples (frequency values) to plot. Defaults to 2048.
    rate :
      Given rate (samples/second) or "s" object from ``sHz``. Defaults to 300.
    min_freq, max_freq :
      Frequency range to be drawn, in rad/sample. Defaults to [0, pi].
    blk :
      Sequence block. Plots the block DFT together with the filter frequency.
      Defaults to None (no block).
    unwrap :
      Boolean that chooses whether should unwrap the data phase or keep it as
      is. Defaults to True.
    freq_scale :
      Chooses whether plot is "linear" or "log" with respect to the frequency
      axis. Defaults to "linear". Case insensitive.
    mag_scale :
      Chooses whether magnitude plot scale should be "linear", "squared" or
      "dB". Defaults do "dB". Case insensitive.

    Returns
    -------
    The matplotlib.figure.Figure instance.

    See Also
    --------
    sHz :
      Second and hertz constants from samples/second rate.
    LinearFilter.zplot :
      Zeros-poles diagram plotting.

    """
    if not self.is_lti():
      raise AttributeError("Filter is not time invariant (LTI)")
    fscale = freq_scale.lower()
    mscale = mag_scale.lower()
    mscale = "dB" if mag_scale == "db" else mag_scale
    if fscale not in ["linear", "log"]:
      raise ValueError("Unknown frequency scale")
    if mscale not in ["linear", "squared", "dB"]:
      raise ValueError("Unknown magnitude scale")

    from .lazy_synth import line
    from .lazy_analysis import dft, unwrap as unwrap_func
    from matplotlib import pyplot as plt

    if fig is None:
      fig = plt.figure()

    # Units! Bizarre "pi/12" just to help MaxNLocator, corrected by fmt_func
    Hz = pi / 12. if rate == None else sHz(rate)[1]
    funit = "rad/sample" if rate == None else "Hz"

    # Sample the frequency range linearly (data scale) and get the data
    freqs = list(line(samples, min_freq, max_freq, finish=True))
    freqs_label = list(line(samples, min_freq / Hz, max_freq / Hz,
                            finish=True))
    data = self.freq_response(freqs)
    if blk is not None:
      fft_data = dft(blk, freqs)

    # Plots the magnitude response
    mag_plot = fig.add_subplot(2, 1, 1)
    if fscale == "symlog":
      mag_plot.set_xscale(fscale, basex=2., basey=2.,
                          steps=[1., 1.25, 1.5, 1.75])
    else:
      mag_plot.set_xscale(fscale)
    mag_plot.set_title("Frequency response")
    mag = {"linear": lzabs,
           "squared": lambda x: [abs(xi) ** 2 for xi in x],
           "dB": dB20
          }[mscale]
    if blk is not None:
      mag_plot.plot(freqs_label, mag(fft_data))
    mag_plot.plot(freqs_label, mag(data))
    mag_plot.set_ylabel("Magnitude ({munit})".format(munit=mscale))
    mag_plot.grid(True)
    plt.setp(mag_plot.get_xticklabels(), visible = False)

    # Plots the phase response
    ph_plot = fig.add_subplot(2, 1, 2, sharex = mag_plot)
    ph = (lambda x: unwrap_func(phase(x))) if unwrap else phase
    if blk is not None:
      ph_plot.plot(freqs_label, [xi * 12 / pi for xi in ph(fft_data)])
    ph_plot.plot(freqs_label, [xi * 12 / pi for xi in ph(data)])
    ph_plot.set_ylabel("Phase (rad)")
    ph_plot.set_xlabel("Frequency ({funit})".format(funit=funit))
    ph_plot.set_xlim(freqs_label[0], freqs_label[-1])
    ph_plot.grid(True)

    # X Ticks (gets strange unit "7.5 * degrees / sample" back ) ...
    fmt_func = lambda value, pos: float_str(value * pi / 12., "p", [8])
    if rate is None:
      if fscale == "linear":
        loc = plt.MaxNLocator(steps=[1, 2, 3, 4, 6, 8, 10])
      elif fscale == "log":
        loc = plt.LogLocator(base=2.)
        loc_minor = plt.LogLocator(base=2., subs=[1.25, 1.5, 1.75])
        ph_plot.xaxis.set_minor_locator(loc_minor)
      ph_plot.xaxis.set_major_locator(loc)
      ph_plot.xaxis.set_major_formatter(plt.FuncFormatter(fmt_func))

    # ... and Y Ticks
    loc = plt.MaxNLocator(steps=[1, 2, 3, 4, 6, 8, 10])
    ph_plot.yaxis.set_major_locator(loc)
    ph_plot.yaxis.set_major_formatter(plt.FuncFormatter(fmt_func))

    mag_plot.yaxis.get_major_locator().set_params(prune="lower")
    ph_plot.yaxis.get_major_locator().set_params(prune="upper")
    fig.subplots_adjust(hspace=0.)
    return fig

  def zplot(self, fig=None, circle=True):
    """
    Plots the filter zero-pole plane into a formatted MatPlotLib figure
    with one subplot, labels and title.

    Parameters
    ----------
    fig :
      A matplotlib.figure.Figure instance. Defaults to None, which means that
      it will create a new figure.
    circle :
      Chooses whether to include the unit circle in the plot. Defaults to
      True.

    Returns
    -------
    The matplotlib.figure.Figure instance.

    Note
    ----
    Multiple roots detection is slow, and roots may suffer from numerical
    errors (e.g., filter ``f = 1 - 2 * z ** -1 + 1 * z ** -2`` has twice the
    root ``1``, but ``f ** 3`` suffer noise from the root finding algorithm).
    For the exact number of poles and zeros, see the result title, or the
    length of LinearFilter.poles() and LinearFilter.zeros().

    See Also
    --------
    LinearFilter.plot :
      Frequency response plotting. Needs MatPlotLib.
    LinearFilter.zeros, LinearFilter.poles :
      Filter zeros and poles, as a list. Needs NumPy.

    """
    if not self.is_lti():
      raise AttributeError("Filter is not time invariant (LTI)")

    from matplotlib import pyplot as plt
    from matplotlib import transforms

    if fig is None:
      fig = plt.figure()

    # Configure the plot matplotlib.axes.Axes artist and circle background
    zp_plot = fig.add_subplot(1, 1, 1)
    if circle:
      zp_plot.add_patch(plt.Circle((0., 0.), radius=1., fill=False,
                                   linewidth=1., color="gray",
                                   linestyle="dashed"))

    # Plot the poles and zeros
    zeros = self.zeros # Start with zeros to avoid overdrawn hidden poles
    for zero in zeros:
      zp_plot.plot(zero.real, zero.imag, "o", markersize=8.,
                   markeredgewidth=1.5, markerfacecolor="c",
                   markeredgecolor="b")
    poles = self.poles
    for pole in poles:
      zp_plot.plot(pole.real, pole.imag, "x", markersize=8.,
                   markeredgewidth=2.5, markerfacecolor="r",
                   markeredgecolor="r")

    # Configure the axis (top/right is translated by 1 internally in pyplot)
    # Note: older MPL versions (e.g. 1.0.1) still don't have the color
    # matplotlib.colors.cname["lightgray"], which is the same to "#D3D3D3"
    zp_plot.spines["top"].set_position(("data", -1.))
    zp_plot.spines["right"].set_position(("data", -1.))
    zp_plot.spines["top"].set_color("#D3D3D3")
    zp_plot.spines["right"].set_color("#D3D3D3")
    zp_plot.axis("scaled") # Keep aspect ratio

    # Configure the plot limits
    border_width = .1
    zp_plot.set_xlim(xmin=zp_plot.dataLim.xmin - border_width,
                     xmax=zp_plot.dataLim.xmax + border_width)
    zp_plot.set_ylim(ymin=zp_plot.dataLim.ymin - border_width,
                     ymax=zp_plot.dataLim.ymax + border_width)

    # Multiple roots (or slightly same roots) detection
    def get_repeats(pairs):
      """
      Find numbers that are almost equal, for the printing sake.
      Input: list of number pairs (tuples with size two)
      Output: dict of pairs {pair: amount_of_repeats}
      """
      result = {idx: {idx} for idx, pair in enumerate(pairs)}
      for idx1, idx2 in it.combinations(xrange(len(pairs)), 2):
        p1 = pairs[idx1]
        p2 = pairs[idx2]
        if almost_eq(p1, p2):
          result[idx1] = result[idx1].union(result[idx2])
          result[idx2] = result[idx1]
      to_verify = [pair for pair in pairs]
      while to_verify:
        idx = to_verify.pop()
        if idx in result:
          for idxeq in result[idx]:
            if idxeq != idx and idx in result:
              del result[idx]
              to_verify.remove(idx)
      return {pairs[k]: len(v) for k, v in iteritems(result) if len(v) > 1}

    # Multiple roots text printing
    td = zp_plot.transData
    tpole = transforms.offset_copy(td, x=7, y=6, units="dots")
    tzero = transforms.offset_copy(td, x=7, y=-6, units="dots")
    tdi = td.inverted()
    zero_pos = [tuple(td.transform((zero.real, zero.imag)))
                for zero in zeros]
    pole_pos = [tuple(td.transform((pole.real, pole.imag)))
                for pole in poles]
    for zero, zrep in iteritems(get_repeats(zero_pos)):
      px, py = tdi.transform(zero)
      txt = zp_plot.text(px, py, "{0:d}".format(zrep), color="darkgreen",
                         transform=tzero, ha="center", va="center",
                         fontsize=10)
      txt.set_bbox(dict(facecolor="white", edgecolor="None", alpha=.4))
    for pole, prep in iteritems(get_repeats(pole_pos)):
      px, py = tdi.transform(pole)
      txt = zp_plot.text(px, py, "{0:d}".format(prep), color="black",
                         transform=tpole, ha="center", va="center",
                         fontsize=10)
      txt.set_bbox(dict(facecolor="white", edgecolor="None", alpha=.4))

    # Labels, title and finish
    zp_plot.set_title("Zero-Pole plot ({0:d} zeros, {1:d} poles)"
                      .format(len(zeros), len(poles)))
    zp_plot.set_xlabel("Real part")
    zp_plot.set_ylabel("Imaginary part")
    return fig

  @property
  def poles(self):
    """
    Returns a list with all poles (denominator roots in ``z``). Needs Numpy.

    See Also
    --------
    LinearFilterProperties.numpoly:
      Numerator polynomials where *x* is ``z ** -1``.
    LinearFilterProperties.denpoly:
      Denominator polynomials where *x* is ``z ** -1``.
    LinearFilterProperties.numpolyz:
      Numerator polynomials where *x* is ``z``.
    LinearFilterProperties.denpolyz:
      Denominator polynomials where *x* is ``z``.

    """
    return self.denpolyz.roots

  @property
  def zeros(self):
    """
    Returns a list with all zeros (numerator roots in ``z``), besides the
    zero-valued "zeros" that might arise from the difference between the
    numerator and denominator order (i.e., the roots returned are the inverse
    from the ``numpoly.roots()`` in ``z ** -1``). Needs Numpy.

    See Also
    --------
    LinearFilterProperties.numpoly:
      Numerator polynomials where *x* is ``z ** -1``.
    LinearFilterProperties.denpoly:
      Denominator polynomials where *x* is ``z ** -1``.
    LinearFilterProperties.numpolyz:
      Numerator polynomials where *x* is ``z``.
    LinearFilterProperties.denpolyz:
      Denominator polynomials where *x* is ``z``.

    """
    return self.numpolyz.roots

  def __eq__(self, other):
    if isinstance(other, LinearFilter):
      return self.numpoly == other.numpoly and self.denpoly == other.denpoly
    return False

  def __ne__(self, other):
    if isinstance(other, LinearFilter):
      return self.numpoly != other.numpoly and self.denpoly != other.denpoly
    return False


class ZFilterMeta(AbstractOperatorOverloaderMeta):
  __operators__ = "+ - * / **"

  def __rbinary__(cls, op):
    op_func = op.func
    def dunder(self, other):
      if isinstance(other, cls):
        raise ValueError("Filter equations have different domains")
      return op_func(cls([other]), self) # The "other" is probably a number
    return dunder

  def __unary__(cls, op):
    op_func = op.func
    def dunder(self):
      return cls(op_func(self.numpoly), self.denpoly)
    return dunder


@avoid_stream
class ZFilter(meta(LinearFilter, metaclass=ZFilterMeta)):
  """
  Linear filters based on Z-transform frequency domain equations.

  Examples
  --------

  Using the ``z`` object (float output because default filter memory has
  float zeros, and the delay in the numerator creates another float zero as
  "pre-input"):

  >>> filt = (1 + z ** -1) / (1 - z ** -1)
  >>> data = [1, 5, -4, -7, 9]
  >>> stream_result = filt(data) # Lazy iterable
  >>> list(stream_result) # Freeze
  [1.0, 7.0, 8.0, -3.0, -1.0]

  Same example with the same filter, but with a memory input, and using
  lists for filter numerator and denominator instead of the ``z`` object:

  >>> b = [1, 1]
  >>> a = [1, -1] # Each index ``i`` has the coefficient for z ** -i
  >>> filt = ZFilter(b, a)
  >>> data = [1, 5, -4, -7, 9]
  >>> stream_result = filt(data, memory=[3], zero=0) # Lazy iterable
  >>> result = list(stream_result) # Freeze
  >>> result
  [4, 10, 11, 0, 2]
  >>> filt2 = filt * z ** -1 # You can add a delay afterwards easily
  >>> final_result = filt2(result, zero=0)
  >>> list(final_result)
  [0, 4, 18, 39, 50]

  """
  def __add__(self, other):
    if isinstance(other, ZFilter):
      if self.denpoly == other.denpoly:
        return ZFilter(self.numpoly + other.numpoly, self.denpoly)
      return ZFilter(self.numpoly * other.denpoly.copy() +
                     other.numpoly * self.denpoly.copy(),
                     self.denpoly * other.denpoly)
    if isinstance(other, LinearFilter):
      raise ValueError("Filter equations have different domains")
    return self + ZFilter([other]) # Other is probably a number

  def __sub__(self, other):
    return self + (-other)

  def __mul__(self, other):
    if isinstance(other, ZFilter):
      return ZFilter(self.numpoly * other.numpoly,
                     self.denpoly * other.denpoly)
    if isinstance(other, LinearFilter):
      raise ValueError("Filter equations have different domains")
    return ZFilter(self.numpoly * other, self.denpoly)

  def __truediv__(self, other):
    if isinstance(other, ZFilter):
      return ZFilter(self.numpoly * other.denpoly,
                     self.denpoly * other.numpoly)
    if isinstance(other, LinearFilter):
      raise ValueError("Filter equations have different domains")
    return self * operator.truediv(1, other)

  def __pow__(self, other):
    if (other < 0) and (len(self.numpoly) >= 2 or len(self.denpoly) >= 2):
      return ZFilter(self.denpoly, self.numpoly) ** -other
    if isinstance(other, (int, float)):
      return ZFilter(self.numpoly ** other, self.denpoly ** other)
    raise ValueError("Z-transform powers only valid with integers")

  def __str__(self):
    num_term_strings = []
    for power, value in self.numpoly.terms():
      if isinstance(value, Iterable):
        value = "b{}".format(power).replace(".", "_").replace("-", "m")
      if value != 0.:
        num_term_strings.append(multiplication_formatter(-power, value, "z"))
    num = "0" if len(num_term_strings) == 0 else \
          reduce(pair_strings_sum_formatter, num_term_strings)

    den_term_strings = []
    for power, value in self.denpoly.terms():
      if isinstance(value, Iterable):
        value = "a{}".format(power).replace(".", "_").replace("-", "m")
      if value != 0.:
        den_term_strings.append(multiplication_formatter(-power, value, "z"))
    den = reduce(pair_strings_sum_formatter, den_term_strings)

    if den == "1": # No feedback
      return num

    line = "-" * max(len(num), len(den))
    spacer_offset = abs(len(num) - len(den)) // 2
    if spacer_offset > 0:
      centralize_spacer = " " * spacer_offset
      if len(num) > len(den):
        den = centralize_spacer + den
      else: # len(den) > len(num)
        num = centralize_spacer + num

    breaks = len(line) // 80
    slices = [slice(b * 80,(b + 1) * 80) for b in xrange(breaks + 1)]
    outputs = ["\n".join([num[s], line[s], den[s]]) for s in slices]
    return "\n\n    ...continue...\n\n".join(outputs)

  __repr__ = __str__

  def diff(self, n=1, mul_after=1):
    """
    Takes n-th derivative, multiplying each m-th derivative filter by
    mul_after before taking next (m+1)-th derivative or returning.
    """
    if isinstance(mul_after, ZFilter):
      den = ZFilter(self.denpoly)
      return reduce(lambda num, order: mul_after *
                      (num.diff() * den - order * num * den.diff()),
                    xrange(1, n + 1),
                    ZFilter(self.numpoly)
                   ) / den ** (n + 1)

    inv_sign = Poly({-1: 1}) # Since poly variable is z ** -1
    den = self.denpoly(inv_sign)
    return ZFilter(reduce(lambda num, order: mul_after *
                            (num.diff() * den - order * num * den.diff()),
                          xrange(1, n + 1),
                          self.numpoly(inv_sign))(inv_sign),
                   self.denpoly ** (n + 1))

  def __call__(self, seq, memory=None, zero=0.):
    """
    IIR, FIR and time variant linear filtering.

    Parameters
    ----------
    seq :
      Any iterable to be seem as the input stream for the filter, or another
      ZFilter for substituition.
    memory :
      Might be an iterable or a callable. Generally, as a iterable, the first
      needed elements from this input will be used directly as the memory
      (not the last ones!), and as a callable, it will be called with the
      size as the only positional argument, and should return an iterable.
      If ``None`` (default), memory is initialized with zeros. Neglect when
      ``seq`` input is a ZFilter.
    zero :
      Value to fill the memory, when needed, and to be seem as previous
      input when there's a delay. Defaults to ``0.0``. Neglect when ``seq``
      input is a ZFilter.

    Returns
    -------
    A Stream that have the data from the input sequence filtered.

    Examples
    --------
    With ZFilter instances:

    >>> filt = 1 + z ** -1
    >>> filt(z ** -1)
    z + 1
    >>> filt(- z ** 2)
    1 - z^-2

    With any iterable (but ZFilter instances):

    >>> filt = 1 + z ** -1
    >>> data = filt([1.0, 2.0, 3.0])
    >>> data
    <audiolazy.lazy_stream.Stream object at ...>
    >>> list(data)
    [1.0, 3.0, 5.0]

    """
    if isinstance(seq, ZFilter):
      return sum(v * seq ** -k for k, v in self.numpoly.terms()) / \
             sum(v * seq ** -k for k, v in self.denpoly.terms())
    else:
      return super(ZFilter, self).__call__(seq, memory=memory, zero=zero)


z = ZFilter({-1: 1})


class FilterListMeta(AbstractOperatorOverloaderMeta):
  __operators__ = "add * > >= < <="

  def __binary__(cls, op):
    op_dname = op.dname
    def dunder(self, other):
      "This operator acts just like it would do with lists."
      return cls(getattr(super(cls, self), op_dname)(other))
    return dunder

  __rbinary__ = __binary__


class FilterList(meta(list, LinearFilterProperties, metaclass=FilterListMeta)):
  """
  Class from which CascadeFilter and ParallelFilter inherits the common part
  of their contents. You probably won't need to use this directly.

  """
  def __init__(self, *filters):
    if len(filters) == 1 and not callable(filters[0]) \
                         and isinstance(filters[0], Iterable):
      filters = filters[0]
    self.extend(filters)

  def is_linear(self):
    """
    Tests whether all filters in the list are linear. CascadeFilter and
    ParallelFilter instances are also linear if all filters they group are
    linear.

    """
    return all(isinstance(filt, LinearFilter) or
               (hasattr(filt, "is_linear") and filt.is_linear())
               for filt in self.callables)

  def is_lti(self):
    """
    Tests whether all filters in the list are linear time invariant (LTI).
    CascadeFilter and ParallelFilter instances are also LTI if all filters
    they group are LTI.

    """
    return self.is_linear() and all(filt.is_lti() for filt in self.callables)

  def is_causal(self):
    """
    Tests whether all filters in the list are causal (i.e., no future-data
    delay in positive ``z`` exponents). Non-linear filters are seem as causal
    by default. CascadeFilter and ParallelFilter are causal if all the
    filters they group are causal.

    """
    return all(filt.is_causal() for filt in self.callables
                                if hasattr(filt, "is_causal"))

  plot = im_func(LinearFilter.plot)
  zplot = im_func(LinearFilter.zplot)

  def __eq__(self, other):
    return type(self) == type(other) and list.__eq__(self, other)

  def __ne__(self, other):
    return type(self) != type(other) or list.__ne__(self, other)

  @property
  def callables(self):
    """
    List of callables with all filters, casting to LinearFilter each one that
    isn't callable.

    """
    return [(filt if callable(filt) else LinearFilter(filt)) for filt in self]


@avoid_stream
class CascadeFilter(FilterList):
  """
  Filter cascade as a list of filters.

  Note
  ----
  A filter is any callable that receives an iterable as input and returns a
  Stream.

  Examples
  --------
  >>> filt = CascadeFilter(z ** -1, 2 * (1 - z ** -3))
  >>> data = Stream(1, 3, 5, 3, 1, -1, -3, -5, -3, -1) # Endless
  >>> filt(data, zero=0).take(15)
  [0, 2, 6, 10, 4, -4, -12, -12, -12, -4, 4, 12, 12, 12, 4]

  """
  def __call__(self, *args, **kwargs):
    return reduce(lambda data, filt: filt(data, *args[1:], **kwargs),
                  self.callables, args[0])

  @property
  def numpoly(self):
    try:
      return reduce(operator.mul, (filt.numpoly for filt in self.callables))
    except AttributeError:
      raise AttributeError("Non-linear filter")

  @property
  def denpoly(self):
    try:
      return reduce(operator.mul, (filt.denpoly for filt in self.callables))
    except AttributeError:
      raise AttributeError("Non-linear filter")

  @elementwise("freq", 1)
  def freq_response(self, freq):
    return reduce(operator.mul, (filt.freq_response(freq)
                                 for filt in self.callables))

  @property
  def poles(self):
    if not self.is_lti():
      raise AttributeError("Not a LTI filter")
    return reduce(operator.concat, (filt.poles for filt in self.callables))

  @property
  def zeros(self):
    if not self.is_lti():
      raise AttributeError("Not a LTI filter")
    return reduce(operator.concat, (filt.zeros for filt in self.callables))


@avoid_stream
class ParallelFilter(FilterList):
  """
  Filters in parallel as a list of filters.

  This list of filters that behaves as a filter, returning the sum of all
  signals that results from applying the the same given input into all
  filters. Besides the name, the data processing done isn't parallel.

  Note
  ----
  A filter is any callable that receives an iterable as input and returns a
  Stream.

  Examples
  --------
  >>> filt = 1 + z ** -1 -  z ** -2
  >>> pfilt = ParallelFilter(1 + z ** -1, - z ** -2)
  >>> list(filt(range(100))) == list(pfilt(range(100)))
  True
  >>> list(filt(range(10), zero=0))
  [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]

  """
  def __call__(self, *args, **kwargs):
    if len(self) == 0:
      return Stream(kwargs["zero"] if "zero" in kwargs else 0.
                    for _ in args[0])
    arg0 = thub(args[0], len(self))
    return reduce(operator.add, (filt(arg0, *args[1:], **kwargs)
                                 for filt in self.callables))

  @property
  def numpoly(self):
    if not self.is_linear():
      raise AttributeError("Non-linear filter")
    return reduce(operator.add, self).numpoly

  @property
  def denpoly(self):
    try:
      return reduce(operator.mul, (filt.denpoly for filt in self.callables))
    except AttributeError:
      raise AttributeError("Non-linear filter")

  @elementwise("freq", 1)
  def freq_response(self, freq):
    return reduce(operator.add, (filt.freq_response(freq)
                                 for filt in self.callables))

  @property
  def poles(self):
    if not self.is_lti():
      raise AttributeError("Not a LTI filter")
    return reduce(operator.concat, (filt.poles for filt in self.callables))

  @property
  def zeros(self):
    if not self.is_lti():
      raise AttributeError("Not a LTI filter")
    return reduce(operator.add, (ZFilter(filt) for filt in self)).zeros


comb = StrategyDict("comb")


@comb.strategy("fb", "alpha", "fb_alpha", "feedback_alpha")
def comb(delay, alpha=1):
  """
  Feedback comb filter for a given alpha (and delay).

    ``y[n] = x[n] + alpha * y[n - delay]``

  Parameters
  ----------
  delay :
    Feedback delay, in number of samples.
  alpha :
    Exponential decay gain. You can find it from time decay ``tau`` in the
    impulse response, bringing us ``alpha = e ** (-delay / tau)``. See
    ``comb.tau`` strategy if that's the case. Defaults to 1 (no decay).

  Returns
  -------
  A ZFilter instance with the comb filter.

  """
  return 1 / (1 - alpha * z ** -delay)


@comb.strategy("tau", "fb_tau", "feedback_tau")
def comb(delay, tau=inf):
  """
  Feedback comb filter for a given time constant (and delay).

    ``y[n] = x[n] + alpha * y[n - delay]``

  Parameters
  ----------
  delay :
    Feedback delay, in number of samples.
  tau :
    Time decay (up to ``1/e``, or -8.686 dB), in number of samples, which
    allows finding ``alpha = e ** (-delay / tau)``. Defaults to ``inf``
    (infinite), which means alpha = 1.

  Returns
  -------
  A ZFilter instance with the comb filter.

  """
  alpha = e ** (-delay / tau)
  return 1 / (1 - alpha * z ** -delay)


@comb.strategy("ff", "ff_alpha", "feedforward_alpha")
def comb(delay, alpha=1):
  """
  Feedforward comb filter for a given alpha (and delay).

    ``y[n] = x[n] + alpha * x[n - delay]``

  Parameters
  ----------
  delay :
    Feedback delay, in number of samples.
  alpha :
    Memory value gain.

  Returns
  -------
  A ZFilter instance with the comb filter.

  """
  return 1 + alpha * z ** -delay


resonator = StrategyDict("resonator")


@resonator.strategy("poles_exp")
def resonator(freq, bandwidth):
  """
  Resonator filter with 2-poles (conjugated pair) and no zeros (constant
  numerator), with exponential approximation for bandwidth calculation.

  Parameters
  ----------
  freq :
    Resonant frequency in rad/sample (max gain).
  bandwidth :
    Bandwidth frequency range in rad/sample following the equation:

      ``R = exp(-bandwidth / 2)``

    where R is the pole amplitude (radius).

  Returns
  -------
  A ZFilter object.
  Gain is normalized to have peak with 0 dB (1.0 amplitude).

  """
  bandwidth = thub(bandwidth, 1)
  R = exp(-bandwidth * .5)
  R = thub(R, 5)
  cost = cos(freq) * (2 * R) / (1 + R ** 2)
  cost = thub(cost, 2)
  gain = (1 - R ** 2) * sqrt(1 - cost ** 2)
  denominator = 1 - 2 * R * cost * z ** -1 + R ** 2 * z ** -2
  return gain / denominator


@resonator.strategy("freq_poles_exp")
def resonator(freq, bandwidth):
  """
  Resonator filter with 2-poles (conjugated pair) and no zeros (constant
  numerator), with exponential approximation for bandwidth calculation.
  Given frequency is the denominator frequency, not the resonant frequency.

  Parameters
  ----------
  freq :
    Denominator frequency in rad/sample (not the one with max gain).
  bandwidth :
    Bandwidth frequency range in rad/sample following the equation:

      ``R = exp(-bandwidth / 2)``

    where R is the pole amplitude (radius).

  Returns
  -------
  A ZFilter object.
  Gain is normalized to have peak with 0 dB (1.0 amplitude).

  """
  bandwidth = thub(bandwidth, 1)
  R = exp(-bandwidth * .5)
  R = thub(R, 3)
  freq = thub(freq, 2)
  gain = (1 - R ** 2) * sin(freq)
  denominator = 1 - 2 * R * cos(freq) * z ** -1 + R ** 2 * z ** -2
  return gain / denominator


@resonator.strategy("z_exp")
def resonator(freq, bandwidth):
  """
  Resonator filter with 2-zeros and 2-poles (conjugated pair). The zeros are
  at the `1` and `-1` (both at the real axis, i.e., at the DC and the Nyquist
  rate), with exponential approximation for bandwidth calculation.

  Parameters
  ----------
  freq :
    Resonant frequency in rad/sample (max gain).
  bandwidth :
    Bandwidth frequency range in rad/sample following the equation:

      ``R = exp(-bandwidth / 2)``

    where R is the pole amplitude (radius).

  Returns
  -------
  A ZFilter object.
  Gain is normalized to have peak with 0 dB (1.0 amplitude).

  """
  bandwidth = thub(bandwidth, 1)
  R = exp(-bandwidth * .5)
  R = thub(R, 5)
  cost = cos(freq) * (1 + R ** 2) / (2 * R)
  gain = (1 - R ** 2) * .5
  numerator = 1 - z ** -2
  denominator = 1 - 2 * R * cost * z ** -1 + R ** 2 * z ** -2
  return gain * numerator / denominator


@resonator.strategy("freq_z_exp")
def resonator(freq, bandwidth):
  """
  Resonator filter with 2-zeros and 2-poles (conjugated pair). The zeros are
  at the `1` and `-1` (both at the real axis, i.e., at the DC and the Nyquist
  rate), with exponential approximation for bandwidth calculation.
  Given frequency is the denominator frequency, not the resonant frequency.

  Parameters
  ----------
  freq :
    Denominator frequency in rad/sample (not the one with max gain).
  bandwidth :
    Bandwidth frequency range in rad/sample following the equation:

      ``R = exp(-bandwidth / 2)``

    where R is the pole amplitude (radius).

  Returns
  -------
  A ZFilter object.
  Gain is normalized to have peak with 0 dB (1.0 amplitude).

  """
  bandwidth = thub(bandwidth, 1)
  R = exp(-bandwidth * .5)
  R = thub(R, 3)
  gain = (1 - R ** 2) * .5
  numerator = 1 - z ** -2
  denominator = 1 - 2 * R * cos(freq) * z ** -1 + R ** 2 * z ** -2
  return gain * numerator / denominator


lowpass = StrategyDict("lowpass")


@lowpass.strategy("pole")
def lowpass(cutoff):
  """
  Low-pass filter with one pole and no zeros (constant numerator), with
  high-precision cut-off frequency calculation.

  Parameters
  ----------
  cutoff :
    Cut-off frequency in rad/sample. It defines the filter frequency in which
    the squared gain is `50%` (a.k.a. magnitude gain is `sqrt(2) / 2` and
    power gain is about `3.0103 dB`).
    Should be a value between 0 and pi.

  Returns
  -------
  A ZFilter object.
  Gain is normalized to have peak with 0 dB (1.0 amplitude) at the DC
  frequency (zero rad/sample).

  """
  cutoff = thub(cutoff, 1)
  x = 2 - cos(cutoff)
  x = thub(x,2)
  R = x - sqrt(x ** 2 - 1)
  R = thub(R, 2)
  return (1 - R) / (1 - R * z ** -1)


@lowpass.strategy("pole_exp")
def lowpass(cutoff):
  """
  Low-pass filter with one pole and no zeros (constant numerator), with
  exponential approximation for cut-off frequency calculation, found by a
  matching the one-pole Laplace lowpass filter.

  Parameters
  ----------
  cutoff :
    Cut-off frequency in rad/sample following the equation:

      ``R = exp(-cutoff)``

    where R is the pole amplitude (radius).

  Returns
  -------
  A ZFilter object.
  Gain is normalized to have peak with 0 dB (1.0 amplitude) at the DC
  frequency (zero rad/sample).

  """
  cutoff = thub(cutoff, 1)
  R = exp(-cutoff)
  R = thub(R, 2)
  return (1 - R) / (1 - R * z ** -1)


highpass = StrategyDict("highpass")


@highpass.strategy("pole")
def highpass(cutoff):
  """
  High-pass filter with one pole and no zeros (constant numerator), with
  high-precision cut-off frequency calculation.

  Parameters
  ----------
  cutoff :
    Cut-off frequency in rad/sample. It defines the filter frequency in which
    the squared gain is `50%` (a.k.a. magnitude gain is `sqrt(2) / 2` and
    power gain is about `3.0103 dB`).
    Should be a value between 0 and pi.

  Returns
  -------
  A ZFilter object.
  Gain is normalized to have peak with 0 dB (1.0 amplitude) at the Nyquist
  frequency (pi rad/sample).

  """
  rev_cutoff = thub(pi - cutoff, 1)
  x = 2 - cos(rev_cutoff)
  x = thub(x,2)
  R = x - sqrt(x ** 2 - 1)
  R = thub(R, 2)
  return (1 - R) / (1 + R * z ** -1)
