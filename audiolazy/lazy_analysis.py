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
Audio analysis and block processing module
"""

from __future__ import division

from math import sin, cos, pi
from collections import deque, Sequence, Iterable
from functools import wraps, reduce
from itertools import chain
import operator

# Audiolazy internal imports
from .lazy_core import StrategyDict
from .lazy_stream import tostream, thub, Stream
from .lazy_math import cexp, ceil
from .lazy_filters import lowpass, z
from .lazy_compat import xrange, xmap, xzip, iteritems
from .lazy_text import format_docstring


__all__ = ["window", "wsymm", "acorr", "lag_matrix", "dft", "zcross",
           "envelope", "maverage", "clip", "unwrap", "amdf", "overlap_add",
           "stft"]


window = StrategyDict("window")
wsymm = StrategyDict("wsymm")

window.symm = wsymm.symm = wsymm
window.periodic = wsymm.periodic = window

window._doc_kwargs = (lambda sname, symm=False, distinct=True, formula=None,
                             name=None, names=None, params=None, math=None,
                             math_symm=None, bib=None, out=None,
                             params_def=None, seealso=None: dict(
  sname = sname,
  symm = symm,
  name = name or sname.capitalize(),
  params = params or "",
  seealso = seealso or "",
  sp_detail = (" (symmetric)" if symm else " (periodic)") if distinct else "",
  out = out or "",

  template_ = """
  {name} windowing/apodization function{sp_detail}.
  {expl}{bib}
  Parameters
  ----------
  size :
    Window size in samples.{params}

  Returns
  -------
  List with the window samples. {out}
  {sp_note}
  Hint
  ----
  All ``window`` and ``wsymm`` strategies have both a ``periodic`` and
  ``symm`` attribute with the respective strategy. The StrategyDict instances
  themselves also have these attributes (with the respective StrategyDict
  instance).{hint_extra}

  See Also
  --------{see_other}{seealso}{see_stft_ola}
  """,

  see_other = "" if not distinct else """
  {other_sdict} :
    StrategyDict instance with {other_sp} windowing/apodization functions.
  {other_sdict}.{sname} :
    {name} windowing/apodization function ({other_sp}).""".format(
    sname = sname,
    name = name or sname.capitalize(),
    other_sp = "periodic" if symm else "symmetric",
    other_sdict = "window" if symm else "wsymm",
  ),

  see_stft_ola = "" if symm or not distinct else """
  stft :
    Short Time Fourier Transform block processor / phase vocoder wrapper.
  overlap_add :
    Overlap-add algorithm for an interables of blocks.""",

  expl = "" if not math else """
  For this model, the resulting :math:`n`-th sample
  (where :math:`n = 0, 1, \\cdots, size - 1`) is:

  .. math:: {math}
  """.format(math = (math_symm or math.replace("size", "size - 1"))
                    if symm else math),

  bib = bib or """
  This window model was taken from:

    ``Harris, F. J. "On the Use of Windows for Harmonic Analysis with the
    Discrete Fourier Transform". Proceedings of the IEEE, vol. 66, no. 1,
    January 1978.``
  """,

  sp_note = ("""
  Warning
  -------
  Don't use this strategy for FFT/DFT/STFT windowing! You should use the
  periodic approach for that. See the F. J. Harris paper for more information.
  """ if symm else """
  Note
  ----
  Be careful as this isn't a "symmetric" window implementation by default, you
  should append the first sample at the end to get a ``size + 1`` symmetric
  window. The "periodic" window implementation returned by this function
  is designed for using directly with DFT/STFT. See the F. J. Harris paper
  for more information on these.

  By default, Numpy, Scipy signal subpackage, GNU Octave and MatLab uses the
  symmetric approach for the window functions, with [1.0] as the result when
  the size is 1 (which means the window is actually empty). Here the
  implementation differ expecting that these functions will be mainly used in
  a DFT/STFT process.
  """) if distinct else """
  Note
  ----
  As this strategy is both "symmetric" and "periodic", ``window.{sname}``
  and ``wsymm.{sname}`` are the very same function/strategy.
  """.format(sname=sname),

  hint_extra = (""" However, in this case, they're the same, i.e.,
  ``window.{sname}`` is ``wsymm.{sname}``.""" if not distinct else
  """ You can get the {other_sp} strategy ``{other_sdict}.{sname}`` with:

  * ``{sdict}.{sname}.{other_meth}``;
  * ``{sdict}.{other_meth}.{sname}`` ({sdict}.{other_meth} is {other_sdict});
  * ``{other_sdict}.{sname}.{other_meth}`` (unneeded ``.{other_meth}``);
  * ``{other_sdict}.{other_meth}.{sname}`` (pleonastically, as
    {other_sdict}.{other_meth} is {other_sdict}).""")
  .format(
    sname = sname,
    sdict = "wsymm" if symm else "window",
    other_meth = "periodic" if symm else "symm",
    other_sp = "periodic" if symm else "symmetric",
    other_sdict = "window" if symm else "wsymm",
  ),
))

window._content_generation_table = [
  dict(
    names = ("hann", "hanning",),
    formula = ".5 * (1 - cos(2 * pi * n / size))",
    math = r"\frac{1}{2} \left[ 1 - \cos \left( \frac{2 \pi n}{size} \right) "
                       r"\right]",
  ),

  dict(
    names = ("hamming",),
    formula = ".54 - .46 * cos(2 * pi * n / size)",
    math = r"0.54 - 0.46 \cos \left( \frac{2 \pi n}{size} \right)",
  ),

  dict(
    names = ("rect", "dirichlet", "rectangular",),
    formula = "1.0",
    name = "Dirichlet/rectangular",
    out = "All values are ones (1.0).",
    distinct = False,
    seealso = """
  ones :
    Lazy ``1.0`` stream generator.""",
  ),

  dict(
    names = ("bartlett",),
    formula = "1 - 2.0 / size * abs(n - size / 2.0)",
    math = r"1 - \frac{2}{size} \left| \frac{n - size}{2} \right|",
    name = "Bartlett (triangular starting with zero)",
    bib = " ",
    seealso = """
  window.triangular :
    Triangular with no zero end-point (periodic).
  wsymm.triangular :
    Triangular with no zero end-point (symmetric).""",
  ),

  dict(
    names = ("triangular", "triangle",),
    formula = "1 - 2.0 / (size + 2) * abs(n - size / 2.0)",
    math = r"1 - \frac{2}{size + 2} \left| \frac{n - size}{2} \right|",
    math_symm = r"1 - \frac{2}{size + 1} \left| \frac{n - size - 1}{2} "
                                       r"\right|",
    name = "Triangular (with no zero end-point)",
    bib = " ",
    seealso = """
  window.bartlett :
    Triangular starting with zero (periodic).
  wsymm.bartlett :
    Triangular starting with zero (symmetric).""",
  ),

  dict(
    names = ("blackman",),
    formula = "(1 - alpha) / 2 + alpha / 2 * cos(4 * pi * n / size)"
              " - .5 * cos(2 * pi * n / size)",
    math = r"\frac{1 - \alpha}{2} "
           r" - \frac{1}{2} \cos \left( \frac{2 \pi n}{size} \right)"
           r" + \frac{\alpha}{2} \cos \left( \frac{4 \pi n}{size} \right)",
    params = """
  alpha :
    Blackman window alpha value. Defaults to 0.16. Use ``2.0 * 1430 / 18608``
    for the 'exact Blackman' window.""",
    params_def = ", alpha=.16"
  ),

  dict(
    names = ("cos",),
    formula = "sin(pi * n / size) ** alpha",
    math = r"\left[ \sin \left( \frac{\pi n}{size} \right) \right]^{\alpha}",
    name = "Cosine to the power of alpha",
    params = """
  alpha :
    Power value. Defaults to 1.""",
    params_def = ", alpha=1"
  ),
]

window._code_template = """
def {sname}(size{params_def}):
  return [{formula} for n in xrange(size)]
"""

wsymm._code_template = """
def {sname}(size{params_def}):
  if size == 1:
    return [1.0]
  size, indexes = size - 1, xrange(size)
  return [{formula} for n in indexes]
"""

def _generate_window_strategies():
  """ Create all window and wsymm strategies """
  for wnd_dict in window._content_generation_table:
    names = wnd_dict["names"]
    sname = wnd_dict["sname"] = names[0]
    wnd_dict.setdefault("params_def", "")
    for sdict in [window, wsymm]:
      docs_dict = window._doc_kwargs(symm = sdict is wsymm, **wnd_dict)
      decorators = [format_docstring(**docs_dict), sdict.strategy(*names)]
      ns = dict(pi=pi, sin=sin, cos=cos, xrange=xrange, __name__=__name__)
      exec(sdict._code_template.format(**wnd_dict), ns, ns)
      reduce(lambda func, dec: dec(func), decorators, ns[sname])
      if not wnd_dict.get("distinct", True):
        wsymm[sname] = window[sname]
        break
    wsymm[sname].periodic = window[sname].periodic = window[sname]
    wsymm[sname].symm = window[sname].symm = wsymm[sname]

_generate_window_strategies()


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
  return lowpass(cutoff)(abs(thub(sig, 1)))


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


def amdf(lag, size):
  """
  Average Magnitude Difference Function non-linear filter for a given
  size and a fixed lag.

  Parameters
  ----------
  lag :
    Time lag, in samples. See ``freq2lag`` if needs conversion from
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
  freq2lag :
    Frequency (in rad/sample) to lag (in samples) converter.

  """
  filt = (1 - z ** -lag).linearize()

  @tostream
  def amdf_filter(sig, zero=0.):
    return maverage(size)(abs(filt(sig, zero=zero)), zero=zero)

  return amdf_filter


overlap_add = StrategyDict("overlap_add")


@overlap_add.strategy("numpy")
@tostream
def overlap_add(blk_sig, size=None, hop=None, wnd=None, normalize=True):
  """
  Overlap-add algorithm using Numpy arrays.

  Parameters
  ----------
  blk_sig :
    An iterable of blocks (sequences), such as the ``Stream.blocks`` result.
  size :
    Block size for each ``blk_sig`` element, in samples.
  hop :
    Number of samples for two adjacent blocks (defaults to the size).
  wnd :
    Windowing function to be applied to each block or any iterable with
    exactly ``size`` elements. If ``None`` (default), applies a rectangular
    window.
  normalize :
    Flag whether the window should be normalized so that the process could
    happen in the [-1; 1] range, dividing the window by its hop gain.
    Default is ``True``.

  Returns
  -------
  A Stream instance with the blocks overlapped and added.

  See Also
  --------
  Stream.blocks :
    Splits the Stream instance into blocks with given size and hop.
  blocks :
    Same to Stream.blocks but for without using the Stream class.
  chain :
    Lazily joins all iterables given as parameters.
  chain.from_iterable :
    Same to ``chain(*data)``, but the ``data`` evaluation is lazy.
  window :
    Window/apodization/tapering functions for a given size as a StrategyDict.

  Note
  ----
  Each block has the window function applied to it and the result is the
  sum of the blocks without any edge-case special treatment for the first
  and last few blocks.
  """
  import numpy as np

  # Finds the size from data, if needed
  if size is None:
    blk_sig = Stream(blk_sig)
    size = len(blk_sig.peek())
  if hop is None:
    hop = size

  # Find the right windowing function to be applied
  if wnd is None:
    wnd = np.ones(size)
  elif callable(wnd) and not isinstance(wnd, Stream):
    wnd = wnd(size)
  if isinstance(wnd, Sequence):
    wnd = np.array(wnd)
  elif isinstance(wnd, Iterable):
    wnd = np.hstack(wnd)
  else:
    raise TypeError("Window should be an iterable or a callable")

  # Normalization to the [-1; 1] range
  if normalize:
    steps = Stream(wnd).blocks(hop).map(np.array)
    gain = np.sum(np.abs(np.vstack(steps)), 0).max()
    if gain: # If gain is zero, normalization couldn't have any effect
      wnd = wnd / gain # Can't use "/=" nor "*=" as Numpy would keep datatype

  # Overlap-add algorithm
  old = np.zeros(size)
  for blk in (wnd * blk for blk in blk_sig):
    blk[:-hop] += old[hop:]
    for el in blk[:hop]:
      yield el
    old = blk
  for el in old[hop:]: # No more blocks, finish yielding the last one
    yield el


@overlap_add.strategy("list")
@tostream
def overlap_add(blk_sig, size=None, hop=None, wnd=None, normalize=True):
  """
  Overlap-add algorithm using lists instead of Numpy arrays. The behavior
  is the same to the ``overlap_add.numpy`` strategy, besides the data types.
  """
  # Finds the size from data, if needed
  if size is None:
    blk_sig = Stream(blk_sig)
    size = len(blk_sig.peek())
  if hop is None:
    hop = size

  # Find the window to be applied, resulting on a list or None
  if wnd is not None:
    if callable(wnd) and not isinstance(wnd, Stream):
      wnd = wnd(size)
    if isinstance(wnd, Iterable):
      wnd = list(wnd)
    else:
      raise TypeError("Window should be an iterable or a callable")

  # Normalization to the [-1; 1] range
  if normalize:
    if wnd:
      steps = Stream(wnd).map(abs).blocks(hop).map(tuple)
      gain = max(xmap(sum, xzip(*steps)))
      if gain: # If gain is zero, normalization couldn't have any effect
        wnd[:] = (w / gain for w in wnd)
    else:
      wnd = [1 / ceil(size / hop)] * size

  # Window application
  if wnd:
    mul = operator.mul
    if len(wnd) != size:
      raise ValueError("Incompatible window size")
    wnd = wnd + [0.] # Allows detecting when block size is wrong
    blk_sig = (xmap(mul, wnd, blk) for blk in blk_sig)

  # Overlap-add algorithm
  add = operator.add
  mem = [0.] * size
  s_h = size - hop
  for blk in xmap(iter, blk_sig):
    mem[:s_h] = xmap(add, mem[hop:], blk)
    mem[s_h:] = blk # Remaining elements
    if len(mem) != size:
      raise ValueError("Wrong block size or declared")
    for el in mem[:hop]:
      yield el
  for el in mem[hop:]: # No more blocks, finish yielding the last one
    yield el


stft = StrategyDict("stft")


@stft.strategy("rfft", "base", "real")
def stft(func=None, **kwparams):
  """
  Short Time Fourier Transform block processor / phase vocoder wrapper.

  This function can be used in many ways:

  * Directly as a signal processor builder, wrapping a spectrum block/grain
    processor function;
  * Directly as a decorator to a block processor;
  * Called without the ``func`` parameter for a partial evalution style
    changing the defaults.

  See the examples below for more information about these use cases.

  The resulting function performs a full block-by-block analysis/synthesis
  phase vocoder keeping this sequence of actions:

  1. Blockenize the signal with the given ``size`` and ``hop``;
  2. Lazily apply the given ``wnd`` window to each block;
  3. Perform the 5 actions calling their functions in order:

    a. ``before``: Pre-processing;
    b. ``transform``: A transform like the FFT;
    c. ``func``: the positional parameter with the single block processor;
    d. ``inverse_transform``: inverse FFT;
    e. ``after``: Post-processing.

  4. Overlap-add with the ``ola`` overlap-add strategy. The given ``ola``
     would deal with its own window application and normalization.

  Any parameter from steps 3 and 4 can be set to ``None`` to skip it from
  the full process, without changing the other [sub]steps. The parameters
  defaults are based on the Numpy FFT subpackage.

  Parameters
  ----------
  func :
    The block/grain processor function that receives a transformed block in
    the frequency domain (the ``transform`` output) and should return the
    processed data (it will be the first ``inverse_transform`` input). This
    parameter shouldn't appear when this function is used as a decorator.
  size :
    Block size for the STFT process, in samples.
  hop :
    Duration in samples between two blocks. Defaults to the ``size`` value.
  transform :
    Function that receives the windowed block (in time domain) and the
    ``size`` as two positional inputs and should return the block (in
    frequency domain). Defaults to ``numpy.fft.rfft``, which outputs a
    Numpy 1D array with length equals to ``size // 2 + 1``.
  inverse_transform :
    Function that receives the processed block (in frequency domain) and the
    ``size`` as two positional inputs and should return the block (in
    time domain). Defaults to ``numpy.fft.irfft``.
  wnd :
    Window function to be called as ``wnd(size)`` or window iterable with
    length equals to ``size``. The windowing/apodization values are used
    before taking the FFT of each block. Defaults to None, which means no
    window should be applied (same behavior of a rectangular window).
  before :
    Function to be applied just before taking the transform, after the
    windowing. Defaults to the ``numpy.fft.ifftshift``, which, together with
    the ``after`` default, puts the time reference at the ``size // 2``
    index of the block, centralizing it for the FFT (e.g. blocks
    ``[0, 1, 0]`` and ``[0, 0, 1, 0]`` would have zero phase). To disable
    this realignment, just change both ``before=None`` and ``after=None``
    keywords.
  after :
    Function to be applied just after the inverse transform, before calling
    the overlap-add (as well as before its windowing, if any). Defaults to
    the ``numpy.fft.fftshift`` function, which undo the changes done by the
    default ``before`` pre-processing for block phase alignment. To avoid
    the default time-domain realignment, set both ``before=None`` and
    ``after=None`` keywords.
  ola :
    Overlap-add strategy. Uses the ``overlap_add`` default strategy when
    not given. The strategy should allow at least size and hop keyword
    arguments, besides a first positional argument for the iterable with
    blocks. If ``ola=None``, the result from using the STFT processor will be
    the ``Stream`` of blocks that would be the overlap-add input.
  ola_* :
    Extra keyword parameters for the overlap-add strategy, if any. The extra
    ``ola_`` prefix is removed when calling it. See the overlap-add strategy
    docs for more information about the valid parameters.

  Returns
  -------
  A function with the same parameters above, besides ``func``, which is
  replaced by the signal input (if func was given). The parameters used when
  building the function should be seen as defaults that can be changed when
  calling the resulting function with the respective keyword arguments.

  Examples
  --------
  Let's process something:

  >>> my_signal = Stream(.1, .3, -.1, -.3, .5, .4, .3)

  Wrapping directly the processor function:

  >>> processor_w = stft(abs, size=64)
  >>> sig = my_signal.copy() # Any iterable
  >>> processor_w(sig)
  <audiolazy.lazy_stream.Stream object at 0x...>
  >>> peek200_w = _.peek(200) # Needs Numpy
  >>> type(peek200_w[0]).__name__ # Result is a signal (numpy.float64 data)
  'float64'

  Keyword parameters in a partial evaluation style (can be reassigned):

  >>> stft64 = stft(size=64) # Same to ``stft`` but with other defaults
  >>> processor_p = stft64(abs)
  >>> sig = my_signal.copy() # Any iterable
  >>> processor_p(sig)
  <audiolazy.lazy_stream.Stream object at 0x...>
  >>> _.peek(200) == peek200_w # This should do the same thing
  True

  As a decorator, this time with other windowing configuration:

  >>> stft64hann = stft64(wnd=window.hann, ola_wnd=window.hann)
  >>> @stft64hann # stft(...) can also be used as an anonymous decorator
  ... def processor_d(blk):
  ...   return abs(blk)
  >>> processor_d(sig) # This leads to a different result
  <audiolazy.lazy_stream.Stream object at 0x...>
  >>> _.peek(200) == peek200_w
  False

  You can also use other iterables as input, and keep the parameters to be
  passed afterwards, as well as change transform calculation:

  >>> stft_no_zero_phase = stft(before=None, after=None)
  >>> stft_no_wnd = stft_no_zero_phase(ola=overlap_add.list, ola_wnd=None,
  ...                                  ola_normalize=False)
  >>> on_blocks = stft_no_wnd(transform=None, inverse_transform=None)
  >>> processor_a = on_blocks(reversed, hop=4) # Reverse
  >>> processor_a([1, 2, 3, 4, 5], size=4, hop=2)
  <audiolazy.lazy_stream.Stream object at 0x...>
  >>> list(_) # From blocks [1, 2, 3, 4] and [3, 4, 5, 0.0]
  [4.0, 3.0, 2.0, 6, 4, 3]
  >>> processor_a([1, 2, 3, 4, 5], size=4) # Default hop instead
  <audiolazy.lazy_stream.Stream object at 0x...>
  >>> list(_) # No overlap, blocks [1, 2, 3, 4] and [5, 0.0, 0.0, 0.0]
  [4, 3, 2, 1, 0.0, 0.0, 0.0, 5]
  >>> processor_a([1, 2, 3, 4, 5]) # Size was never given
  Traceback (most recent call last):
      ...
  TypeError: Missing 'size' argument

  For analysis only, one can set ``ola=None``:

  >>> from numpy.fft import ifftshift # [1, 2, 3, 4, 5] -> [3, 4, 5, 1, 2]
  >>> analyzer = stft(ifftshift, ola=None, size=8, hop=2)
  >>> sig = Stream(1, 0, -1, 0) # A pi/2 rad/sample cosine signal
  >>> result = analyzer(sig)
  >>> result
  <audiolazy.lazy_stream.Stream object at 0x...>

  Let's see the result contents. That processing "rotates" the frequencies,
  converting the original ``[0, 0, 4, 0, 0]`` real FFT block to a
  ``[4, 0, 0, 0, 0]`` block, which means the block cosine was moved to
  a DC-only signal keeping original energy/integral:

  >>> result.take()
  array([ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5])
  >>> result.take() # From [0, 0, -4, 0, 0] to [-4, 0, 0, 0, 0]
  array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5])

  Note
  ----
  Parameters should be passed as keyword arguments. The only exception
  is ``func`` for this function and ``sig`` for the returned function,
  which are always the first positional argument, ald also the one that
  shouldn't appear when using this function as a decorator.

  Hint
  ----
  1. When using Numpy FFT, one can keep data in place and return the
     changed input block to save time;
  2. Actually, there's nothing in this function that imposes FFT or Numpy
     besides the default values. One can still use this even for other
     transforms that have nothing to do with the Fourier Transform.

  See Also
  --------
  overlap_add :
    Overlap-add algorithm for an iterable (e.g. a Stream instance) of blocks
    (sequences such as lists or Numpy arrays). It's also a StrategyDict.
  window :
    Window/apodization/tapering functions for a given size as a StrategyDict.
  """
  # Using as a decorator or to "replicate" this function with other defaults
  if func is None:
    cfi = chain.from_iterable
    mix_dict = lambda *dicts: dict(cfi(iteritems(d) for d in dicts))
    result = lambda f=None, **new_kws: stft(f, **mix_dict(kwparams, new_kws))
    return result

  # Using directly
  @tostream
  @wraps(func)
  def wrapper(sig, **kwargs):
    kws = kwparams.copy()
    kws.update(kwargs)

    if "size" not in kws:
      raise TypeError("Missing 'size' argument")
    if "hop" in kws and kws["hop"] > kws["size"]:
      raise ValueError("Hop value can't be higher than size")

    blk_params = {"size": kws.pop("size")}
    blk_params["hop"] = kws.pop("hop", None)
    ola_params = blk_params.copy() # Size and hop

    blk_params["wnd"] = kws.pop("wnd", None)
    ola = kws.pop("ola", overlap_add)

    class NotSpecified(object):
      pass
    for name in ["transform", "inverse_transform", "before", "after"]:
      blk_params[name] = kws.pop(name, NotSpecified)

    for k, v in kws.items():
      if k.startswith("ola_"):
        if ola is not None:
          ola_params[k[len("ola_"):]] = v
        else:
          raise TypeError("Extra '{}' argument with no overlap-add "
                          "strategy".format(k))
      else:
        raise TypeError("Unknown '{}' extra argument".format(k))

    def blk_gen(size, hop, wnd, transform, inverse_transform, before, after):
      if transform is NotSpecified:
        from numpy.fft import rfft as transform
      if inverse_transform is NotSpecified:
        from numpy.fft import irfft as inverse_transform
      if before is NotSpecified:
        from numpy.fft import ifftshift as before
      if after is NotSpecified:
        from numpy.fft import fftshift as after

      # Find the right windowing function to be applied
      if callable(wnd) and not isinstance(wnd, Stream):
        wnd = wnd(size)
      if isinstance(wnd, Iterable):
        wnd = list(wnd)
        if len(wnd) != size:
          raise ValueError("Incompatible window size")
      elif wnd is not None:
        raise TypeError("Window should be an iterable or a callable")

      # Pad size lambdas
      trans = transform and (lambda blk: transform(blk, size))
      itrans = inverse_transform and (lambda blk:
                                        inverse_transform(blk, size))

      # Continuation style calling
      funcs = [f for f in [before, trans, func, itrans, after]
                 if f is not None]
      process = lambda blk: reduce(lambda data, f: f(data), funcs, blk)

      if wnd is None:
        for blk in Stream(sig).blocks(size=size, hop=hop):
          yield process(blk)
      else:
        blk_with_wnd = wnd[:]
        mul = operator.mul
        for blk in Stream(sig).blocks(size=size, hop=hop):
          blk_with_wnd[:] = xmap(mul, blk, wnd)
          yield process(blk_with_wnd)

    if ola is None:
      return blk_gen(**blk_params)
    else:
      return ola(blk_gen(**blk_params), **ola_params)

  return wrapper


@stft.strategy("cfft", "complex")
def stft(func=None, **kwparams):
  """
  Short Time Fourier Transform for complex data.

  Same to the default STFT strategy, but with new defaults. This is the same
  to:

  .. code-block:: python

    stft.base(transform=numpy.fft.fft, inverse_transform=numpy.fft.ifft)

  See ``stft.base`` docs for more.
  """
  from numpy.fft import fft, ifft
  return stft.base(transform=fft, inverse_transform=ifft)(func, **kwparams)


@stft.strategy("cfftr", "complex_real")
def stft(func=None, **kwparams):
  """
  Short Time Fourier Transform for real data keeping the full FFT block.

  Same to the default STFT strategy, but with new defaults. This is the same
  to:

  .. code-block:: python

    stft.base(transform=numpy.fft.fft,
              inverse_transform=lambda *args: numpy.fft.ifft(*args).real)

  See ``stft.base`` docs for more.
  """
  from numpy.fft import fft, ifft
  ifft_r = lambda *args: ifft(*args).real
  return stft.base(transform=fft, inverse_transform=ifft_r)(func, **kwparams)
