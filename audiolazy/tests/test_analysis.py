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
Testing module for the lazy_analysis module
"""

from __future__ import division

import pytest
p = pytest.mark.parametrize

from functools import reduce
from itertools import compress
import operator

# Audiolazy internal imports
from ..lazy_analysis import (window, wsymm, zcross, envelope, maverage, clip,
                             unwrap, amdf, overlap_add, stft)
from ..lazy_stream import Stream, thub
from ..lazy_misc import almost_eq, rint
from ..lazy_compat import xrange, orange, xzip, xmap
from ..lazy_synth import line, white_noise, ones, sinusoid, zeros
from ..lazy_math import ceil, inf, pi, cexp
from ..lazy_core import OpMethod
from ..lazy_itertools import chain, repeat, count

@p("wnd", window)
class TestWindowWsymm(object):

  def test_empty(self, wnd):
    assert wnd(0) == []
    assert wnd.symm(0) == []

  @p("size", [1, 2, 3, 4, 16, 128, 256, 512, 1024, 768])
  def test_min_max_len_periodic_symmetry_besides_one_sample(self, wnd, size):
    data = wnd(size)
    assert max(data) <= 1.0
    assert min(data) >= 0.0
    assert len(data) == size
    assert almost_eq(data[1:], data[:0:-1])

  @p("size", [1, 2, 3, 4, 16, 128, 256, 512, 1024, 768])
  def test_symm_with_size_at_least_2_compared_with_periodic(self, wnd, size):
    period = wnd(size) # The default "symm" value is False
    symm = wnd.symm(size + 1)
    assert len(symm) - 1 == len(period) == size
    assert symm[:-1] == period
    assert almost_eq.diff(symm[0], symm[-1])
    assert almost_eq(symm[1:-1], symm[-2:0:-1])

  def test_symm_size_1(self, wnd):
    assert wnd.symm(1) == [1.0]

  def test_distinct_periodic_and_symm(self, wnd):
    same = wnd.__name__ in ["rect"]
    assert (wnd.symm is wnd) == same

    assert wsymm[wnd.__name__] is wnd.symm
    assert window.symm[wnd.__name__] is wnd.symm
    assert wnd.symm.periodic.symm is wnd.symm
    assert wnd.symm.symm is wnd.symm

    assert window[wnd.symm.__name__] is wnd
    assert wsymm.periodic[wnd.symm.__name__] is wnd
    assert wnd.symm.periodic is wnd
    assert wnd.periodic.periodic is wnd


class TestZCross(object):

  def test_empty(self):
    assert list(zcross([])) == []

  @p("n", orange(1, 5))
  def test_small_sizes_no_cross(self, n):
    output = zcross(xrange(n))
    assert isinstance(output, Stream)
    assert list(output) == [0] * n

  @p("d0", [-1, 0, 1])
  @p("d1", [-1, 0, 1])
  def test_pair_combinations(self, d0, d1):
    crossed = 1 if (d0 + d1 == 0) and (d0 != 0) else 0
    assert tuple(zcross([d0, d1])) == (0, crossed)

  @p(("data", "expected"),
     [((0., .1, .5, 1.), (0, 0, 0, 0)),
      ((0., .12, -1.), (0, 0, 1)),
      ((0, -.1, 1), (0, 0, 0)),
      ((1., 0., -.09, .5, -1.), (0, 0, 0, 0, 1)),
      ((1., -.1, -.1, -.2, .05, .1, .05, .2), (0, 0, 0, 1, 0, 0, 0, 1))
     ])
  def test_inputs_with_dot_one_hysteresis(self, data, expected):
    assert tuple(zcross(data, hysteresis=.1)) == expected

  @p("sign", [1, -1])
  def test_first_sign(self, sign):
    data = [0, 1, -1, 3, -4, -.1, .1, 2]
    output = zcross(data, first_sign=sign)
    assert isinstance(output, Stream)
    expected = list(zcross([sign] + data))[1:] # Skip first "zero" sample
    assert list(output) == expected


class TestEnvelope(object):
  sig = [-5, -2.2, -1, 2, 4., 5., -1., -1.8, -22, -57., 1., 12.]

  @p("env", envelope)
  def test_always_positive_and_keep_size(self, env):
    out_stream = env(self.sig)
    assert isinstance(out_stream, Stream)
    out_list = list(out_stream)
    assert len(out_list) == len(self.sig)
    for el in out_list:
      assert el >= 0.


class TestMAverage(object):

  @p("val", [0, 1, 2, 3., 4.8])
  @p("size", [2, 8, 15, 23])
  @p("strategy", maverage)
  def test_const_input(self, val, size, strategy):
    signal = Stream(val)
    result = strategy(size)(signal)
    small_result = result.take(size - 1)
    assert almost_eq(small_result, list(line(size, 0., val))[1:])
    const_result = result.take(int(2.5 * size))
    for el in const_result:
      assert almost_eq(el, val)


class TestClip(object):

  @p("low", [None, 0, -3])
  @p("high", [None, 0, 5])
  def test_with_list_based_range(self, low, high):
    data = orange(-10, 10)
    result = clip(data, low=low, high=high)
    assert isinstance(result, Stream)
    if low is None or low < -10:
      low = -10
    if high is None or high > 10:
      high = 10
    expected = [low] * (10 + low) + orange(low, high) + [high] * (10 - high)
    assert expected == list(result)

  def test_with_inverted_high_and_low(self):
    with pytest.raises(ValueError):
      clip([], low=4, high=3.9)


class TestUnwrap(object):

  @p(("data", "out_data"),[
    ([0, 27, 11, 19, -1, -19, 48, 12], [0, -3, 1, 9, 9, 11, 8, 12]),
    ([0, 10, -10, 20, -20, 30, -30, 40, -40, 50], [0] * 10),
    ([-55, -49, -40, -38, -29, -17, -25], [-55, -49, -50, -48, -49, -47, -55]),
  ])
  def test_max_delta_8_step_10(self, data, out_data):
    assert list(unwrap(data, max_delta=8, step=10)) == out_data


class TestAMDF(object):

  schema = ("sig", "lag", "size", "expected")
  signal = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]
  table_test = [
    (signal, 1, 1, [1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0, 1.0, 1.0]),
    (signal, 2, 1, [1.0,  2.0,  2.0,  0.0,  2.0,  0.0, 2.0, 0.0, 2.0]),
    (signal, 3, 1, [1.0,  2.0,  3.0,  1.0,  1.0,  1.0, 1.0, 1.0, 1.0]),
    (signal, 1, 2, [0.5,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0, 1.0, 1.0]),
    (signal, 2, 2, [0.5,  1.5,  2.0,  1.0,  1.0,  1.0, 1.0, 1.0, 1.0]),
    (signal, 3, 2, [0.5,  1.5,  2.5,  2.0,  1.0,  1.0, 1.0, 1.0, 1.0]),
    (signal, 1, 4, [0.25, 0.5,  0.75, 1.0,  1.0,  1.0, 1.0, 1.0, 1.0]),
    (signal, 2, 4, [0.25, 0.75, 1.25, 1.25, 1.5,  1.0, 1.0, 1.0, 1.0]),
    (signal, 3, 4, [0.25, 0.75, 1.5,  1.75, 1.75, 1.5, 1.0, 1.0, 1.0]),
  ]

  @p(schema, table_test)
  def test_input_output_mapping(self, sig, lag, size, expected):
    filt = amdf(lag, size)
    assert callable(filt)
    assert almost_eq(list(filt(sig)), expected)

  @p("size", [1, 12])
  def test_lag_zero(self, size):
    sig_size = 200
    zero = 0
    filt = amdf(lag=0, size=size)
    sig = list(white_noise(sig_size))
    assert callable(filt)
    assert list(filt(sig, zero=zero)) == [zero for el in sig]


@p("oadd", overlap_add)
class TestOverlapAdd(object):

  # A list of 7-sized lists (blocks) without any zero
  list_data = [[ .1,  .6,  .17,  .4, -.8,  .1,  -.7],
               [ .8,  .7,  .9,  -.6,  .7, -.15,  .3],
               [ .4, -.2,  .4,   .1,  .1, -.3,  -.95],
               [-.3,  .54, .12,  .1, -.8, -.3,   .8],
               [.04, -.8, -.43,  .2,  .1,  .9,  -.5]]

  def test_simple_size_7_hop_3_from_lists(self, oadd):
    wnd = [.1, .2, .3, .4, .3, .2, .1]
    ratio = .6 # Expected normalization ratio
    result = oadd(self.list_data, hop=3, wnd=wnd, normalize=False)
    assert isinstance(result, Stream)
    result_list = list(result)
    # Resulting size is (number of blocks - 1) * hop + size
    length = (len(self.list_data) - 1) * 3 + 7
    assert len(result_list) == length

    # Try applying the window externally
    wdata = [[w * r for w, r in xzip(wnd, row)] for row in self.list_data]
    pre_windowed_result = oadd(wdata, hop=3, wnd=None, normalize=False)
    assert result_list == list(pre_windowed_result)

    # Overlapping and adding manually to a list (for size=7 and hop=3)
    expected = [0.] * length
    for blk_idx, blk in enumerate(wdata):
      start = blk_idx * 3
      stop = start + 7
      expected[start:stop] = list(expected[start:stop] + Stream(blk))
    assert expected == result_list

    # Try comparing with the normalized version
    result_norm = oadd(self.list_data, hop=3, wnd=wnd, normalize=True)
    assert almost_eq(expected[0] / result_norm.peek(), ratio)
    assert almost_eq(result_list, list(result_norm * ratio))

  def test_empty(self, oadd):
    data = oadd([])
    assert isinstance(data, Stream)
    assert list(data) == []

  @p("wnd", [None, window.rect, window.triangular])
  def test_size_1_hop_1_sameness(self, oadd, wnd):
    raw_data = [1., -4., 3., -1., 5., -4., 2., 3.]
    blk_sig = Stream(raw_data).blocks(size=1, hop=1)
    data = oadd(blk_sig, wnd=wnd).take(200)
    assert list(data) ==  raw_data

  @p("size", [512, 128, 12, 2])
  @p("dur", [1038, 719, 18])
  def test_ones_detect_size_with_hop_half_no_normalize(self, oadd, size, dur):
    hop = size // 2
    blk_sig = ones(dur).blocks(size, hop)
    result = oadd(blk_sig, hop=hop, wnd=None, normalize=False)
    data = list(result)
    length = int(ceil(dur / hop) * hop) if dur >= hop else 0
    assert len(data) == length
    one_again = max(0, length - hop)
    twos_start = hop if dur > size else 0
    assert all(el == 1. for el in data[:twos_start])
    assert all(el == 2. for el in data[twos_start:one_again])
    assert all(el == 1. for el in data[one_again:dur])
    assert all(el == 0. for el in data[dur:])

  @p("wnd", [None, window.rect])
  @p("normalize", [True, False])
  def test_size_5_hop_2_rect_window(self, oadd, wnd, normalize):
    raw_data = [5, 4, 3, -2, -3, 4] * 7 # 42-sampled example
    blk_sig = Stream(raw_data).blocks(size=5, hop=2) # 43 (1-sample zero pad)
    result = oadd(blk_sig, size=5, hop=2, wnd=wnd, normalize=normalize)
    assert isinstance(result, Stream)
    result_list = result.take(100)
    weights = [1, 1, 2] + Stream(2, 3).take(37) + [2, 1]
    expected_no_normalize = [x * w for x, w in xzip(raw_data, weights)] + [0.]
    if normalize:
      assert almost_eq([el / 3 for el in expected_no_normalize], result_list)
    else:
      assert list(expected_no_normalize) == result_list

  data1 = list(line(197, .4, -7.7) ** 3 * Stream(.7, .9, .4)) # Arbitrary
  data2 = (sinusoid(freq=.2, phase=pi * sinusoid(.07389) ** 5) * 18).take(314)

  @p("size", [8, 6, 4, 17])
  @p("wnd", [window.triangle, window.hamming, window.hann, window.bartlett])
  @p("data", [data1, data2])
  def test_size_minus_hop_is_3_and_detect_size_no_normalize(self, oadd, size,
                                                            wnd, data):
    hop = size - 3
    result = oadd(Stream(data).blocks(size=size, hop=hop),
                  hop=hop, wnd=wnd, normalize=False).take(inf)
    wnd_list = wnd(size)
    expected = None
    for blk in Stream(data).blocks(size=size, hop=hop):
      blk_windowed = [bi * wi for bi, wi in xzip(blk, wnd_list)]
      if expected:
        expected[-3] += blk_windowed[0]
        expected[-2] += blk_windowed[1]
        expected[-1] += blk_windowed[2]
        expected.extend(blk_windowed[3:])
      else: # First time
        expected = blk_windowed
    assert almost_eq(expected, list(result))

  @p(("size", "hop"), [
    (256, 255),
    (200, 50),
    (128, 9),
    (128, 100),
    (128, 64),
    (100, 100),
    (17, 3),
    (3, 2),
    (2, 1),
  ])
  @p("wnd", [
    window.triangle, window.hamming, window.hann, window.bartlett,
    zeros, # Just for fun (this shouldn't break)
    None,  # Rectangular
    lambda n: line(n) ** 2,                       # Asymmetric iterable
    lambda n: (line(n, -2, 1) ** 3).take(inf),    # Negative value
    lambda n: [-el for el in window.triangle(n)], # Negative-only value
  ])
  def test_normalize(self, oadd, size, hop, wnd):
    # Apply the overlap_add
    length = 719
    result = oadd(ones(length).blocks(size=size, hop=hop),
                  hop=hop, wnd=wnd, normalize=True).take(inf)
    assert len(result) >= length

    # Content verification
    wnd_list = [1.] * size if wnd is None else list(wnd(size))
    if all(el == 0. for el in wnd_list): # Zeros! (or bartlett(2))
      assert all(el == 0. for el in result)
    else:

      # All common cases with a valid window
      assert all(el == 0. for el in result[length:]) # Blockenizing zero pad
      one = 1 + 2e-16 # For one significand bit tolerance (needed for
                      # size = 128; hop in [9, 64]; and
                      # wnd in [triangle, negatived triangle])
      assert all(-one <= el <= one for el in result[:length])

      wnd_gain = max(abs(el) for el in wnd_list)
      wnd_list = [el / wnd_gain for el in wnd_list]
      wnd_max = max(wnd_list)
      wnd_min = min(wnd_list)

      if wnd_max * wnd_min >= 0: # Same signal (perhaps including zero)
        if wnd_max >= abs(wnd_min): # No negative values
          assert almost_eq(max(result[:length]), wnd_max)
        else: # No positive values
          assert almost_eq(min(result[:length]), wnd_min)

        for op in OpMethod.get("> >= < <="):
          if all(op.func(el, 0) for el in wnd_list):
            assert all(op.func(el, 0) for el in result[:length])

      else: # Can subtract, do it all again with the abs(wnd)
        wnd_pos = list(xmap(abs, wnd_list))
        rmax = oadd(ones(length).blocks(size=size, hop=hop),
                    hop=hop, wnd=wnd_pos, normalize=True).take(inf)

        assert all(-1 <= el <= 1 for el in rmax[:length]) # Need no tolerance!
        assert len(rmax) == len(result)
        assert all(rmi >= abs(ri) for rmi, ri in xzip(rmax, result))
        assert almost_eq(max(rmax[:length]), max(wnd_pos))

        if 0. in wnd_pos:
          assert all(el >= 0 for el in rmax[:length])
        else:
          assert all(el > 0 for el in rmax[:length])

  @p("wnd", [25, lambda n: 42, lambda n: None])
  def test_invalid_window(self, oadd, wnd):
    result = oadd(ones(500).blocks(1), wnd=wnd)
    with pytest.raises(TypeError) as exc:
      result.take()
    msg_words = ["window", "should", "iterable", "callable"]
    message = str(exc.value).lower()
    assert all(word in message for word in msg_words)

  @p("wdclass", [float, int])
  @p("sdclass", [float, int])
  @p("wconstructor", [tuple, list, Stream, iter])
  def test_float_ints_for_iterable_window_and_signal(self, oadd, wdclass,
                                                     sdclass, wconstructor):
    size = 3 # For a [1, 2, 3] window
    hop = 2

    wnd = wconstructor(wdclass(el) for el in [1, 2, 3])
    wnd_normalized = wconstructor([.25, .5, .75]) # This can't be int

    sig = thub(Stream(7, 9, -2, 1).map(sdclass), 2)
    result_no_norm = oadd(sig.blocks(size=size, hop=hop),
                          hop=hop, wnd=wnd_normalized, normalize=False)
    result_norm = oadd(sig.blocks(size=size, hop=hop),
                       hop=hop, wnd=wnd, normalize=True)
    expected = chain([1], Stream(2, 4)) * Stream(7, 9, -2, 1) * .25

    # Powers of 2 in wnd_normalized allows equalness testing for floats
    assert result_no_norm.peek(250) == expected.take(250)
    assert result_no_norm.take(400) == result_norm.take(400)

  @p("size", [8, 6]) # Actual size of each block in list_data is 7
  def test_wrong_declared_size(self, oadd, size):
    result = oadd(self.list_data, size=size, hop=3)
    with pytest.raises(ValueError):
      result.peek()

  @p("size", [8, 6])
  def test_wrong_window_size(self, oadd, size):
    result = oadd(self.list_data, hop=3, wnd=window.triangle(size))
    with pytest.raises(ValueError):
      result.peek()

  def test_no_hop(self, oadd):
    concat = lambda seq: reduce(lambda a, b: a + b, seq)
    wnd = window.triangle(len(self.list_data[0]))
    wdata = [[w * r for w, r in xzip(wnd, row)] for row in self.list_data]
    result_no_wnd = oadd(self.list_data, wnd=None)
    result_wnd = oadd(self.list_data, wnd=wnd, normalize=False)
    assert concat(self.list_data) == list(result_no_wnd)
    assert concat(wdata) == list(result_wnd)


class KeepDefault:
  """ Possible parameter (the class itself) for parametrized tests """


class TestSTFT(object):

  @p("strategy", [stft.real, stft.complex, stft.complex_real])
  @p("size", [256, 17])
  @p("hop_percent", [1, .5, .23])
  @p("zero_phase", [True, False])
  def test_do_nothing(self, strategy, size, hop_percent, zero_phase):
    data = white_noise()
    hop = rint(size * hop_percent)
    identity = lambda blk: blk
    st = strategy if zero_phase else strategy(before=None, after=None)
    func = st(identity, size=size, hop=hop)
    env = overlap_add(repeat([1.] * size), hop=hop)
    assert almost_eq(data.peek(5000), (func(data) / env).take(5000))

  @p("strategy", [stft.real, stft.complex, stft.complex_real])
  @p("size", [128, 53])
  @p("hop_percent", [1, .5, .31])
  def test_zero_phase(self, strategy, size, hop_percent):
    data = thub(white_noise(), 2)
    hop = rint(size * hop_percent)
    identity = lambda blk: blk
    analyzer = strategy(identity, size=size, hop=hop, ola=None,
                        inverse_transform=None, after=None)
    blocks_zero_phase = analyzer(data).take(50)
    blocks = analyzer(data, before=None).take(50)
    assert len(blocks) == len(blocks_zero_phase) == 50
    blk_len = size // 2 + 1 if strategy is stft.real else size
    change = cexp(line(size, 0, 2j * pi * (size // 2))).take(blk_len)
    for blk, blk_zp in xzip(blocks, blocks_zero_phase):
      assert almost_eq(blk * change, blk_zp)

  integer_wnd = lambda size: [int(500 * el) for el in window.hamming(size)]

  @p("use_before", [True, False])
  @p("use_trans", [True, False])
  @p("use_itrans", [True, False])
  @p("use_after", [True, False])
  @p("wnd", [integer_wnd, None, KeepDefault])
  @p("use_ola", [True, False])
  def test_call_order_and_data_process(self, use_before, use_trans, use_ola,
                                             use_itrans, use_after, wnd):
    """
    Tests whether the stft operation sequence is called in this order:

      "blockenize" -> "windowing" -> before -> transform -> func ...
                                        ... -> inverse_transform -> after

    and if the data flows from a step to the next one, including the cases
    where some parts of this process is [explicitly] missing. Also, the
    result from after in the above process should be the stft output (if
    ``ola=None``) or the overlap-add input. This test doesn't need Numpy.
    """
    size = 15 + 2 * sum([use_before, use_trans, use_itrans, use_after])
    hop = 13 # Arbitrary size and hop

    def appender_factory(name):
      """ Appender Factory, needed to create a closure for a single name """
      def appender(*args):
        """
        Appends the name to "name_order" and the output data to "data_order".
        This output data is found by an elementwise multiplication of the 1st
        input by 1, 2, 4, 8 or 16 (respective to the last 5 operations in the
        stft sequence).
        """
        assert len(args) == (2 if "trans" in name else 1)
        name_order.append(name)
        m = (1 << names.index(name))
        new_data = [el * m for el in args[0]]
        data_order.append(new_data)
        return new_data
      return appender

    def ola(blk_sig, **kwargs):
      """
      Fake overlap-add that outputs each input block reversed.
      """
      assert kwargs.pop("size") == size
      assert kwargs.pop("hop") == hop
      assert kwargs == {}

      def internal_ola():
        """ Overlap-add process done after the "after" """
        assert "name_order" in locals()
        for blk in blk_sig:
          assert name_order == expected_names
          yield blk[::-1]

      # This is called only once and before all the above processing
      assert "name_order" not in locals()
      assert "called" not in vars(ola)
      ola.called = True
      return internal_ola()

    selector = [use_before, use_trans, True, use_itrans, use_after]
    names = ["before", "transform", "func", "inverse_transform", "after"]
    expected_names = [name for sel, name in xzip(selector, names) if sel]
    expected_multipliers = list(compress(1 << count(), selector)) # 1 2 4 8 16

    kwargs = {name: appender_factory(name) if sel else None
              for sel, name in xzip(selector, names)}
    processor = stft(ola=ola if use_ola else None, **kwargs)

    sig = thub(white_noise(low=-1e3, high=1e3).map(int), 2)
    if wnd is KeepDefault:
      blocks_output = processor(sig, size=size, hop=hop)
    else:
      blocks_output = processor(sig, size=size, hop=hop, wnd=wnd)

    for sig_blk in sig.blocks(size=size, hop=hop).limit(5):
      if wnd in [KeepDefault, None]:
        data_order = [sig_blk]
      else:
        data_order = [list(xmap(operator.mul, sig_blk, wnd(size)))]
      name_order = []
      result = blocks_output.take() # This changes data and order
      if use_ola:
        result = result[::-1] # Undo the fake overlap-add action
      assert name_order == expected_names
      assert data_order[-1] == result
      pairs = Stream(data_order).blocks(size=2, hop=1)
      for (a, b), m in xzip(pairs, expected_multipliers):
        assert [el * m for el in a] == b
