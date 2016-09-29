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
Testing module for the lazy_lpc module
"""

import pytest
p = pytest.mark.parametrize

import itertools as it
import operator
from functools import reduce

# Audiolazy internal imports
from ..lazy_lpc import (toeplitz, levinson_durbin, lpc, parcor,
                        parcor_stable, lsf, lsf_stable)
from ..lazy_misc import almost_eq
from ..lazy_compat import xrange, xmap
from ..lazy_filters import z, ZFilter


class TestLPCParcorLSFAndStability(object):

  # Some audio sequences as examples
  block_alternate = [1., 1./2., -1./8., 1./32., -1./128., 1./256., -1./512.,
                     1./1024., -1./4096., 1./8192.]
  real_block = [3744, 2336, -400, -3088, -5808, -6512, -6016, -4576, -3088,
    -1840, -944, 176, 1600, 2976, 3808, 3600, 2384, 656, -688, -1872, -2576,
    -3184, -3920, -4144, -3584, -2080, 144, 2144, 3472, 4032, 4064, 4048,
    4016, 3984, 4032, 4080, 3888, 1712, -1296, -4208, -6720, -6848, -5904,
    -4080, -2480, -1200, -560, 592, 1856, 3264, 4128, 3936, 2480, 480, -1360,
    -2592, -3184, -3456, -3760, -3856, -3472, -2160, -80, 2112, 3760, 4416,
    4304, 3968, 3616, 3568, 3840, 4160, 4144, 2176, -1024, -4144, -6800,
    -7120, -5952, -3920, -2096, -800, -352, 352, 1408, 2768, 4032, 4304,
    3280, 1168, -992, -2640, -3584, -3664, -3680, -3504, -3136, -2304, -800,
    1232, 3088, 4352, 4720, 4432, 3840, 3312, 3248, 3664, 4144, 2928, 96,
    -3088, -6448, -7648, -6928, -4864, -2416, -512, 208, 544, 976, 1760, 3104,
    4064, 4016, 2624, 416, -1904, -3696, -4368, -4320, -3744, -2960, -1984,
    -848, 576, 2112, 3504, 4448, 4832, 4656, 4048, 3552, 3360, 3616, 2912,
    736, -1920, -5280, -7264, -7568, -6320, -3968, -1408, 288, 1184, 1600,
    1744, 2416, 3184
  ]

  # Each dictionary entry is a massive test family, for each strategy. The
  # dictionary entry "k" have the PARCOR coefficients, but not reversed
  table_data = [
    {"blk": block_alternate,
     "strategies": (lpc.autocor, lpc.nautocor, lpc.kautocor),
     "order": 3,
     "lpc": 1 - 0.457681292332 * z ** -1 \
               + 0.297451538058 * z ** -2 \
               - 0.162014679229 * z ** -3,
     "lpc_error": 1.03182436137,
     "k": [-0.342081949224, 0.229319810099, -0.162014679229],
     "lsf": (-2.0461731139434804, -1.4224191795241481, -0.69583069081054594,
             0.0, 0.69583069081054594, 1.4224191795241481, 2.0461731139434804,
             3.1415926535897931),
     "stable": True,
    },

    {"blk": block_alternate,
     "strategies": (lpc.covar, lpc.kcovar),
     "order": 3,
     "lpc": 1 + 0.712617839203 * z ** -1 \
              + 0.114426147267 * z ** -2 \
              + 0.000614348391636 * z ** -3,
     "lpc_error": 3.64963839634e-06,
     "k": [0.6396366551286051, 0.1139883946659675, 0.000614348391636012],
     "lsf": (-2.6203603524613603, -1.9347821510481453, -1.0349253486092844,
             0.0, 1.0349253486092844, 1.9347821510481453, 2.6203603524613603,
             3.1415926535897931),
     "stable": True,
    },

    {"blk": real_block,
     "strategies": (lpc.covar, lpc.kcovar),
     "order": 2,
     "lpc": 1 - 1.765972108770 * z ** -1 \
              + 0.918762660191 * z ** -2,
     "lpc_error": 47473016.7152,
     "k": [-0.9203702705945026, 0.9187626601910946],
     "lsf": (-0.5691351064785074, -0.39341656885093923, 0.0,
             0.39341656885093923, 0.5691351064785074, 3.1415926535897931),
     "stable": True,
    },

    {"blk": real_block,
     "strategies": (lpc.covar, lpc.kcovar),
     "order": 6,
     "lpc": 1 - 2.05030891 * z ** -1 \
              + 1.30257925 * z ** -2 \
              + 0.22477252 * z ** -3 \
              - 0.25553702 * z ** -4 \
              - 0.47493330 * z ** -5 \
              + 0.43261407 * z ** -6,
     "lpc_error": 17271980.6421,
     "k": [-0.9211953262806057, 0.9187524349022875, -0.5396255901174379,
           0.1923394201597473, 0.5069344687875105, 0.4326140684936846],
     "lsf": (-2.5132553398123534, -1.9109023033210299, -0.89749807383952362,
             -0.79811198176990206, -0.38473054441488624, -0.33510868444931502,
             0.0, 0.33510868444931502, 0.38473054441488624,
             0.79811198176990206, 0.89749807383952362, 1.9109023033210299,
             2.5132553398123534, 3.1415926535897931),
     "stable": True,
    },
  ]

  @p(("strategy", "data"),
     [(strategy, data) for data in table_data
                       for strategy in data["strategies"]
     ])
  def test_block_info(self, strategy, data):
    filt = strategy(data["blk"], data["order"])
    assert almost_eq(filt, data["lpc"])
    assert almost_eq(filt.error, data["lpc_error"])
    assert almost_eq(list(parcor(filt))[::-1], data["k"])
    assert almost_eq(lsf(filt), data["lsf"])
    assert parcor_stable(1 / filt) == data["stable"]
    assert lsf_stable(1 / filt) == data["stable"]


class TestLPC(object):

  small_block = [-1, 0, 1.2, -1, -2.7, 3, 7.1, 9, 12.3]
  big_block = list((1 - 2 * z ** -1)(xrange(150), zero=0))
  block_list = [
    [1, 5, 3],
    [1, 2, 3, 3, 2, 1],
    small_block,
    TestLPCParcorLSFAndStability.block_alternate,
    big_block,
  ]
  order_list = [1, 2, 3, 7, 17, 18]
  kcovar_zdiv_error_cases = [ # tuples (blk, order)
    ([1, 5, 3], 2),
    (TestLPCParcorLSFAndStability.block_alternate, 7),
  ]
  blk_order_pairs = list(it.product(block_list, order_list))
  covars_value_error_cases = [(blk, order) for blk, order in blk_order_pairs
                                           if len(blk) <= order]
  kcovar_value_error_cases = (lambda bb, sb, ol: ( # Due to zero "k" coeffs
    [(bb, order) for order in ol if order <= 18] +
    [(sb, order) for order in ol if order <= 7]
  ))(bb=big_block, sb=small_block, ol=order_list)
  kcovar_valid_cases = (lambda ok_pairs, not_ok_pairs:
    [pair for pair in ok_pairs if pair not in not_ok_pairs]
  )(ok_pairs = blk_order_pairs,
    not_ok_pairs = kcovar_zdiv_error_cases + covars_value_error_cases +
                   kcovar_value_error_cases)

  @p(("blk", "order"), blk_order_pairs)
  def test_equalness_all_autocorrelation_strategies(self, blk, order):
    # Common case, tests whether all solutions are the same
    strategy_names = ("autocor", "nautocor", "kautocor")
    filts = [lpc[name](blk, order) for name in strategy_names]
    for f1, f2 in it.combinations(filts, 2):
      assert almost_eq(f1, f2) # But filter comparison don't include errors
      assert almost_eq(f1.error, f2.error)
      assert f1.error >= 0.
      assert f2.error >= 0.

  @p(("blk", "order"), kcovar_zdiv_error_cases)
  def test_kcovar_zdiv_error(self, blk, order):
    with pytest.raises(ZeroDivisionError):
      lpc.kcovar(blk, order)

  @p(("blk", "order"), covars_value_error_cases)
  def test_covar_kcovar_value_error_scenario(self, blk, order):
    for name in ("covar", "kcovar"):
      with pytest.raises(ValueError):
        lpc[name](blk, order)

  @p(("blk", "order"), kcovar_value_error_cases)
  def test_kcovar_value_error_scenario_invalid_coeffs(self, blk, order):
    with pytest.raises(ValueError):
      lpc.kcovar(blk, order)

    # Filter should not be stable
    filt = lpc.covar(blk, order)
    try:
      assert not parcor_stable(1 / filt)

    # See if a PARCOR is "almost one" (stability test isn't "stable")
    except AssertionError:
      assert max(xmap(abs, parcor(filt))) + 1e-7 > 1.

  @p(("blk", "order"), kcovar_valid_cases)
  def test_equalness_covar_kcovar_valid_scenario(self, blk, order):
    # Common case, tests whether all solutions are the same
    strategy_names = ("covar", "kcovar")
    filts = [lpc[name](blk, order) for name in strategy_names]
    f1, f2 = filts
    assert almost_eq(f1, f2) # But filter comparison don't include errors
    try:
      assert almost_eq(f1.error, f2.error)
    except AssertionError: # Near zero? Try again with absolute value
      max_diff = 1e-10 * min(abs(x) for x in f1.numerator + f2.numerator
                                    if x != 0)
      assert almost_eq.diff(f1.error, f2.error, max_diff=max_diff)
      assert almost_eq.diff(f1.error, 0, max_diff=max_diff)
      assert almost_eq.diff(0, f2.error, max_diff=max_diff)
    assert f1.error >= 0.
    assert f2.error >= 0.

  @p("strategy", [lpc.autocor, lpc.kautocor, lpc.nautocor])
  def test_docstring_in_all_autocor_strategies(self, strategy):
    data = [-1, 0, 1, 0] * 4
    filt = strategy(data, 2)
    assert almost_eq(filt, 1 + 0.875 * z ** -2)
    assert almost_eq.diff(filt.numerator, [1, 0., .875])
    assert almost_eq(filt.error, 1.875)


class TestParcorStableLSFStable(object):

  @p("filt", [ZFilter(1),
              1 / (1 - .5 * z ** -1),
              1 / (1 + .5 * z ** -1),
             ])
  def test_stable_filters(self, filt):
    assert parcor_stable(filt)
    assert lsf_stable(filt)

  @p("filt", [z ** -1 / (1 - z ** -1),
              1 / (1 + z ** -1),
              z ** -2 / (1 - z ** -2),
              1 / (1 - 1.2 * z ** -1),
             ])
  def test_unstable_filters(self, filt):
    assert not parcor_stable(filt)
    assert not lsf_stable(filt)


class TestParcor(object):

  filt_e4 = (1 - 0.6752 * z ** -1) * \
            (1 - 1.6077 * z ** -1 + 0.8889 * z ** -2) * \
            (1 - 1.3333 * z ** -1 + 0.8889 * z ** -2) * \
            (1 + 0.4232 * z ** -1 + 0.8217 * z ** -2) * \
            (1 + 1.6750 * z ** -1 + 0.8217 * z ** -2)

  def test_parcor_filt_e4(self):
    parcor_calculated = list(parcor(self.filt_e4))
    assert reduce(operator.mul, (1. / (1. - k ** 2)
                                 for k in parcor_calculated))
    parcor_coeff = [-0.8017212633, 0.912314348674, 0.0262174844236,
                    -0.16162324325, 0.0530245390264, 0.110480347197,
                    0.258134095686, 0.297257621307, -0.360217510101]
    assert almost_eq(parcor_calculated[::-1], parcor_coeff)
    assert parcor_stable(1 / self.filt_e4)


class TestLSF(object):

  filt_e4 = TestParcor.filt_e4

  def test_lsf_filt_e4(self):
    lsf_values_alternated = [-2.76679191844, -2.5285195589, -1.88933753141,
      -1.72283612758, -1.05267495205, -0.798045657668, -0.686406969195,
      -0.554578828901, -0.417528956381, 0.0, 0.417528956381, 0.554578828901,
      0.686406969195, 0.798045657668, 1.05267495205, 1.72283612758,
      1.88933753141, 2.5285195589, 2.76679191844, 3.14159265359]
    assert almost_eq(lsf(self.filt_e4), lsf_values_alternated)
    assert lsf_stable(1 / self.filt_e4)


class TestLevinsonDurbin(object):

  def test_one_five_three(self):
    acdata = [1, 5, 3]
    filt = levinson_durbin(acdata)
    assert almost_eq(filt, 1 - 5./12. * z ** -1 - 11./12. * z ** -2)
    err = (1 - (11./12.) ** 2) * (1 - 5 ** 2)
    assert almost_eq(filt.error, err) # Unstable filter, error is invalid
    assert almost_eq(tuple(parcor(filt)), (-11./12., -5.))
    assert not parcor_stable(1 / filt)
    assert not lsf_stable(1 / filt)


class TestToeplitz(object):

  table_schema = ("vect", "out_data")
  table_data = [
    ([18.2], [[18.2]]),
    ([-1, 19.1], [[-1, 19.1],
                  [19.1, -1]]),
    ([1, 2, 3], [[1, 2, 3],
                 [2, 1, 2],
                 [3, 2, 1]]),
  ]

  @p(table_schema, table_data)
  def test_mapping_io(self, vect, out_data):
    assert toeplitz(vect) == out_data
