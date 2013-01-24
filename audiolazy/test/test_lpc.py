# -*- coding: utf-8 -*-
"""
Testing module for the lazy_lpc module

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

Created on Wed Jan 23 2012
danilo [dot] bellini [at] gmail [dot] com
"""

import pytest
p = pytest.mark.parametrize

import itertools as it

# Audiolazy internal imports
from ..lazy_lpc import (lpc, ParCorError, #acorr, levinson_durbin,
                        parcor#, parcor_stable, lsf, lsf_stable
                        )
from ..lazy_misc import almost_eq, almost_eq_diff
from ..lazy_filters import z


class TestLPC(object):

  block_alternate = [1., 1./2., -1./8., 1./32., -1./128., 1./256., -1./512.,
                     1./1024., -1./4096., 1./8192.]
  block_list = [
    [1, 5, 3],
    [1, 2, 3, 3, 2, 1],
    [-1, 0, 1.2, -1, -2.7, 3, 7.1, 9, 12.3],
    block_alternate,
  ]
  order_list = [1, 2, 3, 7, 18]
  kcovar_parcor_error_cases = [ # tuples (blk, order)
    ([-1, 0, 1.2, -1, -2.7, 3, 7.1, 9, 12.3], 7),
    ([1, 5, 3], 2),
    (block_alternate, 7),
  ]
  blk_order_pairs = list(it.product(block_list, order_list))
  kcovar_value_error_cases = [(blk, order) for blk, order in blk_order_pairs
                                           if len(blk) <= order]
  kcovar_valid_cases = [pair for pair in blk_order_pairs
                             if pair not in kcovar_parcor_error_cases
                             and pair not in kcovar_value_error_cases]

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

  @p(("blk", "order"), kcovar_parcor_error_cases)
  def test_kcovar_parcor_error(self, blk, order):
    with pytest.raises(ParCorError):
      lpc.kcovar(blk, order)

  @p(("blk", "order"), kcovar_value_error_cases)
  def test_covar_kcovar_value_error_scenario(self, blk, order):
    for name in ("covar", "kcovar"):
      with pytest.raises(ValueError):
        lpc[name](blk, order)

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
      max_diff = 1e-14 * min(abs(x) for x in f1.numerator + f2.numerator
                                    if x != 0)
      assert almost_eq_diff(f1.error, f2.error, max_diff=max_diff)
      assert almost_eq_diff(f1.error, 0, max_diff=max_diff)
      assert almost_eq_diff(0, f2.error, max_diff=max_diff)
    assert f1.error >= 0.
    assert f2.error >= 0.

  @p("strategy", [lpc.autocor, lpc.nautocor, lpc.kautocor])
  def test_autocor_block_alternate_order_3(self, strategy):
    order = 3
    kcoeff = [-0.342081949224, 0.229319810099, -0.162014679229]
    solution = 1 - 0.457681292332 * z ** -1 \
                 + 0.297451538058 * z ** -2 \
                 - 0.162014679229 * z ** -3
    eps = 1.03182436137
    filt = strategy(self.block_alternate, order)
    assert almost_eq(filt, solution)
    assert almost_eq(filt.error, eps)
    assert almost_eq(list(parcor(filt))[::-1], kcoeff)

  @p("strategy", [lpc.covar, lpc.kcovar])
  def test_autocor_block_alternate_order_3(self, strategy):
    order = 3
    solution = 1 + 0.712617839203 * z ** -1 \
                 + 0.114426147267 * z ** -2 \
                 + 0.000614348391636 * z ** -3
    eps = 3.64963839634e-06
    filt = strategy(self.block_alternate, order)
    assert almost_eq(filt, solution)
    assert almost_eq(filt.error, eps)
