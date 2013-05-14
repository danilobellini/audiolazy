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
# Created on Sat Oct 13 2012
# danilo [dot] bellini [at] gmail [dot] com
"""
Testing module for the lazy_core module
"""

import pytest
p = pytest.mark.parametrize

# Audiolazy internal imports
from ..lazy_core import AbstractOperatorOverloaderMeta, StrategyDict
from ..lazy_compat import meta


class TestAbstractOperatorOverloaderMeta(object):

  def test_empty_directly_as_metaclass(self):
    with pytest.raises(TypeError):
      try:
        class unnamed(meta(metaclass=AbstractOperatorOverloaderMeta)):
          pass
      except TypeError as excep:
        msg = "Class 'unnamed' has no builder/template for operator method '"
        assert str(excep).startswith(msg)
        raise

  def test_empty_invalid_subclass(self):
    class MyAbstractClass(AbstractOperatorOverloaderMeta):
      pass
    with pytest.raises(TypeError):
      try:
        class DummyClass(meta(metaclass=MyAbstractClass)):
          pass
      except TypeError as excep:
        msg = "Class 'DummyClass' has no builder/template for operator method"
        assert str(excep).startswith(msg)
        raise


class TestStrategyDict(object):

  def test_1x_strategy(self):
    sd = StrategyDict()

    assert len(sd) == 0

    @sd.strategy("test", "t2")
    def sd(a):
      return a + 18

    assert len(sd) == 1

    assert sd["test"](0) == 18
    assert sd.test(0) == 18
    assert sd.t2(15) == 33
    assert sd(-19) == -1
    assert sd.default == sd["test"]


  def test_same_key_twice(self):
    sd = StrategyDict()

    @sd.strategy("data", "main", "data")
    def sd():
      return True

    @sd.strategy("only", "only", "main")
    def sd():
      return False

    assert len(sd) == 2 # Strategies
    assert sd["data"] == sd.default
    assert sd["data"] != sd["main"]
    assert sd["only"] == sd["main"]
    assert sd()
    assert sd["data"]()
    assert not sd["only"]()
    assert not sd["main"]()
    assert sd.data()
    assert not sd.only()
    assert not sd.main()
    sd_keys = list(sd.keys())
    assert ("data",) in sd_keys
    assert ("only", "main") in sd_keys


  @p("add_names", [("t1", "t2"), ("t1", "t2", "t3")])
  @p("mul_names", [("t3",),
                   ("t1", "t2"),
                   ("t1", "t3"),
                   ("t3", "t1"),
                   ("t3", "t2"),
                   ("t1", "t2", "t3"),
                   ("t1")
                  ])
  def test_2x_strategy(self, add_names, mul_names):
    sd = StrategyDict()

    @sd.strategy(*add_names)
    def sd(a, b):
      return a + b

    @sd.strategy(*mul_names)
    def sd(a, b):
      return a * b

    add_names_valid = [name for name in add_names if name not in mul_names]
    if len(add_names_valid) == 0:
      assert len(sd) == 1
    else:
      assert len(sd) == 2

    for name in add_names_valid:
      assert sd[name](5, 7) == 12
      assert sd[name](1, 3) == 4
    for name in mul_names:
      assert sd[name](5, 7) == 35
      assert sd[name](1, 3) == 3

    if len(add_names_valid) > 0:
      assert sd(-19, 3) == -16
    sd.default = sd[mul_names[0]]
    assert sd(-19, 3) == -57
