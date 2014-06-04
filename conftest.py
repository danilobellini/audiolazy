# -*- coding: utf-8 -*-
# This file is part of AudioLazy, the signal processing Python package.
# Copyright (C) 2012-2014 Danilo de Jesus da Silva Bellini
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
# Created on 2014-06-04 02:19:27 BRT
# danilo [dot] bellini [at] gmail [dot] com
"""
AudioLazy testing configuration module for py.test
"""

def add_dunder_test_with_strategies(module):
  """
  Monkeypatches a ``__test__`` dictionary on the module with all strategies
  from StrategyDict instances, if any, verifying only the attributes whose
  names were declared on the module ``__all__`` list.
  """
  from audiolazy import StrategyDict
  docs = {}
  for name, attr in vars(module).items():
    if isinstance(attr, StrategyDict):
      for st in attr: # Each strategy can have a doctest
        if (st.__module__ == module.__name__ and # Avoid getting stuff from
            st.__doc__ and                       # other modules or repeated
            st.__doc__ is not type(st).__doc__):
          docs[".".join([name, st.__name__])] = st
  if docs:
    setattr(module, "__test__", docs)

def pytest_configure(config):
  """
  Called by py.test, this function is needed to ensure that doctests from
  strategies docstrings inside StrategyDict instances are collected for
  testing.
  """
  # Any import done by this function won't count for coverage afterwards, so
  # AudioLazy can't be imported here! Solution is monkeypatching the doctest
  # finding mechanism:
  import doctest, types, functools
  old_find = doctest.DocTestFinder.find
  @functools.wraps(old_find)
  def find(self, obj, *args, **kwargs):
    if isinstance(obj, types.ModuleType):
      add_dunder_test_with_strategies(obj)
    return old_find(self, obj, *args, **kwargs)
  doctest.DocTestFinder.find = find
