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
AudioLazy testing configuration module for py.test
"""

def pytest_configure(config):
  """
  Called by py.test, this function is needed to ensure that doctests from
  strategies docstrings inside StrategyDict instances are collected for
  testing.
  """
  # Any import done by this function won't count for coverage afterwards, so
  # AudioLazy can't be imported here! Solution is monkeypatching the doctest
  # finding mechanism to import AudioLazy just there
  import doctest, types, functools
  old_find = doctest.DocTestFinder.find

  @functools.wraps(old_find)
  def find(self, obj, name=None, module=None, **kwargs):
    tests = old_find(self, obj, name=name, module=module, **kwargs)
    if not isinstance(obj, types.ModuleType):
      return tests

    # Adds the doctests from strategies inside StrategyDict instances
    from audiolazy import StrategyDict
    module_name = obj.__name__
    for name, attr in vars(obj).items(): # We know it's a module
      if isinstance(attr, StrategyDict):
        for st in attr: # Each strategy can have a doctest
          if st.__module__ == module_name: # Avoid stuff from otherwhere
            sname = ".".join([module_name, name, st.__name__])
            tests.extend(old_find(self, st, name=sname, module=obj, **kwargs))
    tests.sort()
    return tests

  doctest.DocTestFinder.find = find


try:
  import numpy
except ImportError:
  from _pytest.doctest import DoctestItem
  import pytest, re

  nn_regex = re.compile(".*#[^#]*\s*needs?\s*numpy\s*$", re.IGNORECASE)

  def pytest_runtest_setup(item):
    """
    Skip doctests that need Numpy, if it's not found. A doctest that needs
    numpy should include a doctest example that ends with a comment with
    the words "Need Numpy" (or "Needs Numpy"), no matter the case nor the
    amount of whitespaces.
    """
    if isinstance(item, DoctestItem) and \
       any(nn_regex.match(ex.source) for ex in item.dtest.examples):
      pytest.skip("Module numpy not found")
