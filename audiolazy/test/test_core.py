#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing module for the lazy_core module

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

Created on Sat Oct 13 2012
danilo [dot] bellini [at] gmail [dot] com
"""

import pytest

# Audiolazy internal imports
from ..lazy_core import AbstractOperatorOverloaderMeta


class TestAbstractOperatorOverloaderMeta(object):

  def test_cant_be_used_directly_as_metaclass(self):
    with pytest.raises(TypeError):
      try:
        class unnamed(object):
          __metaclass__ = AbstractOperatorOverloaderMeta
      except TypeError, excep:
        assert excep.message.startswith("Can't instantiate")
        raise

  def test_subclass_without_operators_dunder(self):
    class MyAbstractClass(AbstractOperatorOverloaderMeta):
      pass
    with pytest.raises(TypeError):
      try:
        class DummyClass(object):
          __metaclass__ = MyAbstractClass
      except TypeError, excep:
        assert excep.message.startswith("Can't instantiate")
        raise
