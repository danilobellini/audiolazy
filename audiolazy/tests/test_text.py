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
# Created on Tue May 14 2013
# danilo [dot] bellini [at] gmail [dot] com
"""
Testing module for the lazy_text module
"""

import pytest
p = pytest.mark.parametrize

# Audiolazy internal imports
from ..lazy_text import rst_table


class TestRSTTable(object):

  simple_input = [
    [1, 2, 3, "hybrid"],
    [3, "mixed", .5, 123123]
  ]

  def test_simple_input_table(self):
    assert rst_table(
             self.simple_input,
             "this is_ a test".split()
           ) == [
             "==== ===== === ======",
             "this  is_   a   test ",
             "==== ===== === ======",
             "1    2     3   hybrid",
             "3    mixed 0.5 123123",
             "==== ===== === ======",
           ]
