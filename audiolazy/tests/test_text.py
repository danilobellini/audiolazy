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
Testing module for the lazy_text module
"""

from __future__ import unicode_literals

import pytest
p = pytest.mark.parametrize

# Audiolazy internal imports
from ..lazy_text import rst_table, format_docstring


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


class TestFormatDocstring(object):

  def test_template_without_parameters(self):
    docstring = "This is a docstring"
    @format_docstring(docstring)
    def not_so_useful():
      return
    assert not_so_useful.__doc__ == docstring

  def test_template_positional_parameter_automatic_counting(self):
    docstring = "Function {} docstring {}"
    @format_docstring(docstring, "Unused", "is weird!")
    def unused_func():
      return
    assert unused_func.__doc__ == "Function Unused docstring is weird!"

  def test_template_positional_parameter_numbered(self):
    docstring = "Let {3}e {1} {0} wi{3} {2}!"
    @format_docstring(docstring, "be", "force", "us", "th")
    def another_unused_func():
      return
    assert another_unused_func.__doc__ == "Let the force be with us!"

  def test_template_keyword_parameters(self):
    docstring = "{name} is a function for {explanation}"
    @format_docstring(docstring, name="Unk", explanation="uncles!")
    def unused():
      return
    assert unused.__doc__ == "Unk is a function for uncles!"

  def test_template_mixed_keywords_and_positional_params(self):
    docstring = "The {name} has to do with {0} and {1}"
    @format_docstring(docstring, "Freud", "psychoanalysis", name="ego")
    def alter():
      return
    assert alter.__doc__ == "The ego has to do with Freud and psychoanalysis"

  def test_with_docstring_in_function(self):
    dok = "This is the {a_name} docstring:{__doc__}with a {0} and a {1}."
    @format_docstring(dok, "prefix", "suffix", a_name="doc'ed")
    def dokked():
      """
      A docstring
      with two lines (but this indeed have 4 lines)
      """
    assert dokked.__doc__ == "\n".join([
      "This is the doc'ed docstring:",
      "      A docstring",
      "      with two lines (but this indeed have 4 lines)",
      "      with a prefix and a suffix.",
    ])

  def test_fill_docstring_in_function(self):
    @format_docstring(descr="dangerous")
    def danger():
      """ A {descr} doc! """
    assert danger.__doc__ == " A dangerous doc! "
