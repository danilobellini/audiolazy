#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core classes and functions module

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

Created on Mon Oct 08 2012
danilo [dot] bellini [at] gmail [dot] com
"""

import operator
from collections import defaultdict
from abc import ABCMeta, abstractproperty


class AbstractOperatorOverloaderMeta(ABCMeta):
  """
  Abstract metaclass for classes with massively overloaded operators.
  Dunders dont't appear within "getattr" nor "getattribute", and they should
  be inside the class dictionary, not the class instance one, otherwise they
  won't be found by the usual mechanism. That's why we have to be eager here.
  """
  # Operator number of inputs, as a dictionary with default value equals to 2
  __operator_inputs__ = defaultdict(lambda: 2)
  for unary in "pos neg invert".split():
    __operator_inputs__[unary] = 1 # coz' we can't inspect the operator module

  def __new__(mcs, name, bases, namespace):
    cls = ABCMeta.__new__(mcs, name, bases, namespace)
    for op_no_under in cls.__operators__.split():

      # Find the operator
      op_name = op_no_under.lstrip("r")
      if op_name == "shift":
        op_name = "rshift"
      op_func = getattr(operator, "__" + op_name + "__")

      # Creates the dunder
      dunder = cls.new_dunder(op_func=op_func,
                              is_reversed=op_no_under.startswith("r") and
                                          op_no_under != "rshift",
                              ninputs=mcs.__operator_inputs__[op_no_under])
      if dunder is not NotImplemented:
        dunder.__name__ = "__" + op_no_under + "__"
        setattr(cls, dunder.__name__, dunder)
    return cls

  @abstractproperty
  def __operators__(cls):
    """
    Should be overridden by a string with all operator names to be overloaded.
    Reversed operators should be given explicitly.
    """

  def new_dunder(cls, op_func, is_reversed, ninputs):
    """
    Return a function to be used as a dunder for the operator op_func (a
    function from the operator module). If is_reversed, it means that besides
    op_func might be operator.add, what we need as return value is a __radd__
    dunder as a function. The ninputs tells us if is the dunder is
    unary(self) or binary(self, other).
    """
    return {(False, 1): cls.unary_dunder,
            (False, 2): cls.binary_dunder,
            (True, 2): cls.reverse_binary_dunder,
           }[is_reversed, ninputs](op_func)

  # The 3 methods below should be overloaded, but they shouldn't be
  # "abstractmethod" since it's unuseful (and perhaps undesirable)
  # when there could be only one type of operator being massively overloaded.
  def _not_implemented(cls, op_func):
    """
    This method should be overridden to return the dunder for the given
    operator function.
    """
    return NotImplemented
  unary_dunder = binary_dunder = reverse_binary_dunder = _not_implemented
