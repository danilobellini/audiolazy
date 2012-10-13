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
  You need a concrete class inherited from this one, and the "abstract"
  enforcement and specification is:
    - You have to override __operators__ with a string, given the operators
      to be used by its dunder name "without the dunders" (i.e., "__add__"
      should be written as "add"), all in a single string separated by spaces
      and including reversed operators, like "add radd sub rsub eq".
      Its a good idea to tell all operators that will be used, since the
      metaclass will enforce their existance.
    - All operators should be implemented by the metaclass hierarchy or by
      the class directly, and the class has priority when both exists,
      neglecting the template in this case.
    - There are three templates: __binary__, __rbinary__, __unary__, all
      receives 2 parameters (the class being instantiated and the operator
      function) and should return a function for the specific dunder.
  """
  # Operator number of inputs, as a dictionary with default value equals to 2
  __operator_inputs__ = defaultdict(lambda: 2)
  for unary in "pos neg invert".split():
    __operator_inputs__[unary] = 1 # coz' we can't inspect the operator module

  def __new__(mcls, name, bases, namespace):
    cls = super(AbstractOperatorOverloaderMeta,
                mcls).__new__(mcls, name, bases, namespace)

    # Enforce __operators__ as an abstract attribute
    if getattr(mcls.__operators__, "__isabstractmethod__", False):
      msg = "Can't instantiate from '{0}' since '__operators__' is abstract"
      raise TypeError(msg.format(mcls.__name__))

    # Inserts the operators into the class
    for op_no_under in cls.__operators__.split():
      op_under = "__" + op_no_under + "__"
      if op_under not in namespace: # Added manually shouldn't use template

        # Find the operator
        op_name = op_no_under.lstrip("r")
        if op_name == "shift":
          op_name = "rshift"
        op_func = getattr(operator, "__" + op_name + "__")

        # Creates the dunder
        dunder = cls.new_dunder(op_func=op_func,
                                is_reversed=op_no_under.startswith("r") and
                                            op_no_under != "rshift",
                                ninputs=mcls.__operator_inputs__[op_no_under])

        # Abstract enforcement
        if not callable(dunder):
          msg = "Class '{0}' has no operator template for '{1}'"
          raise TypeError(msg.format(cls.__name__, op_under))

        # Inserts the dunder into the class
        dunder.__name__ = "__" + op_no_under + "__"
        setattr(cls, dunder.__name__, dunder)
    return cls

  @abstractproperty
  def __operators__(cls):
    """
    Should be overridden by a string with all operator names to be overloaded.
    Reversed operators should be given explicitly.
    """
    return ""

  def new_dunder(cls, op_func, is_reversed, ninputs):
    """
    Return a function to be used as a dunder for the operator op_func (a
    function from the operator module). If is_reversed, it means that besides
    op_func might be operator.add, what we need as return value is a __radd__
    dunder as a function. The ninputs tells us if is the dunder is
    unary(self) or binary(self, other).
    """
    return {(False, 1): cls.__unary__,
            (False, 2): cls.__binary__,
            (True, 2): cls.__rbinary__,
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
  __unary__ = __binary__ = __rbinary__ = _not_implemented
