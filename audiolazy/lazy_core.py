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

__all__ = ["AbstractOperatorOverloaderMeta", "MultiKeyDict", "StrategyDict"]


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


class MultiKeyDict(dict):
  """
  Multiple keys dict. Can be thought as an "inversible" dict where you can
  ask for the one hashable value from one of the keys. By default it iterates
  through all keys, so if you need an iterator for all blocks of keys, use
  iterkeys method instead.
  """
  def __init__(self, *args, **kwargs):
    self._keys_dict = {}
    self._inv_dict = {}
    super(MultiKeyDict, self).__init__(*args, **kwargs)

  def __getitem__(self, key):
    if isinstance(key, tuple): # Avoid errors with IPython
      return super(MultiKeyDict, self).__getitem__(key)
    return super(MultiKeyDict, self).__getitem__(self._keys_dict[key])

  def __setitem__(self, key, value):
    # We want only tuples
    if not isinstance(key, tuple):
      key = (key,)

    # First remove the overwritten data
    for k in key:
      if k in self._keys_dict:
        del self[k]
    if value in self._inv_dict:
      key = self._inv_dict[value] + key

    # Remove duplicated keys
    key_list = []
    for k in key:
      if k not in key_list:
        key_list.append(k)
    key = tuple(key_list)

    # Do the assignment
    for k in key:
      self._keys_dict[k] = key
    self._inv_dict[value] = key
    super(MultiKeyDict, self).__setitem__(key, value)

  def __delitem__(self, key):
    key_tuple = self._keys_dict[key]
    value = self[key]
    new_key = tuple(k for k in key_tuple if k != key)

    # Remove the old data
    del self._keys_dict[key]
    del self._inv_dict[value]
    super(MultiKeyDict, self).__delitem__(key_tuple)

    # Do the assignment (when it makes sense)
    if len(new_key) > 0:
      for k in new_key:
        self._keys_dict[k] = new_key
      self._inv_dict[value] = new_key
      super(MultiKeyDict, self).__setitem__(new_key, value)

  def __iter__(self):
    return iter(self._inv_dict)


class StrategyDict(MultiKeyDict):
  """
  Strategy dictionary manager with default, mainly done for callables and
  multiple implementation algorithms / models.

  Each strategy might have multiple names. The names can be any hashable.
  The "strategy" method creates a decorator for the given strategy names.
  Default is the first strategy you insert, but can be changed afterwards.
  The default strategy is the attribute StrategyDict.default, and might be
  anything outside the dictionary (i.e., it won't be changed if you remove
  the strategy).

  It iterates through the values (i.e., for each strategy, not its name)

  Examples
  --------

  >>> sd = StrategyDict()
  >>> @sd.strategy("sum") # First strategy is default
  ... def sd(a, b, c):
  ...     return a + b + c
  >>> @sd.strategy("min", "m") # Multiple names
  ... def sd(a, b, c):
  ...     return min(a, b, c)
  >>> sd(2, 5, 0)
  7
  >>> sd["min"](2, 5, 0)
  0
  >>> sd["m"](7, -5, -2)
  -5
  >>> sd.default = sd["min"]
  >>> sd(-19, 1e18, 0)
  -19
  """
  default = lambda: NotImplemented

  def __init__(self, name="strategy_dict_unnamed_instance"):
    self.__name__ = name
    super(StrategyDict, self).__init__()

  def strategy(self, *names):
    def decorator(func):
      func.__name__ = names[0]
      self[names] = func
      return self
    return decorator

  def __setitem__(self, key, value):
    if "default" not in self.__dict__:
      self.default = value
    super(StrategyDict, self).__setitem__(key, value)

  def __call__(self, *args, **kwargs):
    return self.default(*args, **kwargs)

  def __getattr__(self, name):
    if name in self._keys_dict:
      return self[name]
    raise NotImplementedError("Unknown attribute '{0}'".format(name))

  def __iter__(self):
    return self.itervalues()
