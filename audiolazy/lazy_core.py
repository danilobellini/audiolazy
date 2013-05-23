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
# Created on Mon Oct 08 2012
# danilo [dot] bellini [at] gmail [dot] com
"""
Core classes module
"""

import sys
import operator
from collections import Iterable
from abc import ABCMeta
import itertools as it

# Audiolazy internal imports
from .lazy_compat import STR_TYPES, iteritems, itervalues

__all__ = ["OpMethod", "AbstractOperatorOverloaderMeta", "MultiKeyDict",
           "StrategyDict"]


class OpMethod(object):
  """
  Internal class to represent an operator method metadata.

  You can acess operator methods directly by using the OpMethod.get() class
  method, which always returns a list from a query.
  This might be helpful if you need to acess the operator module from
  symbols. Given an instance "op", it has the following data:

  ========= ===========================================================
  Attribute Contents (and an example with OpMethod.get("__radd__"))
  ========= ===========================================================
  op.name   Operator name string, e.g. ``"radd"``.
  op.dname  Dunder name string, e.g. ``"__radd__"``.
  op.func   Function reference, e.g. ``operator.add``.
  op.symbol Operator symbol if in a code as a string, e.g. ``"+"``.
  op.rev    Boolean telling if the operator is reversed, e.g. ``True``.
  op.arity  Number of operands, e.g. ``2``.
  ========= ===========================================================

  See the OpMethod.get docstring for more information and examples.

  """
  _all = {}

  @classmethod
  def get(cls, key=None, without=None):
    """
    Returns a list with every OpMethod instance that match the key.

    Parameters
    ----------
    key :
      String with whitespace-separated operator names or symbols, or an
      iterable with names, symbols or functions from the operator class.
    without :
      The same of the key, but used to tell the query something that
      shouldn't appear in the result.

    Returns
    -------
    Generator with all OpMethod instances that matches the query once, keeping
    the order in which it is asked for. For a given symbol with 3 operator
    methods (e.g., "+", which yields __add__, __radd__ and __pos__), the
    yielding order is <binary>, <reversed binary> and <unary>.

    Examples
    --------
    >>> list(OpMethod.get("*")) # By symbol
    [<mul operator method ('*' symbol)>, <rmul operator method ('*' symbol)>]
    >>> OpMethod.get(">>")
    <generator object get at 0x...>
    >>> len(list(_)) # Found __rshift__ and __rrshift__, as a generator
    2
    >>> next(OpMethod.get("__add__")).func(2, 3) # By name, finds 2 + 3
    5
    >>> next(OpMethod.get("rsub")).symbol # Name is without underscores
    '-'
    >>> mod = list(OpMethod.get("%"))
    >>> mod[0].rev # Is it reversed? The __mod__ isn't.
    False
    >>> mod[1].rev # But the __rmod__ is!
    True
    >>> mod[1].arity # Number of operands, the __rmod__ is binary
    2
    >>> add = list(OpMethod.get("+"))
    >>> add[2].arity # Unary "+"
    1
    >>> add[2] is next(OpMethod.get("pos"))
    True
    >>> import operator
    >>> next(OpMethod.get(operator.add)).symbol # Using the operator function
    '+'
    >>> len(list(OpMethod.get(operator.add))) # __add__ and __radd__
    2
    >>> len(list(OpMethod.get("<< >>"))) # Multiple inputs
    4
    >>> len(list(OpMethod.get("<< >>", without="r"))) # Without reversed
    2
    >>> list(OpMethod.get(["+", "&"], without=[operator.add, "r"]))
    [<pos operator method ('+' symbol)>, <and operator method ('&' symbol)>]
    >>> len(set(OpMethod.get(2, without=["- + *", "%", "r"])))
    14
    >>> len(set(OpMethod.get("all"))) # How many operator methods there are?
    33

    """
    if key is None:
      return
    ignore = set() if without is None else set(cls.get(without))
    if isinstance(key, STR_TYPES) or not isinstance(key, Iterable):
      key = [key]
    key = it.chain(*[el.split() if isinstance(el, STR_TYPES) else [el]
                     for el in key])
    for op_descr in key:
      try:
        for op in cls._all[op_descr]:
          if op not in ignore:
            yield op
      except KeyError:
        if op_descr in ["div", "__div__", "rdiv", "__rdiv__"]:
          raise ValueError("Use only 'truediv' for division")
        raise ValueError("Operator '{}' not found".format(op_descr))

  @classmethod
  def _insert(cls, name, symbol):
    self = cls()
    self.name = name
    self.symbol = symbol
    self.rev = name.startswith("r") and name != "rshift"
    self.dname = "__{}__".format(name) # Dunder name
    self.arity = 1 if name in ["pos", "neg", "invert"] else 2
    self.func = getattr(operator, "__{}__".format(name[self.rev:]))

    # Updates the "all" dictionary
    keys = ["all", self.symbol, self.name, self.dname, self.func, self.arity]
    if self.rev:
      keys.append("r")
    for key in keys:
      if key not in cls._all:
        cls._all[key] = [self]
      else:
        cls._all[key].append(self)

  @classmethod
  def _initialize(cls):
    """
    Internal method to initialize the class by creating all
    the operator metadata to be used afterwards.
    """
    op_symbols = """
      + add radd pos
      - sub rsub neg
      * mul rmul
      / truediv rtruediv
      // floordiv rfloordiv
      % mod rmod
      ** pow rpow
      >> lshift rlshift
      << rshift rlshift
      ~ invert
      & and rand
      | or ror
      ^ xor rxor
      < lt
      <= le
      == eq
      != ne
      > gt
      >= ge
    """
    for op_line in op_symbols.strip().splitlines():
      symbol, names = op_line.split(None, 1)
      for name in names.split():
        cls._insert(name, symbol)

  def __repr__(self):
    return "<{} operator method ('{}' symbol)>".format(self.name, self.symbol)


# Creates all operators
OpMethod._initialize()


class AbstractOperatorOverloaderMeta(ABCMeta):
  """
  Abstract metaclass for classes with massively overloaded operators.

  Dunders dont't appear within "getattr" nor "getattribute", and they should
  be inside the class dictionary, not the class instance one, otherwise they
  won't be found by the usual mechanism. That's why we have to be eager here.
  You need a concrete class inherited from this one, and the "abstract"
  enforcement and specification is:

  - Override __operators__ and __without__ with a ``OpMethod.get()`` valid
    query inputs, see that method docstring for more information and examples.
    Its a good idea to tell all operators that will be used, including the
    ones that should be defined in the instance namespace, since the
    metaclass will enforce their existance without overwriting.

    These should be overridden by a string or a list with all operator names,
    symbols or operator functions (from the `operator` module) to be
    overloaded (or neglect, in the __without__).

    - When using names, reversed operators should be given explicitly.
    - When using symbols the reversed operators and the unary are implicit.
    - When using operator functions, the ooperators and the unary are
      implicit.

    By default, __operators__ is "all" and __without__ is None.

  - All operators should be implemented by the metaclass hierarchy or by
    the class directly, and the class has priority when both exists,
    neglecting the method builder in this case.

  - There are three method builders which should be written in the concrete
    metaclass: ``__binary__``, ``__rbinary__`` and ``__unary__``.
    All receives 2 parameters (the class being instantiated and a OpMethod
    instance) and should return a function for the specific dunder, probably
    doing so based on general-use templates.

  Note
  ----
  Don't use "div"! In Python 2.x it'll be a copy of truediv.

  """
  __operators__ = "all"
  __without__ = None

  def __new__(mcls, name, bases, namespace):
    cls = super(AbstractOperatorOverloaderMeta,
                mcls).__new__(mcls, name, bases, namespace)

    # Inserts each operator into the class
    for op in OpMethod.get(mcls.__operators__, without=mcls.__without__):
      if op.dname not in namespace: # Added manually shouldn't use template

        # Creates the dunder method
        dunder = {(False, 1): mcls.__unary__,
                  (False, 2): mcls.__binary__,
                  (True, 2): mcls.__rbinary__,
                 }[op.rev, op.arity](cls, op)

        # Abstract enforcement
        if not callable(dunder):
          msg = "Class '{}' has no builder/template for operator method '{}'"
          raise TypeError(msg.format(cls.__name__, op.dname))

        # Inserts the dunder into the class
        dunder.__name__ = op.dname
        setattr(cls, dunder.__name__, dunder)
      else:
        dunder = namespace[op.dname]

      if sys.version_info.major == 2 and op.name in ["truediv", "rtruediv"]:
        new_name = op.dname.replace("true", "")
        if new_name not in namespace: # If wasn't insert manually
          setattr(cls, new_name, dunder)

    return cls

  # The 3 methods below should be overloaded, but they shouldn't be
  # "abstractmethod" since it's unuseful (and perhaps undesirable)
  # when there could be only one type of operator being massively overloaded.
  def __binary__(cls, op):
    """
    This method should be overridden to return the dunder for the given
    operator function.

    """
    return NotImplemented
  __unary__ = __rbinary__ = __binary__


class MultiKeyDict(dict):
  """
  Multiple keys dict.

  Can be thought as an "inversible" dict where you can ask for the one
  hashable value from one of the keys. By default it iterates through the
  values, if you need an iterator for all tuples of keys,
  use iterkeys method instead.

  Examples
  --------
  Assignments one by one:

  >>> mk = MultiKeyDict()
  >>> mk[1] = 3
  >>> mk[2] = 3
  >>> mk
  {(1, 2): 3}
  >>> mk[4] = 2
  >>> mk[1] = 2
  >>> len(mk)
  2
  >>> mk[1]
  2
  >>> mk[2]
  3
  >>> mk[4]
  2
  >>> sorted(mk)
  [2, 3]
  >>> sorted(mk.keys())
  [(2,), (4, 1)]

  Casting from another dict:

  >>> mkd = MultiKeyDict({1:4, 2:5, -7:4})
  >>> len(mkd)
  2
  >>> sorted(mkd)
  [4, 5]
  >>> del mkd[2]
  >>> len(mkd)
  1
  >>> sorted(list(mkd.keys())[0]) # Sorts the only key tuple
  [-7, 1]

  """
  def __init__(self, *args, **kwargs):
    self._keys_dict = {}
    self._inv_dict = {}
    super(MultiKeyDict, self).__init__()
    for key, value in iteritems(dict(*args, **kwargs)):
      self[key] = value

  def __getitem__(self, key):
    if isinstance(key, tuple): # Avoid errors with IPython
      return super(MultiKeyDict, self).__getitem__(key)
    return super(MultiKeyDict, self).__getitem__(self._keys_dict[key])

  def __setitem__(self, key, value):
    # We want only tuples
    if not isinstance(key, tuple):
      key = (key,)

    # Finds the full new tuple keys
    if value in self._inv_dict:
      key = self._inv_dict[value] + key

    # Remove duplicated keys
    key_list = []
    for k in key:
      if k not in key_list:
        key_list.append(k)
    key = tuple(key_list)

    # Remove the overwritten data
    for k in key:
      if k in self._keys_dict:
        del self[k]

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
  Strategy dictionary manager creator with default, mainly done for callables
  and multiple implementation algorithms / models.

  Each strategy might have multiple names. The names can be any hashable.
  The "strategy" method creates a decorator for the given strategy names.
  Default is the first strategy you insert, but can be changed afterwards.
  The default strategy is the attribute StrategyDict.default, and might be
  anything outside the dictionary (i.e., it won't be changed if you remove
  the strategy).

  It iterates through the values (i.e., for each strategy, not its name).

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

  Note
  ----
  The StrategyDict constructor creates a new class inheriting from
  StrategyDict, and then instantiates it before returning the requested
  instance. This singleton subclassing is needed for docstring
  personalization.

  Warning
  -------
  Every strategy you insert have as a side-effect a change into its module
  ``__test__`` dictionary, to allow the doctests finder locate your
  strategies. Make sure your strategy ``__module__`` attribute is always
  right. Set it to ``None`` (or anything that evaluates to ``False``) if
  you don't want this behaviour.

  """
  def __new__(self, name="strategy_dict_unnamed_instance"):
    """
    Creates a new StrategyDict class and returns an instance of it.
    The new class is needed to ensure it'll have a personalized docstring.

    """
    class StrategyDictInstance(StrategyDict):

      def __new__(cls, name=name):
        del StrategyDictInstance.__new__ # Should be called only once
        return MultiKeyDict.__new__(StrategyDictInstance)

      def __init__(self, name=name):
        self.__name__ = name
        super(StrategyDict, self).__init__()

      @property
      def __doc__(self):
        from .lazy_text import small_doc
        docbase = "This is a StrategyDict instance object called\n" \
                  "``{0}``. Strategies stored: {1}.\n"
        doc = [docbase.format(self.__name__, len(self))]

        pairs = sorted(iteritems(self))
        if self.default not in list(self.values()):
          pairs = it.chain(pairs, [(tuple(), self.default)])

        for key_tuple, value in pairs:
          # First find the part of the docstring related to the keys
          strategies = ["{0}.{1}".format(self.__name__, name)
                        for name in key_tuple]
          if len(strategies) == 0:
            doc.extend("\nDefault unnamed strategy")
          else:
            if value == self.default:
              strategies[0] += " (Default)"
            doc.extend(["\n**Strategy ", strategies[0], "**.\n"]),
            if len(strategies) == 2:
              doc.extend(["An alias for it is ``", strategies[1], "``.\n"])
            elif len(strategies) > 2:
              doc.extend(["Aliases available are ``",
                          "``, ``".join(strategies[1:]), "``.\n"])

          # Get first description paragraph as the docstring related to value
          doc.append("Docstring starts with:\n")
          doc.extend(small_doc(value, indent="\n  "))
          doc.append("\n")

        doc.append("\nNote"
                   "\n----\n"
                   "StrategyDict instances like this one have lazy\n"
                   "self-generated docstrings. If you change something in\n"
                   "the dict, the next docstrings will follow the change.\n"
                   "Calling this instance directly will have the same\n"
                   "effect as calling the default strategy.\n"
                   "You can see the full strategies docstrings for more\n"
                   "details, as well as the StrategyDict class\n"
                   "documentation.\n"
                  )
        return "".join(doc)

    return StrategyDictInstance(name)

  default = lambda: NotImplemented

  def strategy(self, *names):
    def decorator(func):
      func.__name__ = str(names[0])
      self[names] = func
      return self
    return decorator

  def __setitem__(self, key, value):
    if "default" not in self.__dict__: # Avoiding hasattr due to __getattr__
      self.default = value
    super(StrategyDict, self).__setitem__(key, value)

    # Also register strategy into module __test__ (allow doctests)
    if "__doc__" in getattr(value, "__dict__", {}):
      module_name = getattr(value, "__module__", False)
      if module_name:
        module = sys.modules[module_name]
        if not hasattr(module, "__test__"):
          setattr(module, "__test__", {})
        strategy_name = ".".join([self.__name__, value.__name__])
        module.__test__[strategy_name] = value

  def __call__(self, *args, **kwargs):
    return self.default(*args, **kwargs)

  def __getattr__(self, name):
    if name in self._keys_dict:
      return self[name]
    raise AttributeError("Unknown attribute '{0}'".format(name))

  def __iter__(self):
    return itervalues(self)
