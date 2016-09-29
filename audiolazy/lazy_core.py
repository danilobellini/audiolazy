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
Core classes module
"""

import sys
import operator
from collections import Iterable
from abc import ABCMeta
import itertools as it

# Audiolazy internal imports
from .lazy_compat import STR_TYPES, HAS_MATMUL, iteritems, itervalues

__all__ = ["OpMethod", "AbstractOperatorOverloaderMeta", "MultiKeyDict",
           "StrategyDict"]


class OpMethod(object):
  """
  Internal class to represent an operator method metadata.

  You can acess operator methods directly by using the OpMethod.get() class
  method, which always returns a generator from a query.
  This might be helpful if you need to acess the operator module from
  symbols. Given an instance "op", it has the following data:

  ========= ===========================================================
  Attribute Contents (and an example with OpMethod.get("__radd__"))
  ========= ===========================================================
  op.name   Operator name string, e.g. ``"radd"``.
  op.dname  Dunder name string, e.g. ``"__radd__"``.
  op.func   Function reference, e.g. ``operator.__add__``.
  op.symbol Operator symbol if in a code as a string, e.g. ``"+"``.
  op.rev    Boolean telling if the operator is reversed, e.g. ``True``.
  op.arity  Number of operands, e.g. ``2``.
  ========= ===========================================================

  See the ``OpMethod.get`` docstring for more information and examples.

  """
  _all = {}

  @classmethod
  def get(cls, key="all", without=None):
    """
    Returns a list with every OpMethod instance that match the key.

    The valid values for query parameters are:

    * Operator method names such as ``add`` or ``radd`` or ``pos``, with or
      without the double underscores. These would select only one operator;
    * Strings with the operator symbols such as ``"+"``, ``"&"`` or ``"**"``.
      These would select all the binary, reversed binary and unary operators
      when these apply;
    * ``"all"`` for selecting every operator available;
    * ``"r"`` gets only the reversed operators;
    * ``1`` or ``"1"`` for unary operators;
    * ``2`` or ``"2"`` for binary operators (including reversed binary);
    * ``None`` for no operators at all;
    * Operator functions from the ``operator`` module with the double
      underscores (e.g. ``operator.__add__``), for all the operations
      that use the operator function (it and the reversed);

    Parameters
    ----------
    key :
      A query value, a string with whitespace-separated query names, or an
      iterable with valid query values (as listed above). This parameter
      defaults to "all".
    without :
      The same as key, but used to tell the query something that shouldn't
      appear in the result. Defaults to None.

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
    <generator object ... at 0x...>
    >>> len(list(_)) # Found __rshift__ and __rrshift__, as a generator
    2
    >>> next(OpMethod.get("__add__")).func(2, 3) # By name, finds 2 + 3
    5
    >>> next(OpMethod.get("rsub")).symbol # Name is without underscores
    '-'
    >>> mod = list(OpMethod.get("%%"))
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
    >>> len(set(OpMethod.get(2, without=["- + *", "%%", "r"])))
    %s
    >>> len(set(OpMethod.get("all"))) # How many operator methods there are?
    %s

    """ % (15, 35) if HAS_MATMUL else (14, 33)
    ignore = set() if without is None else set(cls.get(without))
    if key is None:
      return
    if isinstance(key, STR_TYPES) or not isinstance(key, Iterable):
      key = [key]
    key = it.chain.from_iterable(el.split() if isinstance(el, STR_TYPES)
                                            else [el] for el in key)
    for op_descr in key:
      try:
        for op in cls._all[op_descr]:
          if op not in ignore:
            yield op
      except KeyError:
        if op_descr in ["div", "__div__", "rdiv", "__rdiv__"]:
          raise ValueError("Use only 'truediv' for division")
        raise ValueError("Operator '{}' unknown".format(op_descr))

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
    keys = ["all", self.symbol, self.name, self.dname, self.func,
            self.arity, str(self.arity)]
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
      >> rshift rrshift
      << lshift rlshift
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
    """.strip().splitlines()
    if HAS_MATMUL:
      op_symbols.append("@ matmul rmatmul")
    for op_line in op_symbols:
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
  >>> del mkd[-7]
  >>> len(mkd) # Again, that's the amount of values, not of keys!
  1

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

    # Remove duplicated keys (last insertion has priority)
    key_list = []
    for k in reversed(key):
      if k not in key_list:
        key_list.append(k)
    key = tuple(reversed(key_list))

    # Remove the overwritten data
    for k in key:
      if k in self._keys_dict:
        MultiKeyDict.__delitem__(self, k)

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

  def key2keys(self, key):
    """ Tuple with every key that points to the same value. """
    return self._keys_dict[key]

  def value2keys(self, value):
    """
    Tuple with every key that points to the given value.
    Result might be empty.
    """
    return self._inv_dict.get(value, tuple())


class StrategyDict(MultiKeyDict):
  """
  Strategy dictionary manager creator with default, mainly done for callables
  and multiple implementation algorithms / models.

  Each strategy might have multiple names. The names can be any hashable.
  The "strategy" method creates a decorator for the given strategy names, see
  its docstrings for more details on this.

  The default strategy is the attribute StrategyDict.default, and might be
  anything from outside the dictionary values. The default strategy is the
  first strategy you insert, unless the instance attribute already exists.

  The instances iterates through its values (i.e., for each strategy, not its
  names). You can type something like this to find all StrategyDict instances
  from the package::

  .. code-block:: python

    import audiolazy
    sorted(k for k, v in vars(audiolazy).items()
             if isinstance(v, audiolazy.StrategyDict))

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
                   "This docstring is self-generated, see the StrategyDict\n"
                   "class and the strategies docs for more details.\n"
                  )
        return "".join(doc)

    return StrategyDictInstance(name)

  default = lambda *args, **kwargs: NotImplemented

  def strategy(self, *names, **kwargs):
    """
    StrategyDict wrapping method for adding a new strategy.

    Parameters
    ----------
    *names :
      Positional arguments with all names (strings) that could be used to
      call the strategy to be added, to be used both as key items and as
      attribute names.
    keep_name :
      Boolean keyword-only parameter for choosing whether the ``__name__``
      attribute of the decorated/wrapped function should be changed or kept.
      Defaults to False (i.e., changes the name by default).

    Returns
    -------
    A decorator/wrapper function to be used once on the new strategy to be
    added.

    Example
    -------
    Let's create a StrategyDict that knows its name:

    >>> txt_proc = StrategyDict("txt_proc")

    Add a first strategy ``swapcase``, using this method as a decorator
    factory:

    >>> @txt_proc.strategy("swapcase")
    ... def txt_proc(txt):
    ...   return txt.swapcase()

    Let's do it again, but wrapping the strategy functions inline. First two
    strategies have multiple names, the last keeps the function name, which
    would otherwise be replaced by the first given name:

    >>> txt_proc.strategy("lower", "low")(lambda txt: txt.lower())
    {(...): <function ... at 0x...>, (...): <function ... at 0x...>}
    >>> txt_proc.strategy("upper", "up")(lambda txt: txt.upper())
    {...}
    >>> txt_proc.strategy("keep", keep_name=True)(lambda txt: txt)
    {...}

    We can now iterate through the strategies to call them or see their
    function names

    >>> sorted(st("Just a Test") for st in txt_proc)
    ['JUST A TEST', 'Just a Test', 'jUST A tEST', 'just a test']
    >>> sorted(st.__name__ for st in txt_proc) # Just the first name
    ['<lambda>', 'lower', 'swapcase', 'upper']

    Calling a single strategy:

    >>> txt_proc.low("TeStInG")
    'testing'
    >>> txt_proc["upper"]("TeStInG")
    'TESTING'
    >>> txt_proc("TeStInG") # Default is the first: swapcase
    'tEsTiNg'
    >>> txt_proc.default("TeStInG")
    'tEsTiNg'
    >>> txt_proc.default = txt_proc.up # Manually changing the default
    >>> txt_proc("TeStInG")
    'TESTING'

    Hint
    ----
    Default strategy is the one stored as the ``default`` attribute, you can
    change or remove it at any time. When removing all keys that are assigned
    to the default strategy, the default attribute will be removed from the
    StrategyDict instance as well. The first strategy added afterwards is the
    one that will become the new default, unless the attribute is created or
    changed manually.
    """
    def decorator(func):
      keep_name = kwargs.pop("keep_name", False)
      if kwargs:
        key = next(iter(kwargs))
        raise TypeError("Unknown keyword argument '{}'".format(key))
      if not keep_name:
        func.__name__ = str(names[0])
      self[names] = func
      return self
    return decorator

  def __setitem__(self, key, value):
    keys = key if isinstance(key, tuple) else (key,)
    for k in keys:
      try:
        del self[k] # Also remove self.default if it loses all keys
      except KeyError:
        pass # Not found!
    super(StrategyDict, self).__setitem__(keys, value)
    for k in keys:
      setattr(self, k, value)
    if "default" not in vars(self):
      self.default = value

  def __delitem__(self, key):
    keys = self.key2keys(key)
    value = self[keys]
    super(StrategyDict, self).__delitem__(key)
    if hasattr(self, key) and getattr(self, key) == value:
      super(StrategyDict, self).__delattr__(key)
    if len(keys) == 1 and value == self.default:
      super(StrategyDict, self).__delattr__("default")

  def __delattr__(self, attr):
    try:
      if self[attr] == getattr(self, attr): # Have both
        del self[attr] # Removes both
      else: # Have both but they're different
        setattr(self, attr, self[attr]) # Put the attribute back
    except KeyError: # Del a non-strategy attribute
      super(StrategyDict, self).__delattr__(attr)

  def __call__(self, *args, **kwargs):
    return self.default(*args, **kwargs)

  def __iter__(self):
    return itervalues(self)
