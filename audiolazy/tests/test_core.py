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
Testing module for the lazy_core module
"""

import pytest
p = pytest.mark.parametrize

from types import GeneratorType
import operator
from functools import reduce

# Audiolazy internal imports
from ..lazy_core import (OpMethod, AbstractOperatorOverloaderMeta,
                         MultiKeyDict, StrategyDict)
from ..lazy_compat import meta

if hasattr(operator, "matmul"):
  HAS_REV = sorted("+ - * @ / // % ** >> << & | ^".split())
else:
  HAS_REV = sorted("+ - * / // % ** >> << & | ^".split())


class TestOpMethod(object):

  def test_get_no_input(self):
    assert set(OpMethod.get()) == set(OpMethod.get("all"))

  def test_get_empty(self):
    for el in [OpMethod.get(None), OpMethod.get(None, without="+"),
               OpMethod.get(without="all"), OpMethod.get("+ -", "- +")]:
      assert isinstance(el, GeneratorType)
      assert list(el) == []

  @p("name", ["cmp", "almosteq", "div"])
  def test_get_wrong_input(self, name):
    for el in [OpMethod.get(name),
               OpMethod.get(without=name),
               OpMethod.get("all", without=name),
               OpMethod.get(name, without="all")]:
      with pytest.raises(ValueError) as exc:
        next(el)
      assert name in str(exc.value)
      assert ("unknown" in str(exc.value)) is (name != "div")

  def test_get_reversed(self):
    result_all_rev = list(OpMethod.get("r"))
    result_no_rev = list(OpMethod.get("all", without="r"))
    result_all = list(OpMethod.get("all"))
    assert len(result_all_rev) == len(HAS_REV)
    assert HAS_REV == sorted(op.symbol for op in result_all_rev)
    assert not any(el in HAS_REV for el in result_no_rev)
    assert set(result_no_rev).union(set(result_all_rev)) == set(result_all)

  @p(("symbol", "name"), {"+": "add",
                          "-": "sub",
                          "*": "mul",
                          "@": "matmul",
                          "/": "truediv",
                          "//": "floordiv",
                          "%": "mod",
                          "**": "pow",
                          ">>": "rshift",
                          "<<": "lshift",
                          "~": "invert",
                          "&": "and",
                          "|": "or",
                          "^": "xor",
                          "<": "lt",
                          "<=": "le",
                          "==": "eq",
                          "!=": "ne",
                          ">": "gt",
                          ">=": "ge"}.items())
  def test_get_by_all_criteria_for_one_symbol(self, symbol, name):
    if symbol == "@" and not hasattr(operator, "matmul"):
      pytest.skip("Matrix multiplication operator '@' requires Python 3.5+")

    # Useful constants
    third_name = {"+": "pos",
                  "-": "neg"}
    without_binary = ["~"]

    # Search by symbol
    result = list(OpMethod.get(symbol))

    # Dunder name
    for res in result:
      assert res.dname == res.name.join(["__", "__"])

    # Name, arity, reversed and function
    assert result[0].name == name
    assert result[0].arity == (1 if symbol in without_binary else 2)
    assert not result[0].rev
    func = getattr(operator, name.join(["__", "__"]))
    assert result[0].func is func
    if symbol in HAS_REV:
      assert result[1].name == "r" + name
      assert result[1].arity == 2
      assert result[1].rev
      assert result[0].func is result[1].func
    if symbol in third_name:
      assert result[2].name == third_name[symbol]
      assert result[2].arity == 1
      assert not result[2].rev
      unary_func = getattr(operator, third_name[symbol].join(["__", "__"]))
      assert result[2].func is unary_func
      assert not (result[0].func is result[2].func)

    # Length
    if symbol in third_name:
      assert len(result) == 3
    elif symbol in HAS_REV:
      assert len(result) == 2
    else:
      assert len(result) == 1

    # Search by name
    result_name = list(OpMethod.get(name))
    assert len(result_name) == 1
    assert result_name[0] is result[0]

    # Search by dunder name
    result_dname = list(OpMethod.get(name.join(["__", "__"])))
    assert len(result_dname) == 1
    assert result_dname[0] is result[0]

    # Search by function
    result_func = list(OpMethod.get(func))
    assert len(result_func) == min(2, len(result))
    assert result_func[0] is result[0]
    assert result_func[-1] is result[:2][-1]
    if symbol in third_name:
      result_unary_func = list(OpMethod.get(unary_func))
      assert len(result_unary_func) == 1
      assert result_unary_func[0] is result[2]

  def test_get_by_arity(self):
    comparison_symbols = "> >= == != < <=" # None is "reversed" here

    # Queries to be used
    res_unary = set(OpMethod.get("1"))
    res_binary = set(OpMethod.get("2"))
    res_reversed = set(OpMethod.get("r"))
    res_not_unary = set(OpMethod.get(without="1"))
    res_not_binary = set(OpMethod.get(without="2"))
    res_not_reversed = set(OpMethod.get(without="r"))
    res_not_reversed_nor_unary = set(OpMethod.get(without="r 1"))
    res_all = set(OpMethod.get("all"))
    res_comparison = set(OpMethod.get(comparison_symbols))

    # Compare!
    assert len(res_unary) == 3
    assert set(op.name for op in res_unary) == {"pos", "neg", "invert"}
    assert len(res_binary) == 2 * len(res_reversed) + len(res_comparison)
    assert all(op in res_binary for op in res_reversed)
    assert all(op in res_binary for op in res_not_reversed_nor_unary)
    assert all(op in res_binary for op in res_comparison)
    assert all(op in res_not_reversed_nor_unary for op in res_comparison)
    assert all(op in res_not_reversed for op in res_not_reversed_nor_unary)
    assert all(op in res_not_reversed for op in res_unary)
    assert all((op in res_reversed) or (op in res_not_reversed_nor_unary)
               for op in res_binary)
    assert all(op in res_binary for op in res_not_reversed_nor_unary)

    # Excluded middle: an operator is always either unary or binary
    assert len(res_all) == len(res_unary) + len(res_binary)
    assert not any(op in res_binary for op in res_unary)
    assert not any(op in res_unary for op in res_binary)
    assert res_not_unary == res_binary
    assert res_not_binary == res_unary

    # Query using other datatypes
    assert res_unary == set(OpMethod.get(1))
    assert res_binary == set(OpMethod.get(2))
    assert res_not_reversed_nor_unary == \
           set(OpMethod.get(without=["r", 1])) == \
           set(OpMethod.get(without=["r", "1"]))

  def test_mixed_format_query(self):
    a = set(OpMethod.get(["+", "invert", "sub rsub >"], without="radd"))
    b = set(OpMethod.get(["+ invert", "sub rsub >"], without="radd"))
    c = set(OpMethod.get(["add invert", "sub rsub >", operator.__pos__]))
    d = set(OpMethod.get("add invert pos sub rsub >"))
    e = set(OpMethod.get(["+ -", operator.__invert__, "__gt__"],
                         without="__radd__ neg"))
    assert a == b == c == d == e


class TestAbstractOperatorOverloaderMeta(object):

  def test_empty_directly_as_metaclass(self):
    with pytest.raises(TypeError):
      try:
        class unnamed(meta(metaclass=AbstractOperatorOverloaderMeta)):
          pass
      except TypeError as excep:
        msg = "Class 'unnamed' has no builder/template for operator method '"
        assert str(excep).startswith(msg)
        raise

  def test_empty_invalid_subclass(self):
    class MyAbstractClass(AbstractOperatorOverloaderMeta):
      pass
    with pytest.raises(TypeError):
      try:
        class DummyClass(meta(metaclass=MyAbstractClass)):
          pass
      except TypeError as excep:
        msg = "Class 'DummyClass' has no builder/template for operator method"
        assert str(excep).startswith(msg)
        raise


class TestMultiKeyDict(object):

  def test_key2keys_value2keys(self):
    md = MultiKeyDict()
    md[1] = 7
    md[2] = 7
    md[3] = -4
    md[4] = -4
    md[-4] = 1
    assert md.key2keys(1) == md.key2keys(2) == (1, 2) == md.value2keys(7)
    assert md.key2keys(3) == md.key2keys(4) == (3, 4) == md.value2keys(-4)
    assert md.key2keys(-4) == (-4,) == md.value2keys(1)
    assert len(md) == 3

    del md[2]
    assert md.key2keys(1) == (1,) == md.value2keys(7)
    assert md.key2keys(3) == md.key2keys(4) == (3, 4) == md.value2keys(-4)
    assert md.key2keys(-4) == (-4,) == md.value2keys(1)
    assert len(md) == 3
    with pytest.raises(KeyError):
      md.key2keys(2)

    del md[1]
    assert md.value2keys(7) == tuple()
    assert md.key2keys(3) == md.key2keys(4) == (3, 4) == md.value2keys(-4)
    assert md.key2keys(-4) == (-4,) == md.value2keys(1)
    assert len(md) == 2
    with pytest.raises(KeyError):
      md.key2keys(1)
    with pytest.raises(KeyError):
      md.key2keys(2)

  def test_insertion_order(self):
    md = MultiKeyDict()
    md[1] = 19
    md[2] = 19
    assert list(md.keys()) == [(1, 2)]
    md[1] = 19 # Re-assign the one
    assert list(md.keys()) == [(2, 1)] # Keeps insertion order
    md[3] = 19 # New key
    assert list(md.keys()) == [(2, 1, 3)]
    del md[1]
    assert list(md.keys()) == [(2, 3)]
    md[1] = 19 # Now one is a new key
    assert list(md.keys()) == [(2, 3, 1)]


class TestStrategyDict(object):

  def test_1x_strategy(self):
    sd = StrategyDict()

    assert len(sd) == 0

    @sd.strategy("test", "t2")
    def sd(a):
      return a + 18

    assert len(sd) == 1

    assert sd["test"](0) == 18
    assert sd.test(0) == 18
    assert sd.t2(15) == 33
    assert sd(-19) == -1
    assert sd.default == sd["test"]


  def test_same_key_twice(self):
    sd = StrategyDict()

    @sd.strategy("data", "main", "data")
    def sd():
      return True

    @sd.strategy("only", "only", "main")
    def sd():
      return False

    assert len(sd) == 2 # Strategies
    assert sd["data"] == sd.default
    assert sd["data"] != sd["main"]
    assert sd["only"] == sd["main"]
    assert sd()
    assert sd["data"]()
    assert not sd["only"]()
    assert not sd["main"]()
    assert sd.data()
    assert not sd.only()
    assert not sd.main()
    sd_keys = list(sd.keys())
    assert ("data",) in sd_keys
    assert ("only", "main") in sd_keys


  @p("add_names", [("t1", "t2"), ("t1", "t2", "t3")])
  @p("mul_names", [("t3",),
                   ("t1", "t2"),
                   ("t1", "t3"),
                   ("t3", "t1"),
                   ("t3", "t2"),
                   ("t1", "t2", "t3"),
                   ("t1")
                  ])
  def test_2x_strategy(self, add_names, mul_names):
    sd = StrategyDict()

    @sd.strategy(*add_names)
    def sd(a, b):
      return a + b

    @sd.strategy(*mul_names)
    def sd(a, b):
      return a * b

    add_names_valid = [name for name in add_names if name not in mul_names]
    if len(add_names_valid) == 0:
      assert len(sd) == 1
    else:
      assert len(sd) == 2

    for name in add_names_valid:
      assert sd[name](5, 7) == 12
      assert sd[name](1, 3) == 4
    for name in mul_names:
      assert sd[name](5, 7) == 35
      assert sd[name](1, 3) == 3

    if len(add_names_valid) > 0:
      assert sd(-19, 3) == -16
    sd.default = sd[mul_names[0]]
    assert sd(-19, 3) == -57

  def test_strategies_names_introspection(self):
    sd = StrategyDict()
    sd.strategy("first", "abc")(lambda val: "abc" + val)
    sd.strategy("second", "def")(lambda val: "def" + val) # Neglect 2nd name
    sd.strategy("third", "123")(lambda val: "123" + val) # Neglect 2nd name

    # Nothing new here: strategies do what they should...
    assert sd("x") == "abcx"
    assert sd.default("p") == "abcp"

    assert sd.first("w") == "abcw" == sd["first"]("w")
    assert sd.second("zsc") == "defzsc" == sd["second"]("zsc")
    assert sd.third("blah") == "123blah" == sd["third"]("blah")

    assert sd.abc("y") == "abcy" == sd["abc"]("y")
    assert sd["def"]("few") == "deffew"
    assert sd["123"]("lots") == "123lots"

    # Valid names for attributes
    all_names = {"first", "second", "third", "abc", "def", "123"}
    assert all(name in dir(sd) for name in all_names)
    assert all(name in vars(sd) for name in all_names)
    assert "default" in dir(sd)
    assert "default" in vars(sd)
    all_keys_tuples = sd.keys()
    all_keys = reduce(operator.concat, all_keys_tuples)
    assert set(all_keys) == all_names # Default not in keys
    assert set(all_keys_tuples) == {("first", "abc"),
                                    ("second", "def"),
                                    ("third", "123")}

    # First name is the __name__
    assert sd["abc"].__name__ == "first"
    assert sd["def"].__name__ == "second"
    assert sd["123"].__name__ == "third"

  def test_empty(self):
    sd = StrategyDict() # No strategy implemented!
    assert "default" in dir(sd)
    assert "default" not in vars(sd) # Only in the class
    assert sd.default() == NotImplemented
    assert sd() == NotImplemented
    assert sd.default(a_key_param="Something") == NotImplemented
    assert sd(some_key_param="Anything") == NotImplemented
    assert sd.default(12) == NotImplemented
    assert sd(34) == NotImplemented
    assert list(sd.keys()) == []
    assert list(iter(sd)) == []

  @p("is_delitem", [True, False])
  def test_delitem_delattr(self, is_delitem):
    sd = StrategyDict()
    sd.strategy("sum")(lambda *args: reduce(operator.add, args))
    sd.strategy("prod")(lambda *args: reduce(operator.mul, args))

    # They work...
    assert sd.sum(7, 2, 3) == 12 == sd(7, 2, 3) == sd.default(7, 2, 3)
    assert sd.prod(7, 2, 3) == 42
    assert sd["sum"](2, 3) == 5 == sd(2, 3) == sd.default(2, 3)
    assert sd["prod"](2, 3) == 6
    with pytest.raises(KeyError): # Default isn't an item
      sd["default"](5, 4)

    # Their names are there
    assert set(sd.keys()) == {("sum",), ("prod",)}
    assert "sum" in dir(sd)
    assert "prod" in dir(sd)
    assert "sum" in vars(sd)
    assert "prod" in vars(sd)
    assert "default" in dir(sd)
    assert "default" in vars(sd)

    # Not anymore!
    if is_delitem:
      del sd["sum"]
    else:
      del sd.sum
    assert "sum" not in dir(sd)
    assert "sum" not in vars(sd)
    assert "default" in dir(sd)
    assert "default" not in vars(sd)
    with pytest.raises(AttributeError):
      sd.sum(-1, 2, 3)
    with pytest.raises(KeyError):
      sd["sum"](5, 4)
    with pytest.raises(KeyError): # About this one, nothing changed
      sd["default"](5, 4)

    # But prod is still there
    assert list(sd.keys()) == [("prod",)]
    assert len(sd) == 1
    assert "prod" in dir(sd)
    assert "prod" in vars(sd)
    assert sd.prod(-1, 2, 3) == -6
    assert sd["prod"](5, 4) == 20

    # And now there's no default strategy
    assert sd(3, 2) == NotImplemented == sd.default(3, 2)

  def test_strategy_keep_name(self):
    sd = StrategyDict("sd")
    func = lambda a, b: a + b
    assert func.__name__ == "<lambda>"
    sd.strategy("add", keep_name=True)(func)
    assert func.__name__ == "<lambda>"
    sd.strategy("+", keep_name=False)(func)
    assert func.__name__ == "+"
    assert list(sd.keys()) == [("add", "+")]
    sd.strategy("add")(func) # Keeping the name is False by default
    assert func.__name__ == "add"
    assert list(sd.keys()) == [("+", "add")] # Insertion order

  def test_strategy_invalid_kwarg(self):
    sd = StrategyDict("sd")
    identity = lambda x: x
    with pytest.raises(TypeError) as exc:
      sd.strategy("add", weird=True)(identity)
    words = ["unknown", "weird"]
    assert all(w in str(exc.value).lower() for w in words)
    assert len(sd) == 0 # Don't add the strategy
    assert identity.__name__ == "<lambda>" # Name is kept
    assert "default" not in vars(sd)
    assert sd("anything") == NotImplemented

  def test_strategy_attribute_replaced(self):
    sd = StrategyDict("sd")
    sd.strategy("add", "+", keep_name=True)(operator.add)
    sd.strategy("mul", "*", keep_name=True)(operator.mul)
    sd.strategy("sub", "-", keep_name=True)(operator.sub)

    # Replaces the strategy attribute, but keeps the strategy there
    sd.sub = 14
    assert set(sd.keys()) == {("add", "+"), ("mul", "*"), ("sub", "-",)}
    assert sd["sub"](5, 4) == 1
    assert sd.sub == 14

    # Removes the strategy with replaced attribute, keeping the attribute
    del sd["sub"] # Removes the strategy
    assert sd.sub == 14 # Still there
    assert set(sd.keys()) == {("add", "+"), ("mul", "*"), ("-",)}

    # Removes the replaced attribute, keeping the strategy
    sd.add = None
    assert sd.add is None
    assert sd["add"](3, 7) == 10
    del sd.add # Removes the attribute
    assert sd.add(4, 7) == 11 == sd["add"](4, 7)
    assert set(sd.keys()) == {("add", "+"), ("mul", "*"), ("-",)}

    # Removes the strategy whose attribute had been replaced
    del sd.add # Removes the strategy
    assert set(sd.keys()) == {("+",), ("mul", "*"), ("-",)}
    with pytest.raises(KeyError):
      sd["add"](5, 4)
    with pytest.raises(AttributeError):
      sd.add(5, 4)

  def test_non_strategy_delattr(self):
    sd = StrategyDict("sd")
    sd.strategy("add", "+", keep_name=True)(operator.add)

    sd.another = 15
    assert sd.another == 15
    assert list(sd.keys()) == [("add", "+")]
    with pytest.raises(KeyError):
      del sd["another"]

    sd.another = lambda x: x # Replaces it
    assert sd.another([2, 3, 7]) == [2, 3, 7]
    with pytest.raises(KeyError):
      del sd["another"]

    del sd.another
    assert not hasattr(sd, "another")
    with pytest.raises(AttributeError):
      del sd.another

  def test_replacing_default(self):
    sd = StrategyDict("sd")
    sd.strategy("add", "+", keep_name=True)(operator.add)
    sd.strategy("sub", "-", keep_name=True)(operator.sub)

    assert sd(2, 4) == 6
    sd.default = sd.sub
    assert sd(2, 4) == -2
    del sd.sub
    assert sd(3, 4) == -1
    del sd["-"]
    assert sd(7, -3) == NotImplemented

    sd.default = lambda *args: None
    sd.strategy("pow", keep_name=True)(operator.pow)
    assert sd(2, 3) is None
    del sd.default
    assert sd(2, 3) == NotImplemented
    del sd.pow

    sd.strategy("mul", keep_name=True)(operator.mul)
    assert sd(7, -3) == -21

    sd.default = lambda *args, **kwargs: 42
    assert sd(7, -3) == 42

    del sd.mul
    assert len(sd) == 1 # Only the add strategy is kept (with 2 keys)
    assert sd(1) == 42

    del sd.add
    del sd["+"]
    assert len(sd) == 0
    assert sd(3, 2, 1) == 42 # Now default strategy isn't an item

    sd.strategy("blah")(sd.default) # Weird way to delete the default
    del sd.blah
    assert sd("hua hua hua") == NotImplemented

  def test_add_strategy_with_setitem(self):
    sdict = StrategyDict("sdict")
    sdict["add"] = operator.add
    sdict["mul"] = operator.mul
    sdict["+"] = operator.add

    assert len(sdict) == 2
    assert set(sdict.keys()) == {("add", "+"), ("mul",)}
    assert all(name in dir(sdict) for name in {"add", "+", "mul"})
    assert all(name in vars(sdict) for name in {"add", "+", "mul"})

    assert sdict.add(2, 3) == 5 == sdict["add"](2, 3)
    assert sdict.mul(2, 3) == 6 == sdict["mul"](2, 3)
    assert sdict(7, 8) == 15 == sdict.default(7, 8)

    del sdict["+"]
    assert len(sdict) == 2
    del sdict.add
    assert len(sdict) == 1
    assert sdict(7, 8) == NotImplemented == sdict.default(7, 8)

    sdict["pow"] = operator.pow
    assert len(sdict) == 2
    assert sdict(2, 3) == 8 == sdict.default(2, 3)
    assert sdict.pow(5, 2) == 25 == sdict["pow"](5, 2)

  @p("use_setitem", [True, False])
  def test_reusing_strategy_name(self, use_setitem):
    sdict = StrategyDict("sdict")
    m1 = lambda el: el - 1
    p1 = lambda el: el + 1
    if use_setitem:
      sdict["minus_one", "m1"] = m1
      sdict["plus_one", "p1"] = p1
    else:
      sdict.strategy("minus_one", "m1")(m1)
      sdict.strategy("plus_one", "p1")(p1)

    assert len(sdict) == 2
    assert set(sdict.keys()) == {("minus_one", "m1"), ("plus_one", "p1")}
    names = {"minus_one", "m1", "plus_one", "p1"}
    assert all(name in dir(sdict) for name in names)
    assert all(name in vars(sdict) for name in names)
    assert "default" in vars(sdict)
    assert sdict.default == m1

    assert sdict.m1(2) == 1 == sdict["p1"](0)
    assert sdict.p1(2) == 3 == sdict["m1"](4)
    assert sdict(7) == 6 == sdict.default(7)

    if use_setitem:
      sdict["m1"] = p1
    else:
      sdict.strategy("m1")(p1)

    assert len(sdict) == 2
    assert set(sdict.keys()) == {("minus_one",), ("plus_one", "p1", "m1")}
    assert all(name in dir(sdict) for name in names)
    assert all(name in vars(sdict) for name in names)
    assert "default" in vars(sdict)
    assert sdict.default == m1

    assert sdict.m1(2) == 3 == sdict["m1"](2) == sdict["plus_one"](2)
    assert sdict.p1(2) == 3 == sdict["p1"](2) == sdict["minus_one"](4)
    assert sdict(5) == 4 == sdict.default(5) # -1

    if use_setitem:
      sdict["minus_one"] = p1
    else:
      sdict.strategy("minus_one")(p1)

    assert len(sdict) == 1
    assert list(sdict.keys()) == [("plus_one", "p1", "m1", "minus_one")]
    assert all(name in dir(sdict) for name in names)
    assert all(name in vars(sdict) for name in names)
    assert "default" in vars(sdict) # But it was replaced
    assert sdict.default == p1

    assert sdict.minus_one(2) == 3 == sdict["m1"](2) == sdict["plus_one"](2)
    assert sdict.plus_one(2) == 3 == sdict["p1"](2) == sdict["minus_one"](2)
    assert sdict(7) == 8 == sdict.default(7) # +1 !!!
