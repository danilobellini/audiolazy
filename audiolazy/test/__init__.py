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
# Created on Thu Jul 19 2012
# danilo [dot] bellini [at] gmail [dot] com
"""
AudioLazy testing sub-package
"""

import pytest
from _pytest.skipping import XFailed
import types
from importlib import import_module, sys

# Audiolazy internal imports
from ..lazy_compat import meta
from ..lazy_core import AbstractOperatorOverloaderMeta


def skipper(msg="There's something not supported in this environment"):
  """
  Internal function to work as the last argument in a ``getattr`` call to
  help skip environment-specific tests when needed.
  """
  def skip(*args, **kwargs):
    pytest.skip(msg.format(*args, **kwargs))
  return skip


class XFailerMeta(AbstractOperatorOverloaderMeta):
  """
  Metaclass for XFailer, ensuring every operator use is a pytest.xfail call.
  """
  def __binary__(cls, op):
    return lambda self, other: self()
  __unary__ = __rbinary__ = __binary__


class XFailer(meta(metaclass=XFailerMeta)):
  """
  Class that responds to mostly uses as a pytest.xfail call.
  """
  def __init__(self, module):
    self.module = module

  def __call__(self, *args, **kwargs):
    pytest.xfail(reason="Module {} not found".format(self.module))

  def __getattr__(self, name):
    return self.__call__

  __iter__ = __call__


class XFailerModule(types.ModuleType):
  """
  Internal fake module creator to ensure xfail in all functions, if module
  doesn't exist.
  """
  def __init__(self, name):
    try:
      if isinstance(import_module(name.split(".", 1)[0]), XFailerModule):
        raise ImportError
      import_module(name)
    except (ImportError, XFailed):
      sys.modules[name] = self
      self.__name__ = name

  __file__ = __path__ = __loader__ = ""

  def __getattr__(self, name):
    return XFailer(self.__name__)


# Creates an XFailer for each module that isn't available
XFailerModule("numpy")
XFailerModule("numpy.fft")
XFailerModule("numpy.linalg")
XFailerModule("_portaudio")
XFailerModule("pyaudio")
XFailerModule("scipy")
XFailerModule("scipy.optimize")
XFailerModule("scipy.signal")
XFailerModule("sympy")

# The two Numpy mocks below are due to lazy_synth.line doctest in pypy (without
# Numpy), since py.test 2.5.2 doesn't accept xfail in doctests
if isinstance(sys.modules["numpy"], XFailerModule):
  np = sys.modules["numpy"]
  from ..lazy_synth import line
  from ..lazy_compat import xmap, xzip

  class MatrixMockMeta(AbstractOperatorOverloaderMeta):
    __operators__ = "add sub mul rmul truediv" # Elementwise or broadcast only

    def __binary__(cls, op):
      def dunder(self, other):
        if isinstance(other, cls):
          op_func_row = lambda r1, r2: list(xmap(op.func, r1, r2))
          return cls(xmap(op_func_row, self, other))
        return cls([[op.func(el, other) for el in row] for row in self])
      return dunder # We don't need matrix multiplication

    def __rbinary__(cls, op):
      def dunder(self, other):
        return cls([[op.func(other, el) for el in row] for row in self])
      return dunder

  class MatrixMock(meta(list, metaclass=MatrixMockMeta)):

    def reshape(self, *args, **kwargs):
      pytest.xfail(reason="Numpy not found, this method wasn't mocked")
    tolist = reshape

    def __str__(self): # To keep the same from that doctest
      num2str = lambda el: "%2d." % el + ("" if el.is_integer() else
                                          ("%g" % el).split(".")[1])
      rows = [[num2str(el) for el in row] for row in self]
      widths = [max(xmap(len, col)) for col in xzip(*rows)]
      rows = [[(el + " " * w)[:w] for w, el in xzip(widths, row)]
                                  for row in rows]
      sqbrackets = lambda el: "[{}]".format(el)
      csv = lambda row: sqbrackets(" ".join(el for el in row))
      return sqbrackets("\n ".join(csv(row) for row in rows))

  class Array(list):
    def __repr__(self):
      csv = ", ".join("{:4g}".format(el) for el in self)
      return "array([{}])".format(csv)

  def linspace(start, stop, size):
    return Array(line(size, start, stop, finish=True))

  np.linspace = linspace
  np.mat = MatrixMock
