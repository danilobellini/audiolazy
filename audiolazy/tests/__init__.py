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
XFailerModule("scipy.interpolate")
XFailerModule("sympy")
