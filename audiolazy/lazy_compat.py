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
Compatibility tools to keep the same source working in both Python 2 and 3
"""

import types
import itertools as it
import sys

__all__ = ["orange", "PYTHON2", "builtins", "xrange", "xzip", "xzip_longest",
           "xmap", "xfilter", "STR_TYPES", "INT_TYPES", "SOME_GEN_TYPES",
           "NEXT_NAME", "iteritems", "itervalues", "im_func", "meta"]


def orange(*args, **kwargs):
  """
  Old Python 2 range (returns a list), working both in Python 2 and 3.
  """
  return list(range(*args, **kwargs))


PYTHON2 = sys.version_info.major == 2
if PYTHON2:
  builtins = sys.modules["__builtin__"]
else:
  import builtins


xrange = getattr(builtins, "xrange", range)
xzip = getattr(it, "izip", zip)
xzip_longest = getattr(it, "izip_longest", getattr(it, "zip_longest", None))
xmap = getattr(it, "imap", map)
xfilter = getattr(it, "ifilter", filter)


STR_TYPES = (getattr(builtins, "basestring", str),)
INT_TYPES = (int, getattr(builtins, "long", None)) if PYTHON2 else (int,)
SOME_GEN_TYPES = (types.GeneratorType, xrange(0).__class__, enumerate, xzip,
                  xzip_longest, xmap, xfilter)
NEXT_NAME = "next" if PYTHON2 else "__next__"


def iteritems(dictionary):
  """
  Function to use the generator-based items iterator over built-in
  dictionaries in both Python 2 and 3.
  """
  try:
    return getattr(dictionary, "iteritems")()
  except AttributeError:
    return iter(getattr(dictionary, "items")())


def itervalues(dictionary):
  """
  Function to use the generator-based value iterator over built-in
  dictionaries in both Python 2 and 3.
  """
  try:
    return getattr(dictionary, "itervalues")()
  except AttributeError:
    return iter(getattr(dictionary, "values")())


def im_func(method):
  """ Gets the function from the method in both Python 2 and 3. """
  return getattr(method, "im_func", method)


def meta(*bases, **kwargs):
  """
  Allows unique syntax similar to Python 3 for working with metaclasses in
  both Python 2 and Python 3.

  Examples
  --------
  >>> class BadMeta(type): # An usual metaclass definition
  ...   def __new__(mcls, name, bases, namespace):
  ...     if "bad" not in namespace: # A bad constraint
  ...       raise Exception("Oops, not bad enough")
  ...     value = len(name) # To ensure this metaclass is called again
  ...     def really_bad(self):
  ...       return self.bad() * value
  ...     namespace["really_bad"] = really_bad
  ...     return super(BadMeta, mcls).__new__(mcls, name, bases, namespace)
  ...
  >>> class Bady(meta(object, metaclass=BadMeta)):
  ...   def bad(self):
  ...     return "HUA "
  ...
  >>> class BadGuy(Bady):
  ...   def bad(self):
  ...     return "R"
  ...
  >>> issubclass(BadGuy, Bady)
  True
  >>> Bady().really_bad() # Here value = 4
  'HUA HUA HUA HUA '
  >>> BadGuy().really_bad() # Called metaclass ``__new__`` again, so value = 6
  'RRRRRR'

  """
  metaclass = kwargs.get("metaclass", type)
  if not bases:
    bases = (object,)
  class NewMeta(type):
    def __new__(mcls, name, mbases, namespace):
      if name:
        return metaclass.__new__(metaclass, name, bases, namespace)
      return super(NewMeta, mcls).__new__(mcls, "", mbases, {})
  return NewMeta("", tuple(), {})
