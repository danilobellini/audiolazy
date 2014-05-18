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
# Created on Sat May 17 22:58:26 2014
# danilo [dot] bellini [at] gmail [dot] com
"""
AudioLazy internals module

The resources found here aren't DSP related and doesn't take part of the
main ``audiolazy`` namespace.
"""

from functools import wraps, reduce
from warnings import warn
from glob import glob
from operator import concat
import os


def _deprecate(func):
  """ A deprecation warning emmiter as a decorator. """
  @wraps(func)
  def wrapper(*args, **kwargs):
    warn("Deprecated, this will be removed in th future", DeprecationWarning)
    return func(*args, **kwargs)
  wrapper.__doc__ = "Deprecated.\n" + wrapper.__doc__
  return wrapper


def _get_module_names(package_path, pattern="lazy_*.py*"):
  """
  All names in the package directory that matches the given glob, without
  their extension. Repeated names should appear only once.
  """
  package_contents = glob(os.path.join(package_path[0], pattern))
  relative_path_names = (os.path.split(name)[1] for name in package_contents)
  no_ext_names = (os.path.splitext(name)[0] for name in relative_path_names)
  return sorted(set(no_ext_names))

def _get_modules(package_name, module_names):
  """ List of module objects from the package, keeping the name order. """
  def get_module(name):
    return __import__(".".join([package_name, name]), fromlist=[package_name])
  return [get_module(name) for name in module_names]

def _dunder_all_concat(modules):
  return reduce(concat, (getattr(m, "__all__", []) for m in modules), [])


def _summary_table(pairs, key_header):
  from .lazy_text import rst_table, small_doc
  max_width = 78 - max(len(k) for k, v in pairs)
  table = [(k, small_doc(v, max_width=max_width)) for k, v in pairs]
  return rst_table(table, (key_header, "Description"))

def _docstring_with_summary(docstring, pairs, key_header, summary_type):
  return "\n".join(
    [docstring, "Summary of {}:".format(summary_type), ""] +
    _summary_table(pairs, key_header) + [""]
  )

def _append_summary_to_module_docstring(module):
  pairs = [(name, getattr(module, name)) for name in module.__all__]
  kws = dict(key_header="Name", summary_type="module contents")
  module.__doc__ = _docstring_with_summary(module.__doc__, pairs, **kws)


def _init_package(package_path, package_name, docstring):
  """
  Package initialization, to be called only by ``__init__.py``.

  - Find all module names;
  - Import all modules (so they're already cached on sys.modules), in
    the sorting order (this might make difference on cyclic imports);
  - Update all module docstrings (with the summary of its contents);
  - Build a module summary for the package docstring.

  Returns
  -------
  A 4-length tuple ``(modules, __all__, __doc__)``. The first one can be
  used by the package to import every module into the main package namespace.
  """
  module_names = _get_module_names(package_path)
  modules = _get_modules(package_name, module_names)
  dunder_all = _dunder_all_concat(modules)
  for module in modules:
    _append_summary_to_module_docstring(module)
  pairs = list(zip(module_names, modules))
  kws = dict(key_header="Module", summary_type="package modules")
  new_docstring = _docstring_with_summary(docstring, pairs, **kws)
  return module_names, dunder_all, new_docstring
