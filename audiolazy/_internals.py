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
AudioLazy internals module

The resources found here aren't DSP related nor take part of the main
``audiolazy`` namespace. Unless you're changing or trying to understand
the AudioLazy internals, you probably don't need to know about this.
"""

from functools import wraps, reduce
from warnings import warn
from glob import glob
from operator import concat
import os


def deprecate(func):
  """ A deprecation warning emmiter as a decorator. """
  @wraps(func)
  def wrapper(*args, **kwargs):
    warn("Deprecated, this will be removed in the future", DeprecationWarning)
    return func(*args, **kwargs)
  wrapper.__doc__ = "Deprecated.\n" + (wrapper.__doc__ or "")
  return wrapper


#
# __init__.py importing resources
#

def get_module_names(package_path, pattern="lazy_*.py*"):
  """
  All names in the package directory that matches the given glob, without
  their extension. Repeated names should appear only once.
  """
  package_contents = glob(os.path.join(package_path[0], pattern))
  relative_path_names = (os.path.split(name)[1] for name in package_contents)
  no_ext_names = (os.path.splitext(name)[0] for name in relative_path_names)
  return sorted(set(no_ext_names))

def get_modules(package_name, module_names):
  """ List of module objects from the package, keeping the name order. """
  def get_module(name):
    return __import__(".".join([package_name, name]), fromlist=[package_name])
  return [get_module(name) for name in module_names]

def dunder_all_concat(modules):
  """ Single list with all ``__all__`` lists from the modules. """
  return reduce(concat, (getattr(m, "__all__", []) for m in modules), [])


#
# Resources for module/package summary tables on doctring
#

def summary_table(pairs, key_header, descr_header="Description", width=78):
  """
  List of one-liner strings containing a reStructuredText summary table
  for the given pairs ``(name, object)``.
  """
  from .lazy_text import rst_table, small_doc
  max_width = width - max(len(k) for k, v in pairs)
  table = [(k, small_doc(v, max_width=max_width)) for k, v in pairs]
  return rst_table(table, (key_header, descr_header))

def docstring_with_summary(docstring, pairs, key_header, summary_type):
  """ Return a string joining the docstring with the pairs summary table. """
  return "\n".join(
    [docstring, "Summary of {}:".format(summary_type), ""] +
    summary_table(pairs, key_header) + [""]
  )

def append_summary_to_module_docstring(module):
  """
  Change the ``module.__doc__`` docstring to include a summary table based
  on its contents as declared on ``module.__all__``.
  """
  pairs = [(name, getattr(module, name)) for name in module.__all__]
  kws = dict(key_header="Name", summary_type="module contents")
  module.__doc__ = docstring_with_summary(module.__doc__, pairs, **kws)


#
# Package initialization, first function to be called internally
#

def init_package(package_path, package_name, docstring):
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
  module_names = get_module_names(package_path)
  modules = get_modules(package_name, module_names)
  dunder_all = dunder_all_concat(modules)
  for module in modules:
    append_summary_to_module_docstring(module)
  pairs = list(zip(module_names, modules))
  kws = dict(key_header="Module", summary_type="package modules")
  new_docstring = docstring_with_summary(docstring, pairs, **kws)
  return module_names, dunder_all, new_docstring
