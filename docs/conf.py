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
AudioLazy documentation configuration file for Sphinx
"""

import sys, os
import audiolazy
import shlex
from subprocess import Popen, PIPE
import time
from collections import OrderedDict
import types
from audiolazy import iteritems


def splitter(lines, sep="-=", keep_idx=False):
  """
  Splits underlined blocks without indentation (reStructuredText pattern).

  Parameters
  ----------
  lines :
    A list of strings
  sep :
    Underline symbols. A line with only such symbols will be seen as a
    underlined one.
  keep_idx :
    If False (default), the function returns a collections.OrderedDict. Else,
    returns a
    list of index pairs

  Returns
  -------
  A collections.OrderedDict instance where a block with underlined key like
  ``"Key\\n==="`` and a list of lines following will have the item (key, list
  of lines), in the order that they appeared in the lists input. Empty keys
  gets an order numbering, and might happen for example after a ``"----"``
  separator. The values (lists of lines) don't include the key nor its
  underline, and is also stripped/trimmed as lines (i.e., there's no empty
  line as the first and last list items, but first and last line may start/end
  with whitespaces).

  """
  separators = audiolazy.Stream(
                 idx - 1 for idx, el in enumerate(lines)
                         if all(char in sep for char in el)
                         and len(el) > 0
               ).append([len(lines)])
  first_idx = separators.copy().take()
  blk_data = OrderedDict()

  empty_count = iter(audiolazy.count(1))
  next_empty = lambda: "--Empty--{0}--".format(next(empty_count))

  if first_idx != 0:
    blk_data[next_empty()] = lines[:first_idx]

  for idx1, idx2 in separators.blocks(size=2, hop=1):
    name = lines[idx1].strip() if lines[idx1].strip() != "" else next_empty()
    blk_data[name] = lines[idx1+2 : idx2]

  # Strips the empty lines
  for name in blk_data:
    while blk_data[name][-1].strip() == "":
      blk_data[name].pop()
    while blk_data[name][0].strip() == "":
      blk_data[name] = blk_data[name][1:]

  return blk_data


def audiolazy_namer(name):
  """
  Process a name to get Sphinx reStructuredText internal references like
  ``:obj:`name <audiolazy.lazy_something.name>``` for a given name string,
  specific for AudioLazy.

  """
  sp_name = name.split(".")
  try:

    # Find the audiolazy module name
    data = getattr(audiolazy, sp_name[0])
    if isinstance(data, audiolazy.StrategyDict):
      module_name = data.default.__module__
    else:
      module_name = data.__module__
      if not module_name.startswith("audiolazy"): # Decorated math, cmath, ...
        del module_name
        for mname in audiolazy.__modules__:
          if sp_name[0] in getattr(audiolazy, mname).__all__:
            module_name = "audiolazy." + mname
            break

    # Now gets the referenced item
    location = ".".join([module_name] + sp_name)
    for sub_name in sp_name[1:]:
      data = getattr(data, sub_name)

    # Finds the role to be used for referencing
    type_dict = OrderedDict([
      (audiolazy.StrategyDict, "obj"),
      (Exception, "exc"),
      (types.MethodType, "meth"),
      (types.FunctionType, "func"),
      (types.ModuleType, "mod"),
      (property, "attr"),
      (type, "class"),
    ])
    role = [v for k, v in iteritems(type_dict) if isinstance(data, k)][0]

  # Not found
  except AttributeError:
    return ":obj:`{0}`".format(name)

  # Found!
  else:
    return ":{0}:`{1} <{2}>`".format(role, name, location)


def pre_processor(app, what, name, obj, options, lines,
                  namer=lambda name: ":obj:`{0}`".format(name)):
  """
  Callback preprocessor function for docstrings.
  Converts data from Spyder pattern to Sphinx, using a ``namer`` function
  that defaults to ``lambda name: ":obj:`{0}`".format(name)`` (specific for
  ``.. seealso::``).

  """
  # Duplication removal
  if what == "module": # For some reason, summary appears twice
    idxs = [idx for idx, el in enumerate(lines) if el.startswith("Summary")]
    if len(idxs) >= 2:
      del lines[idxs.pop():] # Remove the last summary
    if len(idxs) >= 1:
      lines.insert(idxs[-1] + 1, "")
      if obj is audiolazy.lazy_math:
        lines.insert(idxs[-1] + 1, ".. tabularcolumns:: cl")
      else:
        lines.insert(idxs[-1] + 1, ".. tabularcolumns:: CJ")
      lines.insert(idxs[-1] + 1, "")

  # Real docstring format pre-processing
  result = []
  for name, blk in iteritems(splitter(lines)):
    nlower =  name.lower()

    if nlower == "parameters":
      starters = audiolazy.Stream(idx for idx, el in enumerate(blk)
                                      if len(el) > 0
                                      and not el.startswith(" ")
                                 ).append([len(blk)])
      for idx1, idx2 in starters.blocks(size=2, hop=1):
        param_data = " ".join(b.strip() for b in blk[idx1:idx2])
        param, expl = param_data.split(":", 1)
        if "," in param:
          param = param.strip()
          if not param[0] in ("(", "[", "<", "{"):
            param = "[{0}]".format(param)
        while "," in param:
          fparam, param = param.split(",", 1)
          result.append(":param {0}: {1}".format(fparam.strip(), "\.\.\."))
        result.append(":param {0}: {1}".format(param.strip(), expl.strip()))

    elif nlower == "returns":
      result.append(":returns: " + " ".join(blk))

    elif nlower in ("note", "warning", "hint"):
      result.append(".. {0}::".format(nlower))
      result.extend("  " + el for el in blk)

    elif nlower == "examples":
      result.append("**Examples**:")
      result.extend("  " + el for el in blk)

    elif nlower == "see also":
      result.append(".. seealso::")
      for el in blk:
        if el.endswith(":"):
          result.append("") # Skip a line
           # Sphinx may need help here to find some object locations
          refs = [namer(f.strip()) for f in el[:-1].split(",")]
          result.append("  " + ", ".join(refs))
        else:
          result.append("  " + el)

    else: # Unkown block name, perhaps the starting one (empty)
      result.extend(blk)

    # Skip a line after each block
    result.append("")

  # Replace lines with the processed data while keeping the actual lines id
  del lines[:]
  lines.extend(result)


def should_skip(app, what, name, obj, skip, options):
  """
  Callback object chooser function for docstring documentation.

  """
  if name in ["__doc__", "__module__", "__dict__", "__weakref__",
               "__abstractmethods__"
              ] or name.startswith("_abc_"):
    return True
  return False


def setup(app):
  """
  Just connects the docstring pre_processor and should_skip functions to be
  applied on all docstrings.

  """
  app.connect('autodoc-process-docstring',
              lambda *args: pre_processor(*args, namer=audiolazy_namer))
  app.connect('autodoc-skip-member', should_skip)


def file_name_generator_recursive(path):
  """
  Generator function for filenames given a directory path name. The resulting
  generator don't yield any [sub]directory name.

  """
  for name in os.listdir(path):
    full_name = os.path.join(path, name)
    if os.path.isdir(full_name):
      for new_name in file_name_generator_recursive(full_name):
        yield new_name
    else:
      yield full_name


def newest_file(file_iterable):
  """
  Returns the name of the newest file given an iterable of file names.

  """
  return max(file_iterable, key=lambda fname: os.path.getmtime(fname))


#
# README.rst file loading
#

# Gets README.rst file location from git (it's on the repository root)
git_command_location = shlex.split("git rev-parse --show-cdup")
git_output = Popen(git_command_location, stdout=PIPE).stdout.read()
file_location = git_output.decode("utf-8").strip()
readme_file_name = os.path.join(file_location, "README.rst")

# Opens the file (this should be importable!)
with open(readme_file_name, "r") as readme_file:
  readme_file_contents = readme_file.read().splitlines()

# Loads the description
description = "\n".join(splitter(readme_file_contents)["AudioLazy"])


#
# General configuration
#
extensions = [
  "sphinx.ext.autodoc",
  "sphinx.ext.doctest",
  "sphinx.ext.coverage",
  "sphinx.ext.mathjax",
  "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
source_suffix = ".rst"
source_encoding = "utf-8"
master_doc = "index"

# General information about the project.
project = "AudioLazy" # Typed just to keep the UpperCamelCase
title = " ".join([project, "documentation"])
year = "2012-2016" # Not used directly by Sphinx
author = audiolazy.__author__ # Not used directly by Sphinx
copyright = ", ".join([year, author])
version = audiolazy.__version__

# If it's a development version, get release date using the last git commit
if version.endswith("dev"):
  git_command_line = "git log --date-order --date=raw --format=%cd -1".split()
  git_time_string = Popen(git_command_line, stdout=PIPE).stdout.read()
  git_raw_time = git_time_string.split()[0]
  iso_release_time = time.strftime("%Y%m%dT%H%M%SZ", # ISO 8601 format, UTF
                                   time.gmtime(int(git_raw_time)))
  release = version + iso_release_time
else:
  release = version

# Get "today" using the last file modification date
# WARNING: Be careful with git clone, clonning date will be "today"
install_path = audiolazy.__path__[0]
installed_nfile = newest_file(file_name_generator_recursive(install_path))
installed_time = os.path.getmtime(installed_nfile)
today = time.strftime("%Y-%m-%d", time.gmtime(installed_time)) # In UTF time

exclude_patterns = []

add_module_names = False
pygments_style = "sphinx"


# HTML output configuration
html_theme = "default"
html_static_path = ["_static"]
htmlhelp_basename = project + "doc"


# LaTeX output configuration
latex_elements = {
  "papersize": "a4paper",
  "pointsize": "10pt", # Font size
  "preamble": r"  \setlength{\tymax}{360pt}",
  "fontpkg": "\\usepackage{cmbright}",
}

latex_documents = [(
  master_doc,
  project + ".tex", # Target
  title,
  author,
  "manual", # The documentclass ("howto"/"manual")
)]

latex_show_pagerefs = True
latex_show_urls = "footnote"
latex_domain_indices = False


# Man (manual page) output configuration
man_pages = [(
  master_doc,
  project.lower(), # Name
  title, # Description
  [author],
  1, # Manual section
)]


# Texinfo output configuration
texinfo_documents = [(
  master_doc,
  project, # Target
  title,
  author,
  project, # Dir menu entry
  description, # From README.rst
  "Miscellanous", # Category
)]


# Epub output configuration
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright


#
# Item in sys.modules for StrategyDict instances (needed for automodule)
#
for name, sdict in iteritems(audiolazy.__dict__):
  if isinstance(sdict, audiolazy.StrategyDict):
    fname = ".".join([sdict.default.__module__, name])
    sdict.__all__ = tuple(x[0] for x in sdict.keys())
    sys.modules[fname] = sdict
