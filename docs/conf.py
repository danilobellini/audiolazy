# -*- coding: utf-8 -*-
# This file is part of AudioLazy, the signal processing Python package.
# Copyright (C) 2012 Danilo de Jesus da Silva Bellini
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
# Created on Fri Feb 08 2013
# danilo [dot] bellini [at] gmail [dot] com
"""
AudioLazy documentation build configuration file

"""

import sys, os
import audiolazy
from subprocess import Popen, PIPE
import time
from collections import OrderedDict


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
  "Key\n===" and a list of lines following will have the item (key, list of
  lines), in the order that they appeared in the lists input. Empty keys gets
  an order numbering, and might happen for example after a "----" separator.
  The values (lists of lines) don't include the key nor its underline, and is
  also stripped/trimmed (i.e., there's no empty line as the first and last
  list items).

  """
  separators = audiolazy.Stream(
                 idx - 1 for idx, el in enumerate(lines)
                         if all(char in sep for char in el)
                         and len(el) > 0
               ).append([len(lines)])
  first_idx = separators.copy().take()
  blk_data = OrderedDict()

  empty_count = iter(audiolazy.count(1))
  next_empty = lambda: "--Empty--{0}--".format(empty_count.next())

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


def pre_processor(app, what, name, obj, options, lines):
  result = []
  for name, blk in splitter(lines).iteritems():
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

    elif nlower == "note":
      result.append(".. note::")
      result.extend("  " + el for el in blk)

    elif nlower == "examples":
      result.append("**Examples**:")
      result.extend("  " + el for el in blk)

    elif nlower == "see also":
      result.append(".. seealso::")
      for el in blk:
        if el.endswith(":"):
          result.append("") # Skip a line
           # Sphinx needs help here to find some object locations
          funcs = [":obj:`{0}`".format(f.strip()) 
                     .replace("sHz","sHz <audiolazy.lazy_misc.sHz>")
                     .replace("dB10","dB10 <audiolazy.lazy_math.dB10>")
                     .replace("dB20","dB20 <audiolazy.lazy_math.dB20>")
                     .replace("phase","phase <audiolazy.lazy_math.phase>")
                     .replace("acorr","acorr <audiolazy.lazy_analysis.acorr>")
                     .replace("thub","thub <audiolazy.lazy_stream.thub>")
                     .replace("levinson_durbin",
                       "levinson_durbin <audiolazy.lazy_lpc.levinson_durbin>")
                   for f in el[:-1].strip().split(",")]
          result.append("  " + ", ".join(funcs))
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
  if name in ["__doc__", "__module__", "__dict__", "__weakref__",
               "_not_implemented", "__operator_inputs__",
               "__abstractmethods__"
              ] or name.startswith("_abc_"):
    return True
  return False


def setup(app):
  """
  Just connects the docstring pre_processor and should_skip functions to be
  applied on all docstrings.

  """
  app.connect('autodoc-process-docstring', pre_processor)
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


def older_file(file_iterable):
  """
  Returns the name of the older file given an iterable of file names.

  """
  return max(file_iterable, key=lambda fname: os.path.getmtime(fname))


# Description (needed for Texinfo) from README.rst, and file location from git
git_command_location = "git rev-parse --show-cdup".split()
file_location = Popen(git_command_location, stdout=PIPE).stdout.read().strip()
with open(os.path.join(file_location, "README.rst"), "r") as readme_file:
  readme_file_contents = readme_file.read().replace("\r\n", "\n")
  description = "\n".join(splitter(readme_file_contents.splitlines()
                                  ).values()[1])


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
year = "2012-2013" # Not used directly by Sphinx
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
installed_older_file = older_file(file_name_generator_recursive(install_path))
installed_time = os.path.getmtime(installed_older_file)
today = time.strftime("%Y-%m-%d", time.gmtime(installed_time)) # In UTF time

exclude_patterns = []

add_module_names = False
pygments_style = "sphinx"


#
# HTML output configuration
#
html_theme = "default"
html_static_path = ["_static"]
htmlhelp_basename = project + "doc"


#
# LaTeX output configuration
#
latex_elements = {
  "papersize": "a4paper",
  "pointsize": "10pt", # Font size
  "preamble": "",
}

latex_documents = [(
  master_doc,
  project + ".tex", # Target
  title,
  author,
  "manual", # The documentclass ("howto"/"manual")
)]

latex_show_pagerefs = True
latex_show_urls = True
latex_domain_indices = False


#
# Man (manual page) output configuration
#
man_pages = [(
  master_doc,
  project.lower(), # Name
  title, # Description
  [author],
  1, # Manual section
)]


#
# Texinfo output configuration
#
texinfo_documents = [(
  master_doc,
  project, # Target
  title,
  author,
  project, # Dir menu entry
  description,
  "Miscellanous", # Category
)]


#
# Epub output configuration
#
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright


#
# Bizarre changes to StrategyDict instances (it works, though...)
#
for name, sdict in audiolazy.__dict__.iteritems():
  if isinstance(sdict, audiolazy.StrategyDict):
    fname = ".".join([sdict.default.__module__, name])
    sdict.__all__ = tuple(x[0] for x in sdict.keys())
    sys.modules[fname] = sdict

