#!/usr/bin/env python
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
AudioLazy package setup file
"""

from setuptools import setup
from setuptools.command.test import test as TestClass
import os, ast

class Tox(TestClass):
  user_options = []

  def finalize_options(self):
    TestClass.finalize_options(self)
    self.test_args = ["-v"] if self.verbose else []
    self.test_suite = True

  def run_tests(self):
    import sys, tox
    sys.exit(tox.cmdline(self.test_args))


def locals_from_exec(code):
  """ Run code in a qualified exec, returning the resulting locals dict """
  namespace = {}
  exec(code, {}, namespace)
  return namespace

def pseudo_import(fname):
  """ Namespace dict from assignments in the file without ``__import__`` """
  is_d_import = lambda n: isinstance(n, ast.Name) and n.id == "__import__"
  is_assign = lambda n: isinstance(n, ast.Assign)
  is_valid = lambda n: is_assign(n) and not any(map(is_d_import, ast.walk(n)))
  with open(fname, "r") as f:
    astree = ast.parse(f.read(), filename=fname)
  astree.body = [node for node in astree.body if is_valid(node)]
  return locals_from_exec(compile(astree, fname, mode="exec"))


def read_rst_and_process(fname, line_process=lambda line: line):
  """
  The reStructuredText string in file ``fname``, without the starting ``..``
  comment and with ``line_process`` function applied to every line.
  """
  with open(fname, "r") as f:
    data = f.read().splitlines()
  first_idx = next(idx for idx, line in enumerate(data) if line.strip())
  if data[first_idx].strip() == "..":
    next_idx = first_idx + 1
    first_idx = next(idx for idx, line in enumerate(data[next_idx:], next_idx)
                         if line.strip() and not line.startswith(" "))
  return "\n".join(map(line_process, data[first_idx:]))

def image_path_processor_factory(path):
  """ Processor for concatenating the ``path`` to relative path images """
  def processor(line):
    markup = ".. image::"
    if line.startswith(markup):
      fname = line[len(markup):].strip()
      if not(fname.startswith("/") or "://" in fname):
        return "{} {}{}".format(markup, path, fname)
    return line
  return processor

def read_description(readme_file, changes_file, images_url):
  updater = image_path_processor_factory(images_url)
  readme_data = read_rst_and_process(readme_file, updater)
  changes_data = read_rst_and_process(changes_file, updater)
  parts = readme_data.split("\n\n", 12)
  title = parts[0]
  pins = "\n\n".join(parts[1:-2])
  descr = parts[-2]
  sections = parts[-1].rsplit("----", 1)[0]
  long_descr_blocks = ["", title, "", pins, "", sections, "", changes_data]
  return descr, "\n".join(block.strip() for block in long_descr_blocks)


path = os.path.split(__file__)[0]
package_name = "audiolazy"

fname_init = os.path.join(path, package_name, "__init__.py")
fname_readme = os.path.join(path, "README.rst")
fname_changes = os.path.join(path, "CHANGES.rst")
images_url = "https://raw.github.com/danilobellini/audiolazy/master/"

metadata = {k.strip("_") : v for k, v in pseudo_import(fname_init).items()}
metadata["description"], metadata["long_description"] = \
  read_description(fname_readme, fname_changes, images_url)
metadata["classifiers"] = """
Development Status :: 3 - Alpha
Intended Audience :: Developers
Intended Audience :: Education
Intended Audience :: Science/Research
Intended Audience :: Other Audience
License :: OSI Approved :: GNU General Public License v3 (GPLv3)
Operating System :: MacOS
Operating System :: Microsoft :: Windows
Operating System :: POSIX :: Linux
Operating System :: OS Independent
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.2
Programming Language :: Python :: 3.3
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Programming Language :: Python :: Implementation :: CPython
Programming Language :: Python :: Implementation :: PyPy
Topic :: Artistic Software
Topic :: Multimedia :: Sound/Audio
Topic :: Multimedia :: Sound/Audio :: Analysis
Topic :: Multimedia :: Sound/Audio :: Capture/Recording
Topic :: Multimedia :: Sound/Audio :: Editors
Topic :: Multimedia :: Sound/Audio :: Mixers
Topic :: Multimedia :: Sound/Audio :: Players
Topic :: Multimedia :: Sound/Audio :: Sound Synthesis
Topic :: Multimedia :: Sound/Audio :: Speech
Topic :: Scientific/Engineering
Topic :: Software Development
Topic :: Software Development :: Libraries
Topic :: Software Development :: Libraries :: Python Modules
""".strip().splitlines()
metadata["license"] = "GPLv3"
metadata["name"] = package_name
metadata["packages"] = [package_name]
metadata["tests_require"] = ["tox"]
metadata["cmdclass"] = {"test": Tox}
setup(**metadata)
