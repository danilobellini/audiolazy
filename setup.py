#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AudioLazy package setup file

Copyright (C) 2012 Danilo de Jesus da Silva Bellini

This file is part of AudioLazy, the signal processing Python package.

AudioLazy is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

Created on Thu Oct 10 2012
danilo [dot] bellini [at] gmail [dot] com
"""

from setuptools import setup
import os

path = os.path.split(__file__)[0]
pkgname = "audiolazy"
metadata_file = "__init__.py"

# Get metadata from the package file without actually importing it
metadata = {}
with open(os.path.join(path, pkgname, metadata_file), "r") as f:
  for line in f:
    if line.startswith("__"):
      assignment = [side.strip() for side in line.split("=")]
      metadata[assignment[0].strip("_")] = eval(assignment[1])

# Long description is all from README.txt, but the ending copyright message
with open(os.path.join(path, "README.rst"), "r") as f:
  long_description = f.read()
long_description = long_description.rsplit("----", 1)[0].strip()

classifiers = [
  "Development Status :: 2 - Pre-Alpha",
  "Intended Audience :: Developers",
  "Intended Audience :: Education",
  "Intended Audience :: Science/Research",
  "Intended Audience :: Other Audience",
  "License :: OSI Approved :: "
    "GNU General Public License v3 or later (GPLv3+)",
  "Operating System :: Microsoft :: Windows",
  "Operating System :: POSIX :: Linux",
  "Operating System :: OS Independent",
  "Programming Language :: Python",
  "Topic :: Artistic Software",
  "Topic :: Multimedia :: Sound/Audio",
  "Topic :: Multimedia :: Sound/Audio :: Analysis",
  "Topic :: Multimedia :: Sound/Audio :: Players",
  "Topic :: Multimedia :: Sound/Audio :: Sound Synthesis",
  "Topic :: Scientific/Engineering",
  "Topic :: Software Development",
  "Topic :: Software Development :: Libraries",
  "Topic :: Software Development :: Libraries :: Python Modules",
  "Topic :: Software Development",
]

descr = "Expressive Digital Signal Processing (DSP) package for Python"

setup(name="audiolazy",
      description=descr,
      long_description=long_description,
      license="GPLv3",
      packages=[pkgname],
      classifiers=classifiers,
      **metadata
     )
