#!/usr/bin/env python
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
AudioLazy documentation creator via Sphinx

You should call rst_creator first!

"""

from subprocess import call

sphinx_template = "sphinx-build -b {out_type} -d {build_dir}/doctrees "\
                  "-D latex_paper_size=a4 . {build_dir}/{out_type}"
make_template = "make -C {build_dir}/{out_type} {make_param}"

make_after = {"latex": "all-pdf",
              "texinfo": "info"}

out_types = ["text", "html", "latex", "man", "texinfo", "epub"]
build_dir = "build"

for out_type in out_types:
  call_string = sphinx_template.format(build_dir=build_dir,
                                       out_type=out_type)
  if call(call_string.split()) != 0:
    break
  if out_type in make_after:
    make_string = make_template.format(build_dir=build_dir,
                                       out_type=out_type,
                                       make_param=make_after[out_type])
    call(make_string.split())

