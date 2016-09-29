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
AudioLazy documentation creator via Sphinx

Note
----
You should call rst_creator first!
"""

import shlex, sphinx, sys
from subprocess import call

# Call string templates
sphinx_template = "sphinx-build -b {out_type} -d {build_dir}/doctrees "\
                  "-D latex_paper_size=a4 . {build_dir}/{out_type}"
make_template = "make -C {build_dir}/{out_type} {make_param}"

# Make targets given the output type
make_target = {"latex": "all-pdf",
               "texinfo": "info"}

def call_sphinx(out_type, build_dir = "build"):
  """
  Call the ``sphinx-build`` for the given output type and the ``make`` when
  the target has this possibility.

  Parameters
  ----------
  out_type :
    A builder name for ``sphinx-build``. See the full list at
    `<http://sphinx-doc.org/invocation.html>`_.
  build_dir :
    Directory for storing the output. Defaults to "build".

  """
  sphinx_string = sphinx_template.format(build_dir=build_dir,
                                         out_type=out_type)
  if sphinx.main(shlex.split(sphinx_string)) != 0:
    raise RuntimeError("Something went wrong while building '{0}'"
                       .format(out_type))
  if out_type in make_target:
    make_string = make_template.format(build_dir=build_dir,
                                       out_type=out_type,
                                       make_param=make_target[out_type])
    call(shlex.split(make_string)) # Errors here don't need to stop anything

# Calling this as a script builds/makes all targets in the list below
if __name__ == "__main__":
  for target in sys.argv[1:] or ["text", "html", "latex", "man",
                                 "texinfo", "epub"]:
    call_sphinx(target)
