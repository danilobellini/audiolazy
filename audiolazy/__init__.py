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
# Created on Thu Jul 19 2012
# danilo [dot] bellini [at] gmail [dot] com
"""
AudioLazy package

This is the main package file, that already imports all modules into the
system. As the full name might not be small enough for typing it everywhere,
you can import with a helpful alias:

  >>> import audiolazy as lz
  >>> lz.Stream(1, 3, 2).take(8)
  [1, 3, 2, 1, 3, 2, 1, 3]

But there's some parts of the code you probably will find it cleaner to import
directly, like the ``z`` object:

  >>> from audiolazy import z, Stream
  >>> filt = 1 / (1 - z ** -1) # Accumulator linear filter
  >>> filt(Stream(1, 3, 2), zero=0).take(8)
  [1, 4, 6, 7, 10, 12, 13, 16]

For a single use within a console or for trying some new experimental ideas
(perhaps with IPython), you would perhaps find easier to import the full
package contents:

  >>> from audiolazy import *
  >>> s, Hz = sHz(44100)
  >>> delay_a4 = freq_to_lag(440 * Hz)
  >>> filt = ParallelFilter(comb.tau(delay_a4, 20 * s),
  ...                       resonator(440 * Hz, bandwidth=100 * Hz)
  ...                      )
  >>> len(filt)
  2

There's documentation inside the package classes and functions docstrings.
If you try ``dir(audiolazy)`` [or ``dir(lz)``] after importing it [with the
suggested alias], you'll see all the package contents, and the names starting
with ``lazy`` followed by an underscore are modules. If you're starting now,
try to see the docstring from the Stream and ZFilter classes with the
``help(lz.Stream)`` and ``help(lz.ZFilter)`` commands, and then the help from
the other functionalities used above. If you didn't know the ``dir`` and
``help`` built-ins before reading this, it's strongly suggested you to read
first a Python documentation or tutorial, at least enough for you to
understand the basic behaviour and syntax of ``for`` loops, iterators,
iterables, lists, generators, list comprehensions and decorators.

This package was created by Danilo J. S. Bellini and is a free software,
under the terms of the GPLv3.

"""

import os
import sys

# Find all module names
if "__path__" not in locals(): # Happens with Sphinx
  __path__ = os.path.split(__file__)[0]
__modules_prefix__ = "lazy_"
__modules__ = sorted({_mname.split(".")[0]
                      for _mname in os.listdir(__path__[0])
                      if _mname.startswith(__modules_prefix__)
                     })

# Imports all modules to the main namespace
__all__ = []
for _mname in __modules__:
  exec "from .{0} import *".format(_mname)
  __all__ += sys.modules[
               ".".join([__name__, _mname])
             ].__all__ # With that, __all__ don't include the module names

# Remove references just for namespace clean-up
if "_mname" in locals(): # Not all interpreters keep the for loop variable
  del _mname
del os
del sys

#
# <SETUP.PY> #
# Metadata (see setup.py for more information about these)
# This section should not reference anything from before!
#
__version__ = "0.04dev"
__author__ = "Danilo de Jesus da Silva Bellini"
__author_email__  = "danilo [dot] bellini [at] gmail [dot] com"
__url__ = "http://github.com/danilobellini/audiolazy"
