# -*- coding: utf-8 -*-
"""
Itertools module "decorated" replica, where all outputs are Stream instances

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

Created on Sat Oct 06 2012
danilo [dot] bellini [at] gmail [dot] com
"""

import itertools as it
from collections import Iterator

# Audiolazy internal imports
from .lazy_stream import tostream, Stream

# All functions from itertools
__all__ = ["tee"]
it_names = set(dir(it)).difference(__all__)
for func in filter(callable, [getattr(it, name) for name in it_names]):
  name = func.__name__
  __all__.append(name)
  locals()[name] = tostream(func)


def tee(data, n=2):
  """
  Tee or "T" copy to help working with Stream instances as well as with
  numbers.

  Parameters
  ----------
  data :
    Input to be copied. Can be anything.
  n :
    Size of returned tuple. Defaults to 2.

  Returns
  -------
  Tuple of n independent Stream instances, if the input is a Stream or an
  iterator, otherwise a tuple with n times the same object.

  See Also
  --------
  thub : use Stream instances *almost* like constants in your equations.

  """
  if isinstance(data, (Stream, Iterator)):
    return tuple(Stream(cp) for cp in it.tee(data, n))
  else:
    return tuple(data for unused in xrange(n))
