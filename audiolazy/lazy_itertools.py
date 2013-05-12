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
# Created on Sat Oct 06 2012
# danilo [dot] bellini [at] gmail [dot] com
"""
Itertools module "decorated" replica, where all outputs are Stream instances
"""

import itertools as it
from collections import Iterator

# Audiolazy internal imports
from .lazy_stream import tostream, Stream
from .lazy_misc import xrange, xzip
from .lazy_core import StrategyDict


# "Decorates" all functions from itertools
__all__ = ["tee", "chain", "izip"]
it_names = set(dir(it)).difference(__all__)
for func in filter(callable, [getattr(it, name) for name in it_names]):
  name = func.__name__
  if name in ["filterfalse", "zip_longest"]: # These were renamed in Python 3
    name = "i" + name # In AudioLazy, keep the Python 2 names
  __all__.append(name)
  locals()[name] = tostream(func)


# StrategyDict chain, following "from_iterable" from original itertool
chain = StrategyDict("chain")
chain.strategy("chain")(tostream(it.chain))
chain.strategy("star", "from_iterable")(tostream(it.chain.from_iterable))


# StrategyDict izip, allowing izip.longest instead of izip_longest
izip = StrategyDict("izip")
izip.strategy("izip", "smallest")(tostream(xzip))
izip["longest"] = izip_longest


# Includes the imap and ifilter (they're not from itertools in Python 3)
for name, func in zip(["imap", "ifilter"], [map, filter]):
  if name not in __all__:
    __all__.append(name)
    locals()[name] = tostream(func)


# Python 3 has an "accumulate" itertool, this makes it available in Python 2
if "accumulate" not in __all__:
  __all__.append("accumulate")
  @tostream
  def accumulate(iterable):
    " Return series of accumulated sums. "
    iterator = iter(iterable)
    sum_data = next(iterator)
    yield sum_data
    for el in iterator:
      sum_data += el
      yield sum_data


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
  thub :
    use Stream instances *almost* like constants in your equations.

  """
  if isinstance(data, (Stream, Iterator)):
    return tuple(Stream(cp) for cp in it.tee(data, n))
  else:
    return tuple(data for unused in xrange(n))
