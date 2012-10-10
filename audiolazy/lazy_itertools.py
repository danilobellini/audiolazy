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

# Audiolazy internal imports
from .lazy_stream import tostream

# All functions from itertools
for func in filter(callable, [getattr(it, name) for name in dir(it)]):
  locals()[func.__name__] = tostream(func)
