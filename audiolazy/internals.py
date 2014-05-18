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
# Created on Sat May 17 22:58:26 2014
# danilo [dot] bellini [at] gmail [dot] com
"""
AudioLazy internals module

The resources found here aren't DSP related and doesn't take part of the
main ``audiolazy`` namespace.
"""

from functools import wraps
from warnings import warn

def _deprecate(func):
  """ A deprecation warning emmiter as a decorator """
  @wraps(func)
  def wrapper(*args, **kwargs):
    warn("Deprecated, this will be removed in th future", DeprecationWarning)
    return func(*args, **kwargs)
  wrapper.__doc__ = "Deprecated.\n" + wrapper.__doc__
  return wrapper
