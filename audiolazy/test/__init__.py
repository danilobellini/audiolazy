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
# Created on Thu Jul 19 2012
# danilo [dot] bellini [at] gmail [dot] com
"""
AudioLazy testing sub-package
"""

import pytest

def skipper(msg="There's something not supported in this environment"):
  """
  Internal function to work as the last argument in a ``getattr`` call to
  help skip environment-specific tests when needed.

  """
  def skip(*args, **kwargs):
    pytest.skip(msg.format(*args, **kwargs))
  return skip
