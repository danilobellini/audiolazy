#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of AudioLazy, the signal processing Python package.
# Copyright (C) 2012-2014 Danilo de Jesus da Silva Bellini
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
# Created on Wed Nov 07 2012
# danilo [dot] bellini [at] gmail [dot] com
"""
Simple I/O wire example, connecting the input directly to the output

This example uses the default PortAudio API, however you can change it by
using the "api" keyword argument in AudioIO creation, like

  with AudioIO(True, api="jack") as pr:

obviously, you can use another API instead (like "alsa").

Note
----
When using JACK, keep chunks.size = 1
"""

from audiolazy import chunks, AudioIO

chunks.size = 16 # Amount of samples per chunk to be sent to PortAudio
with AudioIO(True) as pr: # A player-recorder
  pr.play(pr.record())
