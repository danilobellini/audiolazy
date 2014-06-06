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
# Created on Mon May 19 01:00:57 2014
# danilo [dot] bellini [at] gmail [dot] com
"""
Resources for opening data from Wave (.wav) files
"""

from __future__ import division

from struct import Struct
import wave

# Audiolazy internal imports
from .lazy_stream import Stream
from .lazy_compat import PYTHON2

__all__ = ["WavStream"]


class WavStream(Stream):
  """
  A Stream related to a Wave file

  A WavStream instance is a Stream with extra attributes:

  * ``rate``: sample rate in samples per second;
  * ``channels``: number of channels (1 for mono, 2 for stereo, etc.);
  * ``bits``: bits per sample, a value in ``[8, 16, 24, 32]``.

  Example
  -------

  .. code-block:: python

    song = WavStream("my_song.wav")
    with AudioIO(True) as player:
      player.play(song)
  """
  unpackers = {
    8 : lambda v: ord(v) / 128 - 1, # The only unsigned
    16: (lambda h: lambda v: h(v)[0] / 2 ** 15)(Struct("<h").unpack),
    32: (lambda f: lambda v: f(v)[0])(Struct("<f").unpack), # Already float
  }
  if PYTHON2:
    unpackers[24] = lambda v: (lambda k: k if k < 1 else k - 2)(
                       sum(ord(vi) << i * 8
                           for i, vi in enumerate(v)) / 2 ** 23
                    )
  else: # Bytes items are already int on Python 3
    unpackers[24] = lambda v: (lambda k: k if k < 1 else k - 2)(
                       sum(vi << i * 8 for i, vi in enumerate(v)) / 2 ** 23
                    )

  def __init__(self, wave_file):
    """ Loads a Wave audio file given its name or a file-behaved object. """
    self._file = wave.open(wave_file, "rb")
    self.rate = self._file.getframerate()
    self.channels = self._file.getnchannels()
    self.bits = 8 * self._file.getsampwidth()

    def data_generator(unpacker):
      """ Internal wave data generator, given a single sample unpacker """
      w = self._file
      try:
        while True:
          el = w.readframes(1)
          if not el:
            break
          yield unpacker(el)
      finally:
        w.close()

    super(WavStream, self).__init__(data_generator(self.unpackers[self.bits]))
