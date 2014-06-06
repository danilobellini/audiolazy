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
  _unpackers = {
    8 : ord, # The only unsigned
    16: (lambda a: lambda v: a(v)[0])(Struct("<h").unpack),
    24: (lambda a: lambda v: a(b"\x00" + v)[0] >> 8)(Struct("<i").unpack),
    32: (lambda a: lambda v: a(v)[0])(Struct("<i").unpack),
  }

  def __init__(self, wave_file, keep_int=False):
    """
    Loads a Wave audio file.

    Parameters
    ----------
    wave_file :
      Wave file name or a already open wave file as a file-behaved object.
    keep_int :
      This flag allows keeping the data on the original range and datatype,
      keeping each sample an int, as stored. False by default, meaning that
      the resulting range is already scaled (but not normalized) to fit
      [-1,1). When True, data scales from ``- (2 ** (bits - 1))`` to
      ``2 ** (bits - 1) - 1`` (signed int), except for 8 bits, where it
      scales from ``0`` to ``255`` (unsigned).
    """
    self._file = wave.open(wave_file, "rb")
    self.rate = self._file.getframerate()
    self.channels = self._file.getnchannels()
    self.bits = 8 * self._file.getsampwidth()

    def data_generator():
      """ Internal wave data generator, given a single sample unpacker """
      w = self._file
      unpacker = WavStream._unpackers[self.bits]
      try:
        if keep_int:

          while True:
            el = w.readframes(1)
            if not el:
              break
            yield unpacker(el)

        else: # Output data should be in [-1;1) range

          d = 1 << (self.bits - 1) # Divide by this number to normalize
          if self.bits == 8: # Unpacker for 8 bits still gives unsigned data
            unpacker = lambda v: ord(v) - 128

          while True:
            el = w.readframes(1)
            if not el:
              break
            yield unpacker(el) / d

      finally:
        w.close()

    super(WavStream, self).__init__(data_generator())
