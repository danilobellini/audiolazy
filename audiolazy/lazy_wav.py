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
  * ``channels``: number of channels (1 for mono, 2 for stereo);
  * ``bits``: bits per sample, a value in ``[8, 16, 24, 32]``.

  Example
  -------

  .. code-block:: python

    song = WavStream("my_song.wav")
    with AudioIO(True) as player:
      player.play(song, rate=song.rate, channels=song.channels)

  Note
  ----
  Stereo data is kept serialized/flat, so the resulting Stream yields first a
  sample from one channel, then the sample from the other channel for that
  same time instant. Use ``Stream.blocks(2)`` to get a Stream with the stereo
  blocks.
  """
  _unpackers = {
    8 : ord, # The only unsigned
    16: (lambda a: lambda v: a(v)[0])(Struct("<h").unpack),
    24: (lambda a: lambda v: a(b"\x00" + v)[0] >> 8)(Struct("<i").unpack),
    32: (lambda a: lambda v: a(v)[0])(Struct("<i").unpack),
  }

  def __init__(self, wave_file, keep=False):
    """
    Loads a Wave audio file.

    Parameters
    ----------
    wave_file :
      Wave file name or a already open wave file as a file-behaved object.
    keep :
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

    def block_reader():
      """ Raw wave data block generator (following block align) """
      w = self._file
      try:
        while True:
          el = w.readframes(1)
          if not el:
            break
          yield el
      finally:
        w.close()

    def sample_reader():
      """ Raw wave data single sample generator (1 or 2 per block) """
      # Mono
      if self.channels == 1:
        return block_reader()

      # Stereo
      sample_width = self.bits // 8
      def stereo_sample_reader():
        for el in block_reader():
          yield el[:sample_width]
          yield el[sample_width:]

      return stereo_sample_reader()

    def data_generator():
      """ Wave data generator with data already converted to float or int """
      unpacker = WavStream._unpackers[self.bits]

      if keep:

        for el in sample_reader():
          yield unpacker(el)

      else: # Output data should be in [-1;1) range

        d = 1 << (self.bits - 1) # Divide by this number to normalize
        if self.bits == 8: # Unpacker for 8 bits still gives unsigned data
          unpacker = lambda v: ord(v) - 128

        for el in sample_reader():
          yield unpacker(el) / d

    super(WavStream, self).__init__(data_generator())
