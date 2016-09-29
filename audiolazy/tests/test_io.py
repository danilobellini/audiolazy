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
Testing module for the lazy_io module
"""

import pytest
p = pytest.mark.parametrize

import pyaudio
import _portaudio
from collections import deque
from time import sleep
import struct

# Audiolazy internal imports
from ..lazy_io import AudioIO, chunks
from ..lazy_synth import white_noise
from ..lazy_stream import Stream
from ..lazy_misc import almost_eq
from ..lazy_compat import orange


class WaitStream(Stream):
  """
  FIFO ControlStream-like class in which ``value`` is a deque object that
  waits a given duration when there's no more data available.
  """

  def __init__(self, duration=.01):
    """ Constructor. Duration in seconds """
    self.value = deque()
    self.active = True

    def data_generator():
      while self.active or self.value:
        try:
          yield self.value.popleft()
        except:
          sleep(duration)

    super(WaitStream, self).__init__(data_generator())


class MockPyAudio(object):
  """
  Fake pyaudio.PyAudio I/O manager class to work with only one output.
  """
  def __init__(self):
    self.fake_output = Stream(0.)
    self._streams = set()
    self.terminated = False

  def terminate(self):
    assert len(self._streams) == 0
    self.terminated = True

  def open(self, **kwargs):
    new_pastream = pyaudio.Stream(self, **kwargs)
    self._streams.add(new_pastream)
    return new_pastream


class MockStream(object):
  """
  Fake pyaudio.Stream class for testing.
  """
  def __init__(self, pa_manager, **kwargs):
    self._pa = pa_manager
    self._stream = self
    self.output = "output" in kwargs and kwargs["output"]
    if self.output:
      pa_manager.fake_output = WaitStream()

  def close(self):
    if self.output: # This is the only output
      self._pa.fake_output.active = False
    self._pa._streams.remove(self)


def mock_write_stream(pa_stream, data, chunk_size, should_throw_exception):
  """
  Fake _portaudio.write_stream function for testing.
  """
  sdata = struct.unpack("{0}{1}".format(chunk_size, "f"), data)
  pa_stream._pa.fake_output.value.extend(sdata)


@p("data", [orange(25), white_noise(100) + 3.])
@pytest.mark.timeout(2)
def test_output_only(monkeypatch, data):
  monkeypatch.setattr(pyaudio, "PyAudio", MockPyAudio)
  monkeypatch.setattr(pyaudio, "Stream", MockStream)
  monkeypatch.setattr(_portaudio, "write_stream", mock_write_stream)

  chunk_size = 16
  data = list(data)
  with AudioIO(True) as player:
    player.play(data, chunk_size=chunk_size)

    played_data = list(player._pa.fake_output)
    ld, lpd = len(data), len(played_data)
    assert all(isinstance(x, float) for x in played_data)
    assert lpd % chunk_size == 0
    assert lpd - ld == -ld % chunk_size
    assert all(x == 0. for x in played_data[ld - lpd:]) # Zero-pad at end
    assert almost_eq(played_data, data) # Data loss (64-32bits conversion)

  assert player._pa.terminated # Test whether "terminate" was called


@p("func", chunks)
class TestChunks(object):

  data = [17., -3.42, 5.4, 8.9, 27., 45.2, 1e-5, -3.7e-4, 7.2, .8272, -4.]
  ld = len(data)
  sizes = [1, 2, 3, 4, ld - 1, ld, ld + 1, 2 * ld, 2 * ld + 1]
  data_segments = (lambda d: [d[:idx] for idx, unused in enumerate(d)])(data)

  @p("size", sizes)
  @p("given_data", data_segments)
  def test_chunks(self, func, given_data, size):
    dfmt="f"
    padval=0.
    data = b"".join(func(given_data, size=size, dfmt=dfmt, padval=padval))
    samples_in = len(given_data)
    samples_out = samples_in
    if samples_in % size != 0:
      samples_out -= samples_in % -size
      assert samples_out > samples_in # Testing the tester...
    restored_data = struct.Struct(dfmt * samples_out).unpack(data)
    assert almost_eq(given_data,
                     restored_data[:samples_in],
                     ignore_type=True)
    assert almost_eq([padval]*(samples_out - samples_in),
                     restored_data[samples_in:],
                     ignore_type=True)

  @p("size", sizes)
  def test_default_size(self, func, size):
    dsize = chunks.size
    assert list(func(self.data)) == list(func(self.data, size=dsize))
    try:
      chunks.size = size
      assert list(func(self.data)) == list(func(self.data, size=size))
    finally:
      chunks.size = dsize
