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
Testing module for the lazy_wav module
"""

import pytest
p = pytest.mark.parametrize

from tempfile import NamedTemporaryFile
from struct import Struct
import io

# Audiolazy internal imports
from ..lazy_wav import WavStream
from ..lazy_stream import Stream, thub
from ..lazy_misc import almost_eq, DEFAULT_SAMPLE_RATE


uint16pack = Struct("<H").pack
uint32pack = Struct("<I").pack

def riff_chunk(chunk_id, *contents, **kwargs):
  """ Build a bytestring object with the RIFF chunk contents. """
  assert len(chunk_id) == 4
  align = kwargs.pop("align", False)
  assert not kwargs # No other keyword argument available
  joined_contents = b"".join(contents)
  content_size = len(joined_contents)
  parts = [chunk_id, uint32pack(content_size), joined_contents]
  if align and content_size & 1: # Tricky! RIFF is word-aligned
    return b"".join(parts + [b"\x00"])
  else:
    return b"".join(parts)

def wave_data(data, bits=16, channels=1, rate=DEFAULT_SAMPLE_RATE):
  """ Build a bytestring object with the full wave contents in data. """
  bytes_per_sample = (bits + 7) // 8
  block_align = bytes_per_sample * channels
  fmt_chunk = riff_chunk(b"fmt ",
    uint16pack(1), # Wave format code (PCM/uncompressed)
    uint16pack(channels), # Number of channels (mono)
    uint32pack(rate), # Sample rate
    uint32pack(rate * block_align), # Average bytes per second
    uint16pack(block_align), # Block align
    uint16pack(bits), # Bits per sample
  )
  data_chunk = riff_chunk(b"data", data, align=False) # To keep main RIFF size
  return riff_chunk(b"RIFF", b"WAVE", fmt_chunk, data_chunk)


@p("save_as", ["bytes_io", "temp_file_obj", "temp_file_name"])
class TestWavStream(object):

  @pytest.yield_fixture
  def wave_file(self, save_as):
    """
    Fixture that calls the wave_data function with the same inputs, but
    returns a file name or a file-like object to be used by the WavStream
    constructor/initializer.
    """
    if save_as == "bytes_io":
      yield lambda *args, **kwargs: io.BytesIO(wave_data(*args, **kwargs))
    else:
      with NamedTemporaryFile(mode="rb+") as f:

        def fixture_func(*args, **kwargs):
          f.write(wave_data(*args, **kwargs))
          if save_as == "temp_file_obj":
            f.seek(0) # File cursor is at EOF but should be at the beginning
            return f
          elif save_as == "temp_file_name":
            f.flush() # Ensure file is saved on disk
            return f.name
          pytest.fail("Invalid test")

        yield fixture_func

  @p(("bits", "rate", "channels"), [
    (8, 44100, 1),
    (16, 3000, 3),
    (24, 8000, 2),
    (32, 12, 8),
  ])
  def test_load_file_empty(self, bits, rate, channels, wave_file):
    file_data = wave_file(b"", channels=channels, bits=bits, rate=rate)
    wav_stream = WavStream(file_data)
    assert isinstance(wav_stream, Stream)
    assert wav_stream.bits == bits
    assert wav_stream.channels == channels
    assert wav_stream.rate == rate
    assert list(wav_stream) == []

  schema_params = ("bits", "rate", "data", "expected")
  params = [
    {"bits": 8,
     "rate": 8000,
     "data": b"\x08\x7f\x18\xfa\xea\xce\x00",
     "expected": (-120, -1, -104, 122, 106, 78, -128),
    },
    {"bits": 16,
     "rate": 48000,
     "data": b"\x08\x91\xf3\x18\xfa\x82\xe4\x2a\xce\x00",
     "expected": (-0x6ef8, 0x18f3, -0x7d06, 0x2ae4, 0xce),
    },
    {"bits": 24,
     "rate": 12345,
     "data": b"\x63\x91\x36\x40\x10\xb0\xfa\xc6\xd0\x80\x78\xaf\x19\x82\xce",
     "expected": (0x369163, -0x4fefc0, -0x2f3906, -0x508780, -0x317de7),
    },
    {"bits": 32,
     "rate": 87654,
     "data": b"\x1f\x85\x6b\x3e\x7b\x14\xae\xbe\x89\xd2\xde\x3a"
             b"\x6c\x09\x79\xba\x9a\x6d\x41\x19",
     "expected": (0x3e6b851f, -0x4151eb85, 0x3aded289,
                  -0x4586f694, 0x19416d9a),
    },
  ]
  params_table = (lambda schema_params, params:
                    [[doc[k] for k in schema_params] for doc in params]
                 )(schema_params, params)

  @p(schema_params, params_table)
  @p("keep", [True, False, None])
  @p("channels", [1, 2])
  def test_load_file(self, bits, rate, data, expected, keep, wave_file,
                     channels):
    if channels == 2:
      data *= 2 # Just to ensure the number of samples is even ...
      expected *= 2
      rate //= 3 # ... and for fun! =)
    file_data = wave_file(data, bits=bits, rate=rate, channels=channels)
    kwargs = {} if keep is None else dict(keep=keep)
    wav_stream = WavStream(file_data, **kwargs)

    assert isinstance(wav_stream, Stream)
    assert wav_stream.bits == bits
    assert wav_stream.channels == channels
    assert wav_stream.rate == rate
    multiplier = 1 << (wav_stream.bits-1) # Scale factor result was divided by

    if keep:
      dtype = int # Never long
      if bits == 8: # The only unsigned
        min_value = 0
        max_value = 255
        result = list(wav_stream.copy() - 128)
      else:
        min_value = -multiplier - 1
        max_value = multiplier
        result = list(wav_stream.copy())
    else: # Data should be float numbers in the [-1;1) interval
      dtype = float
      min_value = -1
      max_value = 1 - 1 / multiplier
      result = list(wav_stream.copy() * multiplier)

    ws = thub(wav_stream, 3)
    assert all(isinstance(el, dtype) for el in ws)
    assert all(ws >= min_value)
    assert all(ws <= max_value)
    assert almost_eq(result, expected)
