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
# Created on Fri Jun 06 00:01:29 2014
# danilo [dot] bellini [at] gmail [dot] com
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

def riff_chunk(chunk_id, *contents):
  """ Build a bytestring object with the RIFF chunk contents """
  assert len(chunk_id) == 4
  joined_contents = b"".join(contents)
  content_size = len(joined_contents)
  parts = [chunk_id, uint32pack(content_size), joined_contents]
  if content_size & 1: # Tricky! RIFF is word-aligned
    return b"".join(parts + [b"\x00"])
  else:
    return b"".join(parts)

def wave_data(data, bits=16, channels=1, rate=DEFAULT_SAMPLE_RATE):
  bytes_per_sample = (bits + 7) // 8
  block_align = bytes_per_sample * channels
  fmt_chunk = riff_chunk(b"fmt ",
    uint16pack(1), # Compression code (PCM/uncompressed)
    uint16pack(channels), # Number of channels (mono)
    uint32pack(rate), # Sample rate
    uint32pack(rate * block_align), # Average bytes per second
    uint16pack(block_align), # Block align
    uint16pack(bits), # Bits per sample
  )
  data_chunk = riff_chunk(b"data", data)
  return riff_chunk(b"RIFF", b"WAVE", fmt_chunk, data_chunk)


class TestWavStream(object):

  @p(("bits", "rate", "channels"), [
    (8, 44100, 1),
    (16, 3000, 3),
    (24, 8000, 2),
    (32, 12, 8),
  ])
  def test_load_file_empty(self, bits, rate, channels):
    file_data = wave_data(b"", channels=channels, bits=bits, rate=rate)
    wav_stream = WavStream(io.BytesIO(file_data))
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
     "expected": [-120, -1, -104, 122, 106, 78, -128],
    },
    {"bits": 16,
     "rate": 48000,
     "data": b"\x08\x91\xf3\x18\xfa\x82\xe4\x2a\xce\x00",
     "expected": [-0x6ef8, 0x18f3, -0x7d06, 0x2ae4, 0xce],
    },
    {"bits": 24,
     "rate": 12345,
     "data": b"\x63\x91\x36\x40\x10\xb0\xfa\xc6\xd0\x80\x78\xaf\x19\x82\xce",
     "expected": [0x369163, -0x4fefc0, -0x2f3906, -0x508780, -0x317de7],
    },
    {"bits": 32,
     "rate": 87654,
     "data": b"\x1f\x85\x6b\x3e\x7b\x14\xae\xbe\x89\xd2\xde\x3a"
             b"\x6c\x09\x79\xba\x9a\x6d\x41\x19",
     "expected": [0x3e6b851f, -0x4151eb85, 0x3aded289,
                  -0x4586f694, 0x19416d9a],
    },
  ]
  params_table = (lambda schema_params, params:
                    [[doc[k] for k in schema_params] for doc in params]
                 )(schema_params, params)

  @p(schema_params, params_table)
  @p("keep_int", [True, False, None])
  @p("save_as", ["bytes_io", "temp_file_obj", "temp_file_name"])
  def test_load_file_1channel(self, bits, rate, data, expected,
                                    keep_int, save_as):
    file_data = wave_data(data, bits=bits, rate=rate)
    kwargs = {} if keep_int is None else dict(keep_int=keep_int)

    def apply_test(*args, **kws):
      wav_stream = WavStream(*args, **kws)
      assert isinstance(wav_stream, Stream)
      assert wav_stream.bits == bits
      assert wav_stream.channels == 1
      assert wav_stream.rate == rate
      multiplier = 1 << (wav_stream.bits - 1)

      if keep_int:
        dtype = int # Never long
        if bits == 8:
          min_value = 0
          max_value = 255
          result = list(wav_stream.copy() - 128)
        else:
          min_value = -multiplier - 1
          max_value = multiplier
          result = list(wav_stream.copy())
      else:
        dtype = float
        min_value = -1
        max_value = 1 - 1 / multiplier
        result = list(wav_stream.copy() * multiplier)

      ws = thub(wav_stream, 3)
      assert all(isinstance(el, dtype) for el in ws)
      assert all(ws >= min_value)
      assert all(ws <= max_value)
      assert almost_eq(result, expected)

    if save_as == "bytes_io":
      apply_test(io.BytesIO(file_data), **kwargs)
    else:
      with NamedTemporaryFile(mode="rb+") as f:
        f.write(file_data)
        if save_as == "temp_file_obj":
          f.seek(0)
          wave_file = f
        elif save_as == "temp_file_name":
          f.flush()
          wave_file = f.name
        else:
          pytest.fail("Invalid test")
        apply_test(wave_file, **kwargs)
