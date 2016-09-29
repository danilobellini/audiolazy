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
Audio recording input and playing output module
"""

import threading
import struct
import array

# Audiolazy internal imports
from ._internals import deprecate
from .lazy_stream import Stream
from .lazy_misc import DEFAULT_SAMPLE_RATE, blocks
from .lazy_compat import xrange, xmap
from .lazy_math import inf
from .lazy_core import StrategyDict

__all__ = ["chunks", "RecStream", "AudioIO", "AudioThread"]


# Conversion dict from structs.Struct() format symbols to PyAudio constants
_STRUCT2PYAUDIO = {"f": 1, #pyaudio.paFloat32
                   "i": 2, #pyaudio.paInt32
                   "h": 8, #pyaudio.paInt16
                   "b": 16, #pyaudio.paInt8
                   "B": 32, #pyaudio.paUInt8
                  }


chunks = StrategyDict("chunks")
chunks.__class__.size = 2048 # Samples


@chunks.strategy("struct")
def chunks(seq, size=None, dfmt="f", byte_order=None, padval=0.):
  """
  Chunk generator based on the struct module (Python standard library).

  Low-level data blockenizer for homogeneous data as a generator, to help
  writing an iterable into a file.
  The dfmt should be one char, chosen from the ones in link:

    `<http://docs.python.org/library/struct.html#format-characters>`_

  Useful examples (integer are signed, use upper case for unsigned ones):

  - "b" for 8 bits (1 byte) integer
  - "h" for 16 bits (2 bytes) integer
  - "i" for 32 bits (4 bytes) integer
  - "f" for 32 bits (4 bytes) float (default)
  - "d" for 64 bits (8 bytes) float (double)

  Byte order follows native system defaults. Other options are in the site:

    `<http://docs.python.org/library/struct.html#struct-alignment>`_

  They are:

  - "<" means little-endian
  - ">" means big-endian

  Note
  ----
  Default chunk size can be accessed (and changed) via chunks.size.

  """
  if size is None:
    size = chunks.size
  dfmt = str(size) + dfmt
  if byte_order is None:
    struct_string = dfmt
  else:
    struct_string = byte_order + dfmt
  s = struct.Struct(struct_string)
  for block in blocks(seq, size, padval=padval):
    yield s.pack(*block)


@chunks.strategy("array")
def chunks(seq, size=None, dfmt="f", byte_order=None, padval=0.):
  """
  Chunk generator based on the array module (Python standard library).

  See chunk.struct for more help. This strategy uses array.array (random access
  by indexing management) instead of struct.Struct and blocks/deque (circular
  queue appending) from the chunks.struct strategy.

  Hint
  ----
  Try each one to find the faster one for your machine, and chooses
  the default one by assigning ``chunks.default = chunks.strategy_name``.
  It'll be the one used by the AudioIO/AudioThread playing mechanism.

  Note
  ----
  The ``dfmt`` symbols for arrays might differ from structs' defaults.

  """
  if size is None:
    size = chunks.size
  chunk = array.array(dfmt, xrange(size))
  idx = 0

  for el in seq:
    chunk[idx] = el
    idx += 1
    if idx == size:
      yield chunk.tostring()
      idx = 0

  if idx != 0:
    for idx in xrange(idx, size):
      chunk[idx] = padval
    yield chunk.tostring()


class RecStream(Stream):
  """
  Recording Stream

  A common Stream class with a ``stop`` method for input data recording
  and a ``recording`` read-only property for status.
  """
  def __init__(self, device_manager, file_obj, chunk_size, dfmt):
    if chunk_size is None:
      chunk_size = chunks.size
    s = struct.Struct("{0}{1}".format(chunk_size, dfmt))

    def rec():
      try:
        while self._recording:
          for k in s.unpack(file_obj.read(chunk_size)):
            yield k
      finally:
        file_obj.close()
        self._recording = False # Loop can be broken by StopIteration
        self.device_manager.recording_finished(self)

    super(RecStream, self).__init__(rec())
    self._recording = True
    self.device_manager = device_manager

  def stop(self):
    """ Finishes the recording stream, so it can raise StopIteration """
    self._recording = False

  @property
  def recording(self):
    return self._recording


class AudioIO(object):
  """
  Multi-thread stream manager wrapper for PyAudio.

  """

  def __init__(self, wait=False, api=None):
    """
    Constructor to PyAudio Multi-thread manager audio IO interface.
    The "wait" input is a boolean about the behaviour on closing the
    instance, if it should or not wait for the streaming audio to finish.
    Defaults to False. Only works if the close method is explicitly
    called.
    """
    import pyaudio
    self._pa = pa = pyaudio.PyAudio()
    self._threads = []
    self.wait = wait # Wait threads to finish at end (constructor parameter)
    self._recordings = []

    # Lockers
    self.halting = threading.Lock() # Only for "close" method
    self.lock = threading.Lock() # "_threads" access locking
    self.finished = False

    # Choosing the PortAudio API (needed to use Jack)
    if not (api is None):
      api_count = pa.get_host_api_count()
      apis_gen = xmap(pa.get_host_api_info_by_index, xrange(api_count))
      try:
        self.api = next(el for el in apis_gen
                           if el["name"].lower().startswith(api))
      except StopIteration:
        raise RuntimeError("API '{}' not found".format(api))

  def __del__(self):
    """
    Default destructor. Use close method instead, or use the class
    instance as the expression of a with block.
    """
    self.close()

  def __exit__(self, etype, evalue, etraceback):
    """
    Closing destructor for use internally in a with-expression.
    """
    self.close()

  def __enter__(self):
    """
    To be used only internally, in the with-expression protocol.
    """
    return self

  def close(self):
    """
    Destructor for this audio interface. Waits the threads to finish their
    streams, if desired.
    """
    with self.halting: # Avoid simultaneous "close" threads

      if not self.finished:  # Ignore all "close" calls, but the first,
        self.finished = True # and any call to play would raise ThreadError

        # Closes all playing AudioThread instances
        while True:
          with self.lock: # Ensure there's no other thread messing around
            try:
              thread = self._threads[0] # Needless to say: pop = deadlock
            except IndexError: # Empty list
              break # No more threads

          if not self.wait:
            thread.stop()
          thread.join()

        # Closes all recording RecStream instances
        while self._recordings:
          recst = self._recordings[-1]
          recst.stop()
          recst.take(inf) # Ensure it'll be closed

        # Finishes
        assert not self._pa._streams # No stream should survive
        self._pa.terminate()

  def terminate(self):
    """
    Same as "close".
    """
    self.close() # Avoids direct calls to inherited "terminate"

  def play(self, audio, **kwargs):
    """
    Start another thread playing the given audio sample iterable (e.g. a
    list, a generator, a NumPy np.ndarray with samples), and play it.
    The arguments are used to customize behaviour of the new thread, as
    parameters directly sent to PyAudio's new stream opening method, see
    AudioThread.__init__ for more.
    """
    with self.lock:
      if self.finished:
        raise threading.ThreadError("Trying to play an audio stream while "
                                    "halting the AudioIO manager object")
      new_thread = AudioThread(self, audio, **kwargs)
      self._threads.append(new_thread)
      new_thread.start()
      return new_thread

  def thread_finished(self, thread):
    """
    Updates internal status about open threads. Should be called only by
    the internal closing mechanism of AudioThread instances.
    """
    with self.lock:
      self._threads.remove(thread)

  def recording_finished(self, recst):
    """
    Updates internal status about open recording streams. Should be called
    only by the internal closing mechanism of children RecStream instances.
    """
    self._recordings.remove(recst)

  def record(self, chunk_size = None,
                   dfmt = "f",
                   channels = 1,
                   rate = DEFAULT_SAMPLE_RATE,
                   **kwargs
            ):
    """
    Records audio from device into a Stream.

    Parameters
    ----------
    chunk_size :
      Number of samples per chunk (block sent to device).
    dfmt :
      Format, as in chunks(). Default is "f" (Float32).
    channels :
      Channels in audio stream (serialized).
    rate :
      Sample rate (same input used in sHz).

    Returns
    -------
    Endless Stream instance that gather data from the audio input device.
    """
    if chunk_size is None:
      chunk_size = chunks.size

    if hasattr(self, "api"):
      kwargs.setdefault("input_device_index", self.api["defaultInputDevice"])

    channels = kwargs.pop("nchannels", channels) # Backwards compatibility

    input_stream = RecStream(self,
                             self._pa.open(format=_STRUCT2PYAUDIO[dfmt],
                                           channels=channels,
                                           rate=rate,
                                           frames_per_buffer=chunk_size,
                                           input=True,
                                           **kwargs),
                             chunk_size,
                             dfmt
                            )
    self._recordings.append(input_stream)
    return input_stream


class AudioThread(threading.Thread):
  """
  Audio output thread.

  This class is a wrapper to ease the use of PyAudio using iterables of
  numbers (Stream instances, lists, tuples, NumPy 1D arrays, generators) as
  audio data streams.

  """
  def __init__(self, device_manager, audio,
                     chunk_size = None,
                     dfmt = "f",
                     channels = 1,
                     rate = DEFAULT_SAMPLE_RATE,
                     daemon = True, # This shouldn't survive after crashes
                     **kwargs
              ):
    """
    Sets a new thread to play the given audio.

    Parameters
    ----------
    chunk_size :
      Number of samples per chunk (block sent to device).
    dfmt :
      Format, as in chunks(). Default is "f" (Float32).
    channels :
      Channels in audio stream (serialized).
    rate :
      Sample rate (same input used in sHz).
    daemon :
      Boolean telling if thread should be daemon. Default is True.

    """
    super(AudioThread, self).__init__()
    self.daemon = daemon # threading.Thread property, couldn't be assigned
                         # before the superclass constructor

    # Stores data needed by the run method
    self.audio = audio
    self.device_manager = device_manager
    self.dfmt = dfmt
    self.channels = kwargs.pop("nchannels", channels)
    self.chunk_size = chunks.size if chunk_size is None else chunk_size

    # Lockers
    self.lock = threading.Lock() # Avoid control methods simultaneous call
    self.go = threading.Event() # Communication between the 2 threads
    self.go.set()
    self.halting = False # The stop message

    # Get the streaming function
    import _portaudio # Just to be slightly faster (per chunk!)
    self.write_stream = _portaudio.write_stream

    if hasattr(device_manager, "api"):
      kwargs.setdefault("output_device_index",
                        device_manager.api["defaultOutputDevice"])

    # Open a new audio output stream
    self.stream = device_manager._pa.open(format=_STRUCT2PYAUDIO[dfmt],
                                          channels=channels,
                                          rate=rate,
                                          frames_per_buffer=self.chunk_size,
                                          output=True,
                                          **kwargs)

  # Backwards compatibility
  nchannels = property(deprecate(lambda self: self.channels))

  def run(self):
    """
    Plays the audio. This method plays the audio, and shouldn't be called
    explicitly, let the constructor do so.
    """
    # From now on, it's multi-thread. Let the force be with them.
    st = self.stream._stream

    for chunk in chunks(self.audio,
                        size=self.chunk_size*self.nchannels,
                        dfmt=self.dfmt):
      #Below is a faster way to call:
      #  self.stream.write(chunk, self.chunk_size)
      self.write_stream(st, chunk, self.chunk_size, False)
      if not self.go.is_set():
        self.stream.stop_stream()
        if self.halting:
          break
        self.go.wait()
        self.stream.start_stream()

    # Finished playing! Destructor-like step: let's close the thread
    with self.lock:
      if self in self.device_manager._threads: # If not already closed
        self.stream.close()
        self.device_manager.thread_finished(self)

  def stop(self):
    """ Stops the playing thread and close """
    with self.lock:
      self.halting = True
      self.go.clear()

  def pause(self):
    """ Pauses the audio. """
    with self.lock:
      self.go.clear()

  def play(self):
    """ Resume playing the audio. """
    with self.lock:
      self.go.set()
