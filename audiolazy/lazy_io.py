#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Audio recording input and playing output module

Copyright (C) 2012 Danilo de Jesus da Silva Bellini

This file is part of AudioLazy, the signal processing Python package.

AudioLazy is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

Created on Fri Jul 20 2012
danilo [dot] bellini [at] gmail [dot] com
"""

# PyAudio PortAudio bindings
try:
  import pyaudio
  import _portaudio # Just to be slightly faster (per chunk!)

  # Conversion dict from structs.Struct() format symbols to PyAudio constants
  _STRUCT2PYAUDIO = {"f": pyaudio.paFloat32,
                     "i": pyaudio.paInt32,
                     "h": pyaudio.paInt16,
                     "b": pyaudio.paInt8,
                     "B": pyaudio.paUInt8,
                    }
except:
  print "PyAudio not found: can't use audio I/O"
  pass

import threading

# Audiolazy internal imports
from .lazy_misc import chunks, DEFAULT_CHUNK_SIZE, DEFAULT_SAMPLE_RATE


class AudioIO(object):
  """
    Multi-thread stream manager wrapper for PyAudio.
  """

  def __init__(self, wait=False):
    """
      Constructor to PyAudio Multi-thread manager audio IO interface.
      The "wait" input is a boolean about the behaviour on closing the
      instance, if it should or not wait for the streaming audio to finish.
      Defaults to False. Only works if the close method is explicitly
      called.
    """
    self._pa = pyaudio.PyAudio()
    self._threads = []
    self.wait = wait # Wait threads to finish at end (constructor parameter)

    # Lockers
    self.halting = threading.Lock() # Only for "close" method
    self.lock = threading.Lock() # "_threads" access locking
    self.finished = False

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

        while True:
          with self.lock: # Ensure there's no other thread messing around
            try:
              thread = self._threads[0] # Needless to say: pop = deadlock
            except IndexError: # Empty list
              break # No more threads

          if not self.wait:
            thread.stop()
          thread.join()

        assert not self._pa._streams # No stream should survive.
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
      the internal closing mechanism of children threads.
    """
    with self.lock:
      self._threads.remove(thread)


class AudioThread(threading.Thread):
  """
    This class is a wrapper to ease the use of PyAudio using iterables of
    numbers (lists, tuples, NumPy 1D arrays or, better, generators) as audio
    data streams.
  """
  def __init__(self, device_manager, audio,
                     chunk_size = DEFAULT_CHUNK_SIZE,
                     dfmt = "f",
                     nchannels = 1,
                     rate = DEFAULT_SAMPLE_RATE,
                     daemon = True # This shouldn't survive after crashes
              ):
    """
      Sets a new thread to play the given audio.
      Possible keyword arguments:
        chunk_size   - Number of samples per chunk (block sent to device).
        dfmt         - Format, as in chunks(). Default is "f" (Float32).
        num_channels - Channels in audio stream (serialized).
        rate         - Sample rate.
        daemon       - Thread should be daemon.
    """
    super(AudioThread, self).__init__()
    self.daemon = daemon # threading.Thread property, couldn't be assigned
                         # before the superclass constructor

    # Stores data needed by the run method
    self.audio = audio
    self.device_manager = device_manager
    self.dfmt = dfmt
    self.nchannels = nchannels
    self.chunk_size = chunk_size

    # Lockers
    self.lock = threading.Lock() # Avoid control methods simultaneous call
    self.go = threading.Event() # Communication between the 2 threads
    self.go.set()
    self.halting = False # The stop message

    # Open a new audio output stream
    self.stream = device_manager._pa.open(format=_STRUCT2PYAUDIO[dfmt],
                                          channels=nchannels,
                                          rate=rate,
                                          frames_per_buffer=chunk_size,
                                          output=True)

  def run(self):
    """
      Plays the audio. This method plays the audio, and shouldn't be called
      explicitly, let the constructor do so.
    """
    # From now on, it's multi-thread. Let the force be with them.
    st = self.stream._stream
    for chunk in chunks(self.audio, size=self.chunk_size*self.nchannels, dfmt=self.dfmt):
      #Below is a faster way to call:
      #  self.stream.write(chunk, self.chunk_size)
      _portaudio.write_stream(st, chunk, self.chunk_size, False)
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
