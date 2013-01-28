AudioLazy
=========

Expressive Digital Signal Processing (DSP) package for Python.

Lazyness
--------

There are several tools and packages that let the Python use and
expressiveness look like languages such as MatLab and Octave. However, the
eager evaluation done by most of these tools make it difficult, perhaps
impossible, to use them for real time audio processing. Another difficulty
concerns expressive code creation for audio processing in blocks through
indexes and vectors.

Goal
----

Prioritizing code expressiveness, clarity and simplicity, without precluding
the lazy evaluation, and aiming to be used together with Numpy, Scipy and
Matplotlib as well as default Python structures like lists and generators,
AudioLazy is a package written in pure Python proposing digital
audio signal processing (DSP), featuring a simple synthesizer, analysis
tools, filters, biological auditory periphery modeling, among other
functionalities.

Installing
----------

The package works both on Linux and on Windows. With the file below you can
install it with the usual Python installing mechanism:

   $ python setup.py install

If you have pip, you can go directly:

   $ pip install audiolazy

For the *bleeding-edge* version, you can install directly from the github
repository (requires ``git`` for cloning):

   $ pip install git+http://github.com/danilobellini/audiolazy

The package doesn't have any strong dependency for its core besides the Python
itself and its standard library, but you might need:

- PyAudio: needed for playing and recording audio
- NumPy: needed for doing some maths, such as finding the LSFs from a filter
- MatPlotLib: needed for all default plotting, like in LinearFilter.plot
  method and several examples
- wxPython: used by one example with GUI
- Music21: Bach chorals from its corpora, used in a synthesis and play example


Getting started
---------------

Before all examples below, it's easier to get everything from audiolazy
namespace:

  >>> from audiolazy import *

All modules starts with "lazy_", but their data is already loaded in the main
namespace. These two lines of code do the same thing:

  >>> from audiolazy.lazy_stream import Stream
  >>> from audiolazy import Stream

Endless iterables with operators (be careful with loops through an endless
iterator!):

  >>> a = Stream(2) # Periodic
  >>> b = Stream(3, 7, 5, 4) # Periodic
  >>> c = a + b # Elementwise sum, periodic
  >>> c.take(15) # First 15 elements from the Stream object
  [5, 9, 7, 6, 5, 9, 7, 6, 5, 9, 7, 6, 5, 9, 7]

And also finite iterators (you can think on any Stream as a generator with
elementwise operators):

  >>> a = Stream([1, 2, 3, 2, 1]) # Finite, since it's a cast from an iterable
  >>> b = Stream(3, 7, 5, 4) # Periodic
  >>> c = a + b # Elementwise sum, finite
  >>> list(c)
  [4, 9, 8, 6, 4]

LTI Filtering from system equations (Z-transform). After this, try summing,
composing, multiplying ZFilter objects.

  >>> filt = 1 - z ** -1 # Diff between a sample and the previous one
  >>> filt
  1 - z^-1
  >>> data = filt([.1, .2, .4, .3, .2, -.1, -.3, -.2]) # Past memory has 0.0
  >>> data # This should have internally [.1, .1, .2, -.1, -.1, -.3, -.2, .1]
  <audiolazy.lazy_stream.Stream object at ...>
  >>> data *= 10 # Elementwise gain
  >>> [int(x) for x in data] # Streams are iterables
  [1, 1, 2, -1, -1, -3, -2, 1]
  >>> data_int = filt([1, 2, 4, 3, 2, -1, -3, -2], zero=0) # Now zero is int
  >>> list(data_int)
  [1, 1, 2, -1, -1, -3, -2, 1]

The AudioLazy core doesn't depend on NumPy, SciPy nor MatPlotLib, but there
are some parts of it that needs them. Below are some examples:

LTI Filter frequency response plot (needs MatPlotLib):

  >>> (1 + z ** -2).plot().show()

Linear Predictive Coding (LPC) autocorrelation method analysis filter
frequency response plot (MatPlotLib):

  >>> lpc([1, -2, 3, -4, -3, 2, -3, 2, 1], order=3).plot().show()

Linear Predictive Coding covariance method analysis and synthesis filter,
followed by the frequency response plot together with block data DFT
(MatPlotLib):

  >>> data = Stream(-1., 0., 1., 0.) # Periodic
  >>> blk = data.take(200)
  >>> analysis_filt = lpc.covar(blk, 4)
  >>> analysis_filt
  1 + 0.5 * z^-2 - 0.5 * z^-4
  >>> residual = list(analysis_filt(blk))
  >>> residual[:10]
  [-1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  >>> synth_filt = 1 / analysis_filt
  >>> synth_filt(residual).take(10)
  [-1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]
  >>> gain_rms = sqrt(analysis_filt.error)
  >>> amplified_blk = list(Stream(blk) * 200) # Just for alignment w/ DFT gain
  >>> synth_filt.plot(blk=amplified_blk).show()

AudioLazy doesn't need any audio card to process audio, but needs PyAudio to
play some sound:

  >>> rate = 44100 # Sampling rate, in samples/second
  >>> s, Hz = sHz(rate) # Seconds and hertz
  >>> ms = 1e-3 * s
  >>> note1 = karplus_strong(440 * Hz) # Pluck "digitar" synth
  >>> note2 = zeros(300 * ms).append(karplus_strong(880 * Hz))
  >>> notes = note1 + note2
  >>> sound = notes.take(int(2 * s)) # 2 seconds of a Karplus-Strong note
  >>> with AudioIO(True) as player: # True means "wait for all sounds to stop"
  ...   player.play(sound, rate=rate)

See also the docstrings and the "examples" directory at the github repository
for more help. Also, the huge test suite might help you understanding how the
package works and how to use it.

----

Copyright (C) 2012 Danilo de Jesus da Silva Bellini
- danilo [dot] bellini [at] gmail [dot] com

License is GPLv3. See COPYING.txt for more details.
