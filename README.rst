AudioLazy
=========

Expressive Digital Signal Processing (DSP) package for Python.

Lazyness and object representation
----------------------------------

There are several tools and packages that let the Python use and
expressiveness look like languages such as MatLab and Octave. However, the
eager evaluation done by most of these tools make it difficult, perhaps
impossible, to use them for real time audio processing. Another difficulty
concerns expressive code creation for audio processing in blocks through
indexes and vectors.

What does it do?
----------------

Prioritizing code expressiveness, clarity and simplicity, without precluding
the lazy evaluation, and aiming to be used together with Numpy, Scipy and
Matplotlib as well as default Python structures like lists and generators,
AudioLazy is a package written in pure Python proposing digital audio signal
processing (DSP), featuring:

- A ``Stream`` class for finite and endless signals representation with
  elementwise operators (auto-broadcast with non-Stream) in a common Python
  iterable container accepting heterogeneous data;
- Strongly sample-based representation (Stream class) with easy conversion
  to block representation using the ``Stream.blocks(size, hop)`` method;
- Sample-based interactive processing with ``ControlStream``;
- ``Streamix`` mixer for iterables given their starting time deltas;
- Multi-thread audio I/O integration with PyAudio;
- Linear filtering with Z-transform filters directly as equations (e.g.
  ``filt = 1 / (1 - .3 * z ** -1)``), including linear time variant filters
  (i.e., the ``a`` in ``a * z ** k`` can be a Stream instance), cascade
  filters (behaves as a list of filters), resonators, etc.. Each
  ``LinearFilter`` instance is compiled just in time when called;
- Zeros and poles plots and frequency response plotting integration with
  MatPlotLib ``pylab``;
- Linear Predictive Coding (LPC) directly to ``ZFilter`` instances, from
  which you can find PARCOR coeffs and LSFs;
- Both sample-based (Zero-cross rate, envelope, moving average, clipping,
  unwrapping) and block-based (Window functions, DFT, autocorrelation, lag
  matrix) analysis and processing tools;
- A simple synthesizer (Table lookup, Karplus-Strong) with processing tools
  (Linear ADSR envelope, fade in/out, fixed duration line stream) and basic
  wave data generation (sinusoid, white noise, impulse);
- Biological auditory periphery modeling (ERB and gammatone filter models);
- Multiple implementation organization as ``StrategyDict`` instances:
  callable dictionaries that allows the same name to have several different
  implementations (e.g. ``erb``, ``gammatone``, ``lowpass``, ``resonator``,
  ``lpc``, ``window``);
- Converters among MIDI pitch numbers, strings like "F#4" and frequencies;
- Polynomials, Stream-based functions from itertools, math, cmath, and more!
  Go try yourself! =)

Installing
----------

The package works both on Linux and on Windows. You can find the last stable
version at `http://pypi.python.org/pypi/audiolazy` and install it with
the usual Python installing mechanism::

  python setup.py install

If you have pip, you can go directly::

  pip install audiolazy

For the *bleeding-edge* version, you can install directly from the github
repository (requires ``git`` for cloning)::

  pip install git+http://github.com/danilobellini/audiolazy

The package doesn't have any strong dependency for its core besides the Python
itself and its standard library, but you might need:

- PyAudio: needed for playing and recording audio (``AudioIO`` class);
- NumPy: needed for doing some maths, such as finding the LSFs from a filter
  or roots from a polynomial;
- MatPlotLib: needed for all default plotting, like in ``LinearFilter.plot``
  method and several examples;
- SciPy (testing only): used as an oracle for LTI filter testing;
- pytest and pytest-cov (testing only): runs test suite and shows code
  coverage status;
- wxPython (example only): used by one example with FM synthesis in an
  interactive GUI;
- Music21 (example only): there's one example that gets the Bach chorals from that package
  corpora for synthesizing and and playing.

Beside examples and tests, only the LinearFilter and CascadeFilter plotting
with ``plot`` and ``zplot`` methods needs MatPlotLib. Also, the routines that
needs NumPy up to now are:

- LinearFilter and CascadeFilter root finding with ``zeros`` and ``poles``
  properties;
- Poly ``roots`` property;
- Some Linear Predictive Coding (``lpc``) strategies: ``nautocor``,
  ``autocor`` and ``covar``;
- Line Spectral Frequencies ``lsf`` and ``lsf_stable`` functions.

Getting started
---------------

Before all examples below, it's easier to get everything from audiolazy
namespace:

.. code-block:: python

  from audiolazy import *

All modules starts with "lazy\_", but their data is already loaded in the main
namespace. These two lines of code do the same thing:

.. code-block:: python

  from audiolazy.lazy_stream import Stream
  from audiolazy import Stream

Endless iterables with operators (be careful with loops through an endless
iterator!):

.. code-block:: python

  >>> a = Stream(2) # Periodic
  >>> b = Stream(3, 7, 5, 4) # Periodic
  >>> c = a + b # Elementwise sum, periodic
  >>> c.take(15) # First 15 elements from the Stream object
  [5, 9, 7, 6, 5, 9, 7, 6, 5, 9, 7, 6, 5, 9, 7]

And also finite iterators (you can think on any Stream as a generator with
elementwise operators):

.. code-block:: python

  >>> a = Stream([1, 2, 3, 2, 1]) # Finite, since it's a cast from an iterable
  >>> b = Stream(3, 7, 5, 4) # Periodic
  >>> c = a + b # Elementwise sum, finite
  >>> list(c)
  [4, 9, 8, 6, 4]

LTI Filtering from system equations (Z-transform). After this, try summing,
composing, multiplying ZFilter objects:

.. code-block:: python

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

LTI Filter frequency response plot (needs MatPlotLib):

.. code-block:: python

  (1 + z ** -2).plot().show()

.. image:: images/filt_plot.png

CascadeFilters are lists of filters with the same operator behaviour as a
list, and also works for plotting linear filters. For example, a zeros and
poles plot (needs MatPlotLib):

.. code-block:: python

  filt1 = CascadeFilter(0.2 - z ** -3) # 3 zeros
  filt2 = CascadeFilter(1 / (1 -.8 * z ** -1 + .6 * z ** -2)) # 2 poles
  # Here __add__ concatenates and __mul__ by an integer make reference copies
  filt = (filt1 * 5 + filt2 * 10) # 15 zeros and 20 poles
  filt.zplot().show()

.. image:: images/cascade_plot.png

Linear Predictive Coding (LPC) autocorrelation method analysis filter
frequency response plot (needs MatPlotLib):

.. code-block:: python

  lpc([1, -2, 3, -4, -3, 2, -3, 2, 1], order=3).plot().show()

.. image:: images/lpc_plot.png

Linear Predictive Coding covariance method analysis and synthesis filter,
followed by the frequency response plot together with block data DFT
(MatPlotLib):

.. code-block:: python

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
  >>> amplified_blk = list(Stream(blk) * -200) # For alignment w/ DFT
  >>> synth_filt.plot(blk=amplified_blk).show()

.. image:: images/dft_lpc_plot.png

AudioLazy doesn't need any audio card to process audio, but needs PyAudio to
play some sound:

.. code-block:: python

  rate = 44100 # Sampling rate, in samples/second
  s, Hz = sHz(rate) # Seconds and hertz
  ms = 1e-3 * s
  note1 = karplus_strong(440 * Hz) # Pluck "digitar" synth
  note2 = zeros(300 * ms).append(karplus_strong(880 * Hz))
  notes = note1 + note2
  sound = notes.take(int(2 * s)) # 2 seconds of a Karplus-Strong note
  with AudioIO(True) as player: # True means "wait for all sounds to stop"
    player.play(sound, rate=rate)

See also the docstrings and the "examples" directory at the github repository
for more help. Also, the huge test suite might help you understanding how the
package works and how to use it.

----

Copyright (C) 2012 Danilo de Jesus da Silva Bellini
- danilo [dot] bellini [at] gmail [dot] com

License is GPLv3. See COPYING.txt for more details.
