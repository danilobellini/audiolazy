..
  This file is part of AudioLazy, the signal processing Python package.
  Copyright (C) 2012 Danilo de Jesus da Silva Bellini

  AudioLazy is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, version 3 of the License.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.

  danilo [dot] bellini [at] gmail [dot] com

AudioLazy changes history
-------------------------

*** Version 0.04 (Documentation, LPC, Plots!) ***

+ examples:

  - Random Bach Choral playing example (needs Music21 corpus)

+ general:

  - Sphinx documentation!
  - Self-generated package and module summary at the docstring
  - Integration with NumPy (tested on 1.5.0, 1.6.1 and 1.6.2) and MatPlotLib
    (tested on 1.0.1 and 1.2.0)
  - More docstrings and doctests, besides lots of corrections
  - Itemized package description, installation instructions and getting
    started examples with plots in README.rst
  - Now with 5400+ tests and 75% code coverage

+ lazy_analysis:

  - One-dimensional autocorrelation function with ``acorr`` and lag
    "covariance" (due to lpc.covar) with ``lag_matrix``
  - DFT for any frequency, given a block
  - Three envelope filtering strategies (time domain)
  - Three moving average filter strategies
  - Signal clipping function
  - Signal unwrap, defaults to the ``2 * pi`` radians range but configurable
    to other units and max signal difference allowed
  - New AMDF algorithm as a non-linear filter

+ lazy_core:

  - StrategyDict instances now are singletons of a new class, which have
    lazy non-memoized docstrings based on their contents

+ lazy_filters:

  - ZFilter composition/substitution, e.g., ``(1 + z ** -1)(1 / z)`` results
    to the ZFilter instance ``1 + z``
  - New LinearFilter.plot() directly plots the frequency response of a LTI
    filter to a MatPlotLib figure. Configurable:

    * Linear (default) or logarithmic frequency scale
    * Linear, squared or dB (default) magnitude scale
    * Plots together the DFT of a given block, if needed. Useful for LPC
    * Phase unwrapping (defaults to True)
    * Allows frequency in Hz and in rad/sample. When using radians units,
      the tick locator is based on ``pi``, as well as the formatter

  - New LinearFilter.zplot() for plotting the zero-pole plane of a LTI filter
    directly into a MatPlotLib figure
  - New LinearFilterProperties read-only properties ``numpolyz`` and
    ``denpolyz`` returning polynomials based on ``x = z`` instead of the
    polynomials based on ``x = z ** -1`` returned from ``numpoly`` and
    ``denpoly``
  - New LinearFilter properties ``poles`` and ``zeros``, based on NumPy
  - New class ``FilterList`` for filter grouping with a ``callables``
    property, for casting from lists with constant gain values as filters.
    It is an instance of ``FilterListMeta`` (old CascadeFilterMeta), and
    CascadeFilter now inherits from this FilterList
  - More LinearFilter behaviour into FilterList: Plotting (``plot`` and
    ``zplot``), ``poles``, ``zeros``, ``is_lti`` and ``is_causal``
  - New ``ParallelFilter`` class, inheriting from FilterList
  - Now comb is a StrategyDict too, with 3 strategies:

    * ``comb.fb`` (default): Feedback comb filter (IIR or time variant)
    * ``comb.tau``: Same to the feedback strategy, but with a time decay
      ``tau`` parameter (time in samples up to ``1/e`` amplitude, or
      -8.686 dB) instead of a gain ``alpha``
    * ``comb.ff``: Feed-forward comb filter (FIR or time variant)

+ lazy_lpc (*new!*):

  - Linear Predictive Coding (LPC) coefficients as a ZFilter from:

    * ``lpc.autocor`` (default): Auto-selects autocorrelation implementation
      (Faster)
    * ``lpc.nautocor``: Autocorrelation, with linear system solved by NumPy
      (Safer)
    * ``lpc.kautocor``: Autocorrelation, using the Levinson-Durbin algorithm
    * ``lpc.covar`` or ``lpc.ncovar``: Covariance, with linear system solved
      by NumPy
    * ``lpc.kcovar``: Covariance, slower. Mainly for those without NumPy
    * ``levinson_durbin``: Same to the ``lpc.kautocor``, but with the
      autocorrelation vector as the input, not the signal data

  - Toeplitz matrix as a list of lists
  - Partial correlation coefficients (PARCOR) or reflection coefficients
  - Line Spectral Frequencies (LSF)
  - Stability testers for filters with LSF and PARCOR

+ lazy_math:

  - New ``sign`` gets the sign of a given sequence.

+ lazy_midi:

  - Completed converters between frequency (in hertz), string and MIDI pitch
    numbers
  - New ``octaves`` for finding all octaves in a frequency range given one
    frequency

+ lazy_misc:

  - New ``rational_formatter``: casts floats to strings, perhaps with a symbol
    string as multiplier
  - New ``pi_formatter``: same to ``rational_formatter``, but with the symbol
    fixed to pi, mainly for use in MatPlotLib labels

+ lazy_poly:

  - New Poly.roots property, based on NumPy

+ lazy_stream:

  - Streamix class for mixing Streams based on delta starting times,
    automatically managing the need for multiple "tracks"

+ lazy_synth:

  - Karplus-Strong algorithm now uses ``tau`` time decay constant instead of
    the comb filter ``alpha`` gain.


*** Version 0.03 (Time variant filters, examples, etc.. Major changes!) ***

+ examples (*new!*):

  - Gammatone frequency and impulse response plots example
  - FM synthesis example for benchmarking with CPython and PyPy
  - Simple I/O wire example, connecting the input directly to the output
  - Modulo Counter graphics w/ FM synthesis audio in a wxPython application
  - Window functions plot example (all window strategies)

+ general:

  - Namespace cleanup with __all__
  - Lots of optimization and refactoring, also on tests and setup.py
  - Better docstrings and README.rst
  - Doctests (with pytest) and code coverage (needs pytest-cov)
  - Now with 5200+ tests and 79% code coverage

+ lazy_analysis (*new!*):

  - New ``window`` StrategyDict instance, with:

    * Hamming (default)
    * Hann
    * Rectangular
    * Bartlett (triangular with zero endpoints)
    * Triangular (without zeros)
    * Blackman

+ lazy_auditory (*new!*):

  - Two ERB (Equivalent Rectangular Bandwidth) models (both by Glasberg and
    Moore)
  - Function to find gammatone bandwidth from ERB for any gammatone order
  - Three gammatone filter implementations: sampled impulse response, Slaney,
    Klapuri

+ lazy_core:

  - MultiKeyDict: an "inversible" dict (i.e., a dict whose values must be
    hasheable) that may have several keys for each value
  - StrategyDict: callable dict to store multiple function implementations
    in. Inherits from MultiKeyDict, so the same strategy may have multiple
    names. It's also an iterable on its values (functions)

+ lazy_filters:

  - LTI and LTIFreq no longer exists! They were renamed to LinearFilter and
    ZFilter since filters now can have Streams as coefficients (they don't
    need to be "Time Invariant" anymore)
  - Linear filters are now iterables, allowing:

    * Comparison with almost_eq like ``assert almost_eq(filt1, filt2)``
    * Expression like ``numerator_data, denominator_data = filt``, where
      each data is a list of pairs that can be used as input for Poly,
      LinearFilter or ZFilter

  - LinearFilterProperties class, implementing numlist, denlist, numdict and
    dendict, besides numerator and denominator, from numpoly and denpoly
  - Comparison "==" and "!=" are now strict
  - CascadeFilter: list of filters that behave as a filter
  - LinearFilter.__call__ now has the "zero" optional argument (allows
    non-float)
  - LinearFilter.__call__ memory input can be a function or a Stream
  - LinearFilter.linearize: linear interpolated delay-line from fractional
    delays
  - Feedback comb filter
  - 4 resonator filter models with 2-poles with exponential approximation
    for finding the radius from the bandwidth
  - Simple one pole lowpass and highpass filters

+ lazy_io:

  - AudioIO.record method, creating audio Stream instances from device data

+ lazy_itertools:

  - Now with a changed tee function that allows not-iterable inputs,
    helpful to let the same code work with Stream instances and constants

+ lazy_math (*new!*):

  - dB10, dB20 functions for converting amplitude (squared or linear,
    respectively) to logarithmic dB (power) values from complex-numbers
    (like the ones returned by LinearFilter.freq_response)
  - Most functions from math module, but working decorated with elementwise
    (``sin``, ``cos``, ``sqrt``, etc.), and the constants ``e`` and ``pi``
  - Other functions: ``factorial``, ``ln`` (the ``log`` from math), ``log2``,
    ``cexp`` (the ``exp`` from cmath) and ``phase`` (from cmath)

+ lazy_midi:

  - MIDI pitch numbers and Hz frequency converters from strings like "C#4"

+ lazy_misc:

  - Elementwise decorator now based on both argument keyword and position

+ lazy_poly:

  - Horner-like scheme for Poly.__call__ evaluation
  - Poly now can have Streams as coefficients
  - Comparison "==" and "!=" are now strict

+ lazy_stream:

  - Methods and attributes from Stream elements can be used directly,
    elementwise, like ``my_stream.imag`` and ``my_stream.conjugate()`` in a
    stream with complex numbers
  - New thub() function and StreamTeeHub class: tee (or "T") hub auto-copier
    to help working with Stream instances *almost* the same way as you do with
    numbers

+ lazy_synth:

  - Karplus-Strong synthesis algorithm
  - ADSR envelope
  - Impulse, ones, zeros/zeroes and white noise Stream generator
  - Faster sinusoid not based on the TableLookup class


*** Version 0.02 (Interactive Stream objects & Table lookup synthesis!) ***

+ general:

  - 10 new tests

+ lazy_midi (*new!*):

  - MIDI to frequency (Hz) conversor

+ lazy_misc:

  - sHz function for explicit time (s) and frequency (Hz) units conversion

+ lazy_stream:

  - Interactive processing with ControlStream instances
  - Stream class now allows inheritance

+ lazy_synth (*new!*):

  - TableLookup class, with sinusoid and sawtooth instances
  - Endless counter with modulo, allowing Stream inputs, mainly created for
    TableLookup instances
  - Line, fade in, fade out, ADS attack with endless sustain


*** Version 0.01 (First "pre-alpha" version!) ***

+ general:

  - 4786 tests (including parametrized tests), based on pytest

+ lazy_core:

  - AbstractOperatorOverloaderMeta class to help massive operator
    overloading as needed by Stream, Poly and LTIFreq (now ZFilter) classes

+ lazy_filters:

  - LTI filters, callable objects with operators and derivatives, returning
    Stream instances
  - Explicit filter formulas with the ``z`` object, e.g.
    ``filt = 1 / (.5 + z ** -1)``

+ lazy_io:

  - Multi-thread audio playing (based on PyAudio), with context manager
    interface

+ lazy_itertools:

  - Stream-based version of all itertools

+ lazy_misc:

  - Block-based processing, given size and (optionally) hop
  - Simple zero padding generator
  - Elementwise decorator for functions
  - Bit-based and diff-based "almost equal" comparison function for floats
    and iterables with floats. Also works with (finite) generators

+ lazy_poly:

  - Poly: polynomials based on dictionaries, with list interface and
    operators

+ lazy_stream:

  - Stream: each instance is basically a generator with elementwise
    operators
  - Decorator ``tostream`` so generator functions can return Stream objects
