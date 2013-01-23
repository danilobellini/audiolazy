AudioLazy changes history
-------------------------

*** Version 0.03 (Time variant filters, examples, etc.. Major changes!) ***

+ examples (*new!*):

  - Gammatone frequency and impulse response plots example
  - FM synthesis example for benchmarking between CPython and PyPy
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

  - New window StrategyDict instance, with:

    * Hamming (default)
    * Hann
    * Rectangular
    * Bartlett (triangular with zero endpoints)
    * Triangular (without zeros)
    * Blackman

+ lazy_auditory (*new!*):

  - Two ERB (Equivalent Rectangular Bandwidth) models (Glasberg and Moore)
  - Function to find gammatone bandwidth from ERB for any gammatone order
  - Three gammatone filter implementations: sampled, Slaney, Klapuri

+ lazy_core:

  - MultiKeyDict: an "inversible" dict (i.e., a dict whose values must be
  - StrategyDict: callable dict to store multiple function implementations
    hasheable) that may have several keys for each value
    in. Inherits from MultiKeyDict, so the same strategy may have multiple
    names. It is also an iterable on its values (functions)

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
  - Other functions: ``factorial``, ``ln`` (the `log` from math), ``log2``,
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
  - Faster sinusoid not based on table lookup


*** Version 0.02 (Table lookup synthesis!) ***

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
