AudioLazy changes history
-------------------------

*** Development... ***

- lazy_auditory (*new!*):
    - Two ERB (Equivalent Rectangular Bandwidth) models (Glasberg and Moore)
    - Function to find gammatone bandwidth from ERB for any gammatone order
    - Three gammatone filter implementations: sampled, Slaney, Klapuri
- lazy_core:
    - MultiKeyDict: an "inversible" dict (i.e., a dict whose values must be
      hasheable) that may have several keys for each value
    - StrategyDict: callable dict to store multiple function implementations
      in. Inherits from MultiKeyDict, so the same strategy may have multiple
      names. It is also an iterable on its values (functions)
- lazy_synth:
    - Karplus-Strong synthesis algorithm
    - ADSR envelope
    - Impulse, ones, zeros/zeroes and white noise Stream generator
    - Faster sinusoid not based on table lookup
- lazy_filters:
    - CascadeFilter: list of filters that behave as a filter
    - LTIFreq.__call__ now has the "zero" optional argument (allows non-float)
    - LTIFreq.__call__ memory input can be a function or a Stream
    - LTI.linearize: linear interpolated delay-line from fractional delays
    - Feedback comb filter
    - 4 resonator filter models with 2-poles based on exponential bandwidth
- lazy_stream:
    - Methods and attributes from Stream elements can be used directly,
      elementwise, like ``my_stream.imag`` and ``my_stream.conjugate()`` in a
      stream with complex numbers
- lazy_poly:
    - Horner-like scheme for Poly.__call__ evaluation
- lazy_midi:
    - MIDI pitch numbers and Hz frequency converters from strings like "C#4"
- lazy_misc:
    - dB10, dB20 functions for converting amplitude (squared or linear,
      respectively) to logarithmic dB (power) values. Complex-numbers (like
      the ones returned by LTIFreq.freq_response)
    - Elementwise decorator now based on both argument keyword and position
- examples (new):
    - Gammatone frequency and impulse response plots example
    - FM synthesis example for benchmarking between CPython and PyPy
- general:
    - Namespace cleanup with __all__
    - Lots of optimization and refactoring, also on tests and setup.py
    - Better docstrings and README.rst
    - Doctests (with pytest) and code coverage (needs pytest-cov)
    - Now with 4865+ tests and 68% code coverage


*** Version 0.02 (Table lookup synthesis!) ***

- lazy_synth (*new!*):
    - TableLookup class, with sinusoid and sawtooth instances
    - Endless counter with modulo, allowing Stream inputs, mainly created for
      TableLookup instances
    - Line, fade in, fade out, ADS attack with endless sustain
- lazy_midi (*new!*):
    - MIDI to frequency (Hz) conversor
- lazy_stream:
    - Interactive processing with ControlStream instances
    - Stream class now allows inheritance
- lazy_misc:
    - sHz function for explicit time (s) and frequency (Hz) units conversion
- general:
    - 10 new tests


*** Version 0.01 (First "pre-alpha" version!) ***

- lazy_stream:
    - Stream: each instance is basically a generator with elementwise
      operators
    - Decorator `tostream` so generator functions can return Stream objects
- lazy_poly:
    - Poly: polynomials based on dictionaries, with list interface and
      operators
- lazy_filters:
    - LTI filters, callable objects with operators and derivatives, returning
      Stream instances
    - Explicit filter formulas with the `z` object, e.g.
      ``filt = 1 / (.5 + z ** -1)``
- lazy_io:
    - Multi-thread audio playing (based on PyAudio), with context manager
      interface
- lazy_itertools:
    - Stream-based version of all itertools
- lazy_misc:
    - Block-based processing, given size and (optionally) hop
    - Simple zero padding generator
    - Elementwise decorator for functions
    - Bit-based and diff-based "almost equal" comparison function for floats
      and iterables with floats. Also works with (finite) generators
- lazy_core:
    - AbstractOperatorOverloaderMeta class to help massive operator
      overloading as needed by Stream, Poly and LTIFreq classes
- general:
    - 4786 tests (including parametrized tests), based on pytest