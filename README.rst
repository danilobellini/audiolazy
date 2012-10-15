AudioLazy
=========

Expressive Digital Signal Processing (DSP) package for Python.


Lazyness
""""""""

There are several tools and packages that let the Python use and
expressiveness look like languages such as MatLab and Octave. However, the
eager evaluation done by most of these tools make it difficult, perhaps
impossible, to use them for real time audio processing. Another difficulty
concerns expressive code creation for audio processing in blocks through
indexes and vectors.


Goal
""""

Prioritizing code expressiveness, clarity and simplicity, without precluding
the lazy evaluation, and aiming to be used together with Numpy, Scipy and
Matplotlib as well as default Python structures like lists and generators,
AudioLazy is a starting project written in pure Python proposing digital
audio signal processing (DSP), featuring a simple synthesizer, analysis
tools, filters, biological auditory periphery modeling, among other
functionalities.


Status
""""""

This is a pre-alpha package. For now, you can do some lazy element-wise maths
with Stream objects; use the lazy_itertools module, a itertools decorated
replica that have element-wise operators; create your own LTI filters using
the filter equation directly on Z-domain with the lazy_filter.z object; use
the lazy_synth module to do some synthesis with TableLookup objects, like FM
synthesis calling the sinusoid twice; play your sound with a AudioIO instance,
perhaps seem as a context manager, with its play method.

Some documentation will be included soon.


----

Copyright (C) 2012 Danilo de Jesus da Silva Bellini
- danilo [dot] bellini [at] gmail [dot] com

License is GPLv3. See COPYING.txt for more details.
