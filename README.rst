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
tools, filters, biologial auditory periphery modeling, among other
functionalities.


Status
""""""

This is the first deploy of a pre-alpha package. For now, you can use already
the audiolazy.lazy_itertools with element-wise operators, and create your own
filters with the lazy_filter module. Some documentation will be included soon.


----

Copyright (C) 2012 Danilo de Jesus da Silva Bellini
- danilo [dot] bellini [at] gmail [dot] com

License is GPLv3. See COPYING.txt for more details.
