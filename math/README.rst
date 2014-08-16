..
  This file is part of AudioLazy, the signal processing Python package.
  Copyright (C) 2012-2014 Danilo de Jesus da Silva Bellini

  AudioLazy is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, version 3 of the License.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.

  Created on Thu Aug 14 23:51:16 2014
  danilo [dot] bellini [at] gmail [dot] com

AudioLazy Math
==============

This directory have scripts with some symbolic math calculations that should
be seen as a proof, or perhaps as a helper on understanding, for some
equations written as part of the AudioLazy code. For such a symbolic
processing, the `Sympy <http://sympy.org/>`__ CAS (Computer Algebra System)
Python package was used.

Originally, some of these results were done manually, but all were needed
for designing some part of AudioLazy. Regardless of their difficulty,
the blocks implemented in AudioLazy just use the results from here, and
doesn't require Sympy to work.


Running
-------

They're all scripts, you should just run them or call Python. Works on both
Python 2 and 3, but be sure to have Sympy installed before that::

  pip install scipy

So you can run them directly like you would with an AudioLazy example, with
one of the following lines::

  ./script_name.py
  python script_name.py
  python3 script_name.py


Proofs
------

* `lowpass_highpass_pole_z.py <lowpass_highpass_pole_z.py>`__

  Includes the digital filter design of the ``lowpass.pole``, ``highpass.z``,
  ``highpass.pole`` and ``lowpass.z`` strategies.
