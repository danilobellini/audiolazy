#!/usr/bin/env python
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
Matplotlib animated plot with mic input data.

Call with the API name like ...
  ./animated_plot.py jack
... or without nothing for the default PortAudio API.
"""
from __future__ import division
from audiolazy import sHz, chunks, AudioIO, line, pi, window
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.fft import rfft
import numpy as np
import collections, sys, threading

# AudioLazy init
rate = 44100
s, Hz = sHz(rate)
ms = 1e-3 * s

length = 2 ** 12
data = collections.deque([0.] * length, maxlen=length)
wnd = np.array(window.hamming(length)) # For FFT

api = sys.argv[1] if sys.argv[1:] else None # Choose API via command-line
chunks.size = 1 if api == "jack" else 16

# Creates a data updater callback
def update_data():
  with AudioIO(api=api) as rec:
    for el in rec.record(rate=rate):
      data.append(el)
      if update_data.finish:
        break

# Creates the data updater thread
update_data.finish = False
th = threading.Thread(target=update_data)
th.start() # Already start updating data

# Plot setup
fig = plt.figure("AudioLazy in a Matplotlib animation", facecolor='#cccccc')

time_values = np.array(list(line(length, -length / ms, 0)))
time_ax = plt.subplot(2, 1, 1,
                      xlim=(time_values[0], time_values[-1]),
                      ylim=(-1., 1.),
                      axisbg="black")
time_ax.set_xlabel("Time (ms)")
time_plot_line = time_ax.plot([], [], linewidth=2, color="#00aaff")[0]

dft_max_min, dft_max_max = .01, 1.
freq_values = np.array(line(length, 0, 2 * pi / Hz).take(length // 2 + 1))
freq_ax = plt.subplot(2, 1, 2,
                      xlim=(freq_values[0], freq_values[-1]),
                      ylim=(0., .5 * (dft_max_max + dft_max_min)),
                      axisbg="black")
freq_ax.set_xlabel("Frequency (Hz)")
freq_plot_line = freq_ax.plot([], [], linewidth=2, color="#00aaff")[0]

# Functions to setup and update plot
def init(): # Called twice on init, also called on each resize
  time_plot_line.set_data([], []) # Clear
  freq_plot_line.set_data([], [])
  fig.tight_layout()
  return [] if init.rempty else [time_plot_line, freq_plot_line]

init.rempty = False # At first, init() should show what belongs to the plot

def animate(idx):
  array_data = np.array(data)
  spectrum = np.abs(rfft(array_data * wnd)) / length

  time_plot_line.set_data(time_values, array_data)
  freq_plot_line.set_data(freq_values, spectrum)

  # Update y range if needed
  smax = spectrum.max()
  top = freq_ax.get_ylim()[1]
  if top < dft_max_max and abs(smax/top) > 1:
    freq_ax.set_ylim(top=top * 2)
  elif top > dft_max_min and abs(smax/top) < .3:
    freq_ax.set_ylim(top=top / 2)
  else:
    init.rempty = True # So "init" return [] (update everything on resizing)
    return [time_plot_line, freq_plot_line] # Update only what changed
  return []

# Animate! (assignment to anim is needed to avoid garbage collecting it)
anim = FuncAnimation(fig, animate, init_func=init, interval=10, blit=True)
plt.ioff()
plt.show() # Blocking

# Stop the recording thread after closing the window
update_data.finish = True
th.join()
