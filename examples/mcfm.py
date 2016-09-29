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
Modulo Counter graphics with FM synthesis audio in a wxPython application
"""

# The GUI in this example is based on the dose TDD semaphore source code
# https://github.com/danilobellini/dose

import wx, sys
from math import pi
from audiolazy import (ControlStream, modulo_counter, chunks,
                       AudioIO, sHz, sinusoid)

MIN_WIDTH = 15 # pixels
MIN_HEIGHT = 15
FIRST_WIDTH = 200
FIRST_HEIGHT = 200
MOUSE_TIMER_WATCH = 50 # ms
DRAW_TIMER = 50

s, Hz = sHz(44100)

class McFMFrame(wx.Frame):

  def __init__(self, parent):
    frame_style = (wx.FRAME_SHAPED |     # Allows wx.SetShape
                   wx.FRAME_NO_TASKBAR |
                   wx.STAY_ON_TOP |
                   wx.NO_BORDER
                  )
    super(McFMFrame, self).__init__(parent, style=frame_style)
    self.Bind(wx.EVT_ERASE_BACKGROUND, lambda evt: None)
    self._paint_width, self._paint_height = 0, 0 # Ensure update_sizes at
                                                 # first on_paint
    self.ClientSize = (FIRST_WIDTH, FIRST_HEIGHT)
    self.Bind(wx.EVT_PAINT, self.on_paint)
    self._draw_timer = wx.Timer(self)
    self.Bind(wx.EVT_TIMER, self.on_draw_timer, self._draw_timer)
    self.on_draw_timer()
    self.angstep = ControlStream(pi/90)
    self.rotstream = modulo_counter(modulo=2*pi, step=self.angstep)
    self.rotation_data = iter(self.rotstream)

  def on_draw_timer(self, evt=None):
    self.Refresh()
    self._draw_timer.Start(DRAW_TIMER, True)

  def on_paint(self, evt):
    dc = wx.AutoBufferedPaintDCFactory(self)
    gc = wx.GraphicsContext.Create(dc) # Anti-aliasing

    gc.SetPen(wx.Pen("blue", width=4))
    gc.SetBrush(wx.Brush("black"))
    w, h = self.ClientSize
    gc.DrawRectangle(0, 0, w, h)

    gc.SetPen(wx.Pen("gray", width=2))
    w, h = w - 10, h - 10
    gc.Translate(5, 5)
    gc.DrawEllipse(0, 0, w, h)
    gc.SetPen(wx.Pen("red", width=1))
    gc.SetBrush(wx.Brush("yellow"))
    gc.Translate(w * .5, h * .5)
    gc.Scale(w, h)
    rot = next(self.rotation_data)
    gc.Rotate(-rot)
    gc.Translate(.5, 0)
    gc.Rotate(rot)
    gc.Scale(1./w, 1./h)
    gc.DrawEllipse(-5, -5, 10, 10)


class InteractiveFrame(McFMFrame):
  def __init__(self, parent):
    super(InteractiveFrame, self).__init__(parent)
    self._timer = wx.Timer(self)
    self.Bind(wx.EVT_RIGHT_DOWN, self.on_right_down)
    self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
    self.Bind(wx.EVT_TIMER, self.on_timer, self._timer)

  @property
  def player(self):
    return self._player

  @player.setter
  def player(self, value):
    # Also initialize playing thread
    self._player = value
    self.volume_ctrl = ControlStream(.2)
    self.carrier_ctrl = ControlStream(220)
    self.mod_ctrl = ControlStream(440)
    sound = sinusoid(freq=self.carrier_ctrl * Hz,
                     phase=sinusoid(self.mod_ctrl * Hz)
                    ) * self.volume_ctrl
    self.playing_thread = player.play(sound)

  def on_right_down(self, evt):
    self.Close()

  def on_left_down(self, evt):
    self._key_state = None # Ensures initialization
    self.on_timer(evt)

  def on_timer(self, evt):
    """
    Keep watching the mouse displacement via timer
    Needed since EVT_MOVE doesn't happen once the mouse gets outside the
    frame
    """
    ctrl_is_down = wx.GetKeyState(wx.WXK_CONTROL)
    ms = wx.GetMouseState()

    # New initialization when keys pressed change
    if self._key_state != ctrl_is_down:
      self._key_state = ctrl_is_down

      # Keep state at click
      self._click_ms_x, self._click_ms_y = ms.x, ms.y
      self._click_frame_x, self._click_frame_y = self.Position
      self._click_frame_width, self._click_frame_height = self.ClientSize

      # Avoids refresh when there's no move (stores last mouse state)
      self._last_ms = ms.x, ms.y

      # Quadrant at click (need to know how to resize)
      width, height = self.ClientSize
      self._quad_signal_x = 1 if (self._click_ms_x -
                                  self._click_frame_x) / width > .5 else -1
      self._quad_signal_y = 1 if (self._click_ms_y -
                                  self._click_frame_y) / height > .5 else -1

    # "Polling watcher" for mouse left button while it's kept down
    if ms.leftDown:
      if self._last_ms != (ms.x, ms.y): # Moved?
        self._last_ms = (ms.x, ms.y)
        delta_x = ms.x - self._click_ms_x
        delta_y = ms.y - self._click_ms_y

        # Resize
        if ctrl_is_down:
          # New size
          new_w = max(MIN_WIDTH, self._click_frame_width +
                                 2 * delta_x * self._quad_signal_x
                     )
          new_h = max(MIN_HEIGHT, self._click_frame_height +
                                  2 * delta_y * self._quad_signal_y
                     )
          self.ClientSize = new_w, new_h
          self.SendSizeEvent() # Needed for wxGTK

          # Center should be kept
          center_x = self._click_frame_x + self._click_frame_width / 2
          center_y = self._click_frame_y + self._click_frame_height / 2
          self.Position = (center_x - new_w / 2,
                           center_y - new_h / 2)

          self.Refresh()
          self.volume_ctrl.value = (new_h * new_w) / 3e5

        # Move the window
        else:
          self.Position = (self._click_frame_x + delta_x,
                           self._click_frame_y + delta_y)

          # Find the new center position
          x, y = self.Position
          w, h = self.ClientSize
          cx, cy = x + w/2, y + h/2
          self.mod_ctrl.value = 2.5 * cx
          self.carrier_ctrl.value = 2.5 * cy
          self.angstep.value = (cx + cy) * pi * 2e-4

      # Since left button is kept down, there should be another one shot
      # timer event again, without creating many timers like wx.CallLater
      self._timer.Start(MOUSE_TIMER_WATCH, True)


class McFMApp(wx.App):

  def OnInit(self):
    self.SetAppName("mcfm")
    self.wnd = InteractiveFrame(None)
    self.wnd.Show()
    self.SetTopWindow(self.wnd)
    return True # Needed by wxPython


if __name__ == "__main__":
  api = sys.argv[1] if sys.argv[1:] else None # Choose API via command-line
  chunks.size = 1 if api == "jack" else 16
  with AudioIO(api=api) as player:
    app = McFMApp(False, player)
    app.wnd.player = player
    app.MainLoop()
