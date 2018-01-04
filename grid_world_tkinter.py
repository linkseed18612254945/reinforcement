import tkinter as tk
import numpy as np
import sys
import time

HEIGHT = 4
WIDTH = 4
UNIT = 40


class Grid(tk.Tk, object):
    def __init__(self):
        super(Grid, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('Grid')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, WIDTH * UNIT))
        self._build_grid()

    def _build_grid(self):
        self.canvas = tk.Canvas(self, bg='white', height=HEIGHT * UNIT, width=WIDTH * UNIT)

        for c in range(0, WIDTH * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, WIDTH * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        origin = np.array([UNIT / 2, UNIT / 2])
        self.oval = self.canvas.create_oval(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')
        oval_center = origin + UNIT * 2
        self.target = self.canvas.create_rectangle(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(1)
        self.canvas.delete(self.oval)
        origin = np.array([UNIT / 2, UNIT / 2])
        self.oval = self.canvas.create_oval(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        return self.canvas.coords(self.oval)

    def step(self, action):
        s = self.canvas.coords(self.oval)
        base_action = np.array([0, 0])
        if action == 'u':
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 'd':
            if s[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 'r':
            if s[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 'l':
            if s[0] > UNIT:
                base_action[0] -= UNIT
        self.canvas.move(self.oval, base_action[0], base_action[1])
        s_ = self.canvas.coords(self.oval)
        if s_ == self.canvas.coords(self.target):
            reward = 1
            done = True
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        return s_, reward, done

    def render(self):
        self.update()
        time.sleep(0.01)


