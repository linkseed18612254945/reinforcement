import tkinter as tk
import numpy as np
import pandas as pd
import time

F_HEIGHT = 3
F_WIDTH = 3
UNIT = 40
WINNER_COUNT = 3


class FlagWorld(tk.Tk):
    def __init__(self):
        super(FlagWorld, self).__init__()
        self.title = 'Flag'
        self.actions = [(i, j) for i in range(F_HEIGHT) for j in range(F_WIDTH)]
        self.n_actions = len(self.actions)
        self.state = np.zeros((F_HEIGHT, F_WIDTH))
        self.geometry('{0}x{1}'.format(F_HEIGHT * UNIT, F_WIDTH * UNIT))
        self._init_build()

    def _init_build(self):
        self.canvas = tk.Canvas(self, height=F_HEIGHT * UNIT, width=F_WIDTH * UNIT, bg='white')

        for c in range(0, F_WIDTH * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, F_HEIGHT * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, F_HEIGHT * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, F_WIDTH * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
        self.canvas.pack()

    def _check_winner_end(self):
        winner = 0
        end = False
        result = []
        for i in range(F_HEIGHT):
            result.append(self.state[i, :].sum())
            pos = np.array([i, 0])
            result.append(0)
            while pos[0] < F_HEIGHT:
                result[-1] += self.state[pos[0], pos[1]]
                pos += (1, 1)
            pos = np.array([i, 0])
            result.append(0)
            while pos[0] >= 0:
                result[-1] += self.state[pos[0], pos[1]]
                pos += (-1, 1)

        for i in range(F_WIDTH):
            result.append(self.state[:, i].sum())
            pos = np.array([0, i])
            result.append(0)
            while pos[1] < F_HEIGHT:
                result[-1] += self.state[pos[0], pos[1]]
                pos += (1, 1)
            pos = np.array([i, 0])
            result.append(0)
            while pos[1] >= 0:
                result[-1] += self.state[pos[0], pos[1]]
                pos += (1, -1)

        for i in result:
            if i == WINNER_COUNT:
                winner = 1
                end = True
                return winner, end
            elif i == -WINNER_COUNT:
                winner = -1
                end = True
                return winner, end

        if np.sum(np.abs(self.state)) == F_HEIGHT * F_WIDTH:
            winner = 0
            end = True
            return winner, end
        return winner, end

    def reset(self):
        self.state = np.zeros((F_HEIGHT, F_WIDTH))
        self.canvas.destroy()
        self._init_build()
        return self.state

    def step(self, action, role):
        if action not in self.actions:
            return
        if role != 1 and role != -1:
            return
        if self.state[action] != 0:
            return

        role_color = 'red' if role == 1 else 'blue'
        center = np.array([action[0] * UNIT + UNIT / 2, action[1] * UNIT + UNIT / 2])
        self.canvas.create_oval(
            center[0] - 15, center[1] - 15,
            center[0] + 15, center[1] + 15,
            fill=role_color)
        self.state[action] = role
        winner, end = self._check_winner_end()
        if winner == role:
            reward = 1
        else:
            reward = 0
        return self.state, reward, end

    def render(self, sleep_time=1):
        self.update()
        time.sleep(sleep_time)


if __name__ == '__main__':
    env = FlagWorld()
    env.render()
    env.step((0, 0), -1)
    env.render()
    env.step((0, 1), 1)
    env.render()
    env.reset()
    env.render(4)