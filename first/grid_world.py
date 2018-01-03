import random

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import HTML
from torch.autograd import Variable

from first.helpers import *

EDGE_VALUE = -10
GOAL_VALUE = 10
VISIBLE_RADIUS = 1
MIN_PLANT_VALUE = -1
MAX_PLANT_VALUE = 0.5

START_HEALTH = 1
STEP_VALUE = -0.02

class Grid(object):
    def __init__(self, grid_size=8, planet_num=64):
        self.grid_size = grid_size
        self.planet_num = 64

        self.reset()

    def reset(self):
        padded_size = self.grid_size + 2 * VISIBLE_RADIUS
        self.grid = np.zeros((padded_size, padded_size))
        self.grid[0:VISIBLE_RADIUS, :] = EDGE_VALUE
        self.grid[-VISIBLE_RADIUS:, :] = EDGE_VALUE
        self.grid[:, 0: VISIBLE_RADIUS] = EDGE_VALUE
        self.grid[:, -VISIBLE_RADIUS:] = EDGE_VALUE

        S = VISIBLE_RADIUS
        E = padded_size - VISIBLE_RADIUS - 1
        for i in range(self.planet_num):
            plant_value = MIN_PLANT_VALUE + (MAX_PLANT_VALUE - MIN_PLANT_VALUE) * random.random()
            random_y = random.randint(S, E)
            random_x = random.randint(S, E)
            self.grid[random_y, random_x] = plant_value
        goal_points = [(S, S), (S, E), (E, S), (E, E)]
        goal_point = random.choice(goal_points)
        self.grid[goal_point] = GOAL_VALUE

    def visible(self, pos):
        y, x = pos
        return self.grid[y - VISIBLE_RADIUS: y + VISIBLE_RADIUS + 1, x - VISIBLE_RADIUS: x + VISIBLE_RADIUS + 1]


class Agent:
    def reset(self):
        self.health = START_HEALTH

    def act(self, action):
        # 0-UP, 1-RIGHT, 2- DOWN, 3-LEFT
        y, x = self.pos
        if action == 0:
            y -= 1
        if action == 1:
            x += 1
        if action == 2:
            y += 1
        if action == 3:
            x -= 1
        self.pos = (y, x)
        self.health += STEP_VALUE

class Environment:
    def __init__(self):
        self.grid = Grid()
        self.agent = Agent()

    def reset(self):
        self.grid.reset()
        self.agent.reset()
        center = self.grid.grid_size // 2
        self.agent.pos = (center, center)

        self.t = 0
        self.history = []
        self.record_step()

    def record_step(self):
        grid = self.grid.grid.copy()
        grid[self.agent.pos] = self.agent.health / 2
        visible = self.grid.visible(self.agent.pos).copy()
        self.history.append((grid, visible, self.agent.health))

    @property
    def visible_state(self):
        visible = self.grid.visible(self.agent.pos)
        y, x = self.agent.pos
        yp = (y - VISIBLE_RADIUS) / self.grid.grid_size
        xp = (x - VISIBLE_RADIUS) / self.grid.grid_size
        extras = [self.agent.health, yp, xp]
        return np.concatenate((visible.flatten(), extras), 0)

    def step(self, action):
        self.agent.act(action)

        value = self.grid.grid[self.agent.pos]
        self.grid.grid[self.agent.pos] = 0
        self.agent.health += value

        won = value == GOAL_VALUE
        lost = self.agent.health <= 0
        done = won or lost

        if won:
            reward = 1
        elif lost:
            reward = -1
        else:
            reward = 0

        self.record_step()
        return self.visible_state, reward, done

def animate(history):
    frames = len(history)
    fig = plt.figure(figsize=(6, 2))
    fig_grid = fig.add_subplot(121)
    fig_health = fig.add_subplot(243)
    fig_visible = fig.add_subplot(244)
    fig_health.set_autoscale_on(False)
    health_plot = np.zeros((frames, 1))

    def render_frame(i):
        grid, visible, health = history[i]
        # Render grid
        fig_grid.matshow(grid, vmin=-1, vmax=1, cmap='jet')
        fig_visible.matshow(visible, vmin=-1, vmax=1, cmap='jet')
        # Render health chart
        health_plot[i] = health
        fig_health.clear()
        fig_health.axis([0, frames, 0, 2])
        fig_health.plot(health_plot[:i + 1])

    anim = matplotlib.animation.FuncAnimation(
        fig, render_frame, frames=frames, interval=100
    )
    plt.show()
    plt.close()
    HTML(anim.to_html5_video())


class Police(nn.Module):
    def __init__(self, hidden_size):
        super(Police, self).__init__()
        visible_size = (VISIBLE_RADIUS * 2 + 1) ** 2
        input_size = visible_size + 3

        self.inp = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, 4 + 1, bias=False)

    def forward(self, x):
        x = x.view(1, -1)
        x = F.tanh(x)
        x = F.relu(self.inp(x))
        x = self.out(x)

        scores = x[:, :4]
        value = x[:, 4]
        return scores, value


DROP_MAX = 0.3
DROP_MIN = 0.05
DROP_OVER = 200000
def select_action(e, state, drop=True):
    state = Variable(torch.from_numpy(state).float())
    scores, value = police_model.forward(state)
    if drop:
        drop = interpolate(e, DROP_MAX, DROP_MIN, DROP_OVER)
        scores = F.dropout(scores, drop)
    scores = F.softmax(scores)
    action = scores.multinomial()
    return action, value

def run_episode(e):
    env.reset()
    state = env.visible_state
    actions = []
    values = []
    rewards = []
    done = False

    while not done:
        action, value = select_action(e, state)
        state, reward, done = env.step(action.data[0, 0])
        actions.append(action)
        values.append(value)
        rewards.append(reward)
    return actions, values, rewards

gamma = 0.9
mse = nn.MSELoss()
def finish_episode(e, actions, values, rewards):
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.Tensor(discounted_rewards)
    value_loss = 0
    for action, value, reward in zip(actions, values, discounted_rewards):
        reward_diff = reward - value.data[0]
        action.reinforce(reward_diff)
        value_loss += mse(value, Variable(torch.Tensor([reward])))

    optimizer.zero_grad()
    nodes = [value_loss] + actions
    gradients = [torch.ones(1)] + [None for _ in actions]
    torch.autograd.backward(nodes, gradients)
    optimizer.step()

    return discounted_rewards, value_loss


if __name__ == '__main__':
    hidden_size = 50
    learning_rate = 1e-4
    weight_decay = 1e-5
    log_every = 1000
    render_every = 20000

    env = Environment()
    police_model = Police(hidden_size)
    optimizer = torch.optim.Adam(police_model.parameters(), lr=1e-4)

    reward_avg = SlidingAverage('reward avg', steps=log_every)
    value_avg = SlidingAverage('value avg', steps=log_every)

    e = 0
    while reward_avg < 0.75:
        actions, values, rewards = run_episode(e)
        final_reward = rewards[-1]

        discounted_rewards, value_loss = finish_episode(e, actions, values, rewards)

        reward_avg.add(final_reward)
        value_avg.add(value_loss.data[0])

        if e % log_every == 0:
            print('[epoch=%d]' % e, reward_avg, value_avg)

        if e > 0 and e % render_every == 0:
            animate(env.history)

        e += 1
