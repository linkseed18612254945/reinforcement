import numpy as np
import pickle
import torch


class DeepQNewWork:
    def __init__(self, actions, name, reward_decay=0.9, greedy_rate=0.9):
        self.name = name
        self.actions = actions
        self.n_actions = len(actions)
        self.gamma = reward_decay
        self.epsilon = greedy_rate
        self.value_model = None

    def choose_action(self, observation):
        if np.random.uniform() < self.epsilon:
            action_values = self.value_model.foward(observation)
            action = self.actions[np.argmax(action_values)]
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self):
        pass