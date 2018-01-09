import numpy as np
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class DeepQNewWork:
    def __init__(self, actions, n_features, name, memory_size=500, batch_size=64,
                 reward_decay=0.9, greedy_rate=0.9, learning_rate=0.001):
        self.name = name
        self.actions = actions
        self.n_actions = len(actions)
        self.n_features = n_features
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = reward_decay
        self.epsilon = greedy_rate
        self.alpha = learning_rate

        self.evaluate_model = ValueModel(n_features, self.n_actions)
        self.policy_model = ValueModel(n_features, self.n_actions)
        self.optimizer = torch.optim.Adam(self.evaluate_model.parameters(), lr=self.alpha)
        self.mse = nn.MSELoss()
        self.memory = np.zeros((self.memory_size, ))
        self.memory_counter = 0

    def _store_transition(self, s, a, s_, r):
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = np.hstack((s, a, r, s_))
        self.memory_counter += 1

    def _build_target(self, a, s_, r):
        target = np.zeros(self.n_actions)
        target[a] = r + self.gamma * np.max(self.policy_model.foward(s_))
        return target

    def choose_action(self, observation):
        if np.random.uniform() < self.epsilon:
            action_values = self.policy_model.foward(observation)
            action = self.actions[np.argmax(action_values, self.n_features)]
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self):
        if self.memory_counter >= self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory_counter[sample_index, :]
        batch_features = batch_memory[:, self.n_features:]
        batch_targets = batch_memory[:, :-self.n_features]
        action, reward = batch_memory[:, self.n_features]
        q_predict = np.zeros(self.n_actions)
        q_predict[action] = self.evaluate_model.forward(batch_features)[action]
        q_target = self._build_target(action, reward, batch_targets)
        q_diff = Variable(torch.from_numpy(q_target - q_predict))





class ValueModel(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=64):
        super(ValueModel, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, state_batch):
        x = state_batch
        x = F.relu(self.linear1(x))
        actions_value = self.linear2(x)
        return actions_value
