import pandas as pd
import numpy as np
import pickle


class TableRL(object):
    def __init__(self, actions, name='qlearn', learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.name = name
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.rl_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.rl_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.argmax()
        else:
            action = np.random.choice(self.actions)
        return action

    def check_state(self, observation):
        if observation not in self.rl_table.index:
            self.rl_table = self.rl_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.actions,
                    name=observation,
                )
            )

    def save_table(self):
        with open('save_model/' + self.name + '.pkl', 'wb') as f:
            pickle.dump(self.rl_table, f)

    def load_table(self):
        with open('save_model/' + self.name + '.pkl', 'rb') as f:
            self.rl_table = pickle.load(f)

    def learn(self, *args):
        pass


class QLearningTable(TableRL):
    def __init__(self, actions, name='qlearn', learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, name, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, s_, r, done):
        q_predict = self.rl_table.ix[s, a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * self.rl_table.ix[s_, :].max()
        self.rl_table.ix[s, a] += self.alpha * (q_target - q_predict)


class SarsaTable(TableRL):
    def __init__(self, actions, name='sarsa', learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.8):
        super(SarsaTable, self).__init__(actions, name, learning_rate, reward_decay, e_greedy)
        self.lambda_ = trace_decay
        self.eligibility_trace = self.rl_table.copy()

    def learn(self, s, a, s_, a_, r, done):
        sa_predict = self.rl_table.ix[s, a]
        if done:
            sa_target = r
        else:
            sa_target = r + self.gamma * self.rl_table.ix[s_, a_]
        sa_error = sa_target - sa_predict
        self.eligibility_trace.ix[s, :] *= 0
        self.eligibility_trace.ix[s, a] = 1

        self.rl_table += sa_error * self.alpha * self.eligibility_trace
        self.eligibility_trace *= self.gamma * self.lambda_

    def check_state(self, observation):
        if observation not in self.rl_table.index:
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index=self.actions,
                    name=observation,
                )
            self.rl_table = self.rl_table.append(to_be_append)
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def clean_trace(self):
        self.eligibility_trace *= 0
