import numpy as np
import pandas as pd
import time

N_STATES = 8 # 状态数, 1维世界长度
ACTIONS = ['left', 'right'] # 行为空间
EPSILON = 0.9 # 贪婪度
ALPHA = 0.1 # 学习率
GAMMA = 0.9 # 奖励递减比例
MAX_EPISODES = 50 # 最大回合数
FRESH_TIME = 0.1 # 变化间隔时间


def build_q_table(n_states, actions):
    table = np.zeros((n_states, len(actions)))
    table = pd.DataFrame(table, columns=actions)
    return table


def choose_action(state, q_table):
    state_values = q_table.iloc[state, :]
    if state_values.all() == 0 or np.random.uniform() > EPSILON:
        action = np.random.choice(ACTIONS)
    else:
        action = state_values.argmax()
    return action


def next_state(state, action):
    if state == 0 and action == 'left':
        return state
    elif state == N_STATES - 1 and action == 'right':
        return state
    else:
        if action == 'left':
            return state - 1
        else:
            return state + 1


def take_action(state, action):
    is_end = False
    reward = 0
    n_state = next_state(state, action)
    if n_state == N_STATES - 1:
        is_end = True
        reward = 1
    return n_state, reward, is_end


def update_env(state, is_end, episode, step_counter):
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if is_end:
        print('Episode %s: total_steps = %s' % (episode+1, step_counter))
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print(interaction)
        time.sleep(FRESH_TIME)


def update_q_table(q_table, state, action, n_state, reward, is_end):
    if not is_end:
        q_target = reward + GAMMA * q_table.iloc[n_state, :].max()
    else:
        q_target = reward
    q_table.ix[state, action] += ALPHA * (q_target - q_table.ix[state, action])
    return q_table

def main():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        # reset
        state = 0
        step = 0
        is_end = False
        update_env(state, is_end, episode, step)
        while not is_end:
            action = choose_action(state, q_table)
            n_state, reward, is_end = take_action(state, action)
            q_table = update_q_table(q_table, state, action, n_state, reward, is_end)
            state = n_state
            update_env(state, is_end, episode, step)
            step += 1
        print(q_table)
    return q_table



if __name__ == '__main__':
    table = main()
    print(table)

