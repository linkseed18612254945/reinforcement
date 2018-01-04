from grid_world_tkinter import Grid
import random
from brain import *

EPISODE_NUM = 100


def train():
    for episode in range(EPISODE_NUM):
        sarsa_model.clean_trace()
        print(sarsa_model.rl_table)
        observation = env.reset()
        action = sarsa_model.choose_action(str(observation))
        while True:
            # 更新可视化环境
            env.render()
            # 根据策略选择行为

            # 采取行为进行交互,得到下一个状态,结束情况以及回报值
            observation_, reward, done = env.step(action)
            action_ = sarsa_model.choose_action(str(observation_))
            # RL根据状态行为以及预测结果来学习,更新价值函数等
            sarsa_model.learn(str(observation), action, str(observation_), action_, reward, done)

            observation = observation_
            action = action_
            if done:
                break
    print('Game Over')
    env.destroy()


if __name__ == '__main__':
    env = Grid()
    qlearn_model = QLearningTable(env.action_space)
    sarsa_model = SarsaTable(env.action_space)
    env.after(100, train)
    env.mainloop()