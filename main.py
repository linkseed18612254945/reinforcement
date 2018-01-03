from grid_world_tkinter import Grid
import random

EPISODE_NUM = 100


def train():
    for episode in range(EPISODE_NUM):
        observation = env.reset()
        while True:
            # 更新可视化环境
            env.render()
            # 根据策略选择行为
            action = random.choice(env.action_space)
            # 采取行为进行交互,得到下一个状态,结束情况以及回报值
            observation_, reward, done = env.step(action)
            # RL根据状态行为以及预测结果来学习,更新价值函数等
            # rl_model.learn(observation, action, observation_, reward, done)

            observation = observation_
            if done:
                break
    print('Game Over')
    env.destroy()


if __name__ == '__main__':
    env = Grid()
    rl_model = None
    env.after(100, train)
    env.mainloop()