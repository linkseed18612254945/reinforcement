import gym
import time
env = gym.make('CartPole-v0')
observation = env.reset()
count = 0
for t in range(100):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        break
    env.render()
    count += 1
    time.sleep(0.2)
print(count)