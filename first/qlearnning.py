import gym
import random

def greedy(qfunc, state, actions):
    amax = 0
    qmax = qfunc['{}_{}'.format(state, actions[0])]
    for i, a in enumerate(actions):
        s_a_pair = '{}_{}'.format(state, a)
        if qfunc[s_a_pair] > qmax:
            qmax = qfunc[s_a_pair]
            amax = i
    return amax


def epsilon_greedy(qfunc, state, actions, epsilon):
    amax = greedy(qfunc, state, actions)
    action_pros = [epsilon / len(actions) for _ in actions]
    action_pros[amax] += 1 - epsilon
    s = 0.0
    r = random.random()
    for i in range(len(actions)):
        s += action_pros[i]
        if s >= r:
            return actions[i]
        else:
            return actions[-1]


def qlearn(num_iter1, alpha, epsilon, env):
    qfunc = {}
    actions = env.getActions()
    for s in env.getStates():
        for a in actions:
            qfunc['{}_{}'.format(s, a)] = 0.0

    for iter1 in range(num_iter1):
        #print(qfunc)
        s = env.reset()
        a = random.choice(actions)
        t = False
        count = 0

        while not t and count < 100:
            state_action_pair = '{}_{}'.format(s, a)
            s1, r, t, _ = env.step(state_action_pair)
            next_action = actions[greedy(qfunc, s1, actions)]
            next_pair = '{}_{}'.format(s1, next_action)
            qfunc[state_action_pair] += alpha * (r + env.getGamma() * qfunc[next_pair] - qfunc[state_action_pair])
            s = s1
            a = epsilon_greedy(qfunc, s1, actions, epsilon)
            count += 1
            if t:
                print(count)
                print(r)
    return qfunc


if __name__ == '__main__':
    grid_env = gym.make('GridWorld-v0')
    state = grid_env.reset()
    for i in range(100):
        grid_env.render()
    value = qlearn(500, 0.2, 0.2, grid_env.env)
