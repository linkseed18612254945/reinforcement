from grid_world_tkinter import Grid
from flag_tkinter import FlagWorld
import random
from brain import *


TRY_NUM = 20


def sarsa_train(env, rl_model, episode_num=100):
    for episode in range(episode_num):
        print('Train {0}/{1}'.format(episode, episode_num))
        rl_model.clean_trace()
        observation = env.reset()
        action = rl_model.choose_action(observation)
        while True:
            observation_, reward, done = env.step(action)
            action_ = rl_model.choose_action(observation_)
            rl_model.learn(observation, action, observation_, action_, reward, done)
            observation = observation_
            action = action_
            if done:
                break
    print('Train Over')
    rl_model.save_table()
    return rl_model


def qlearn_train(env, rl_model, episode_num=100):
    for episode in range(episode_num):
        print('Train {0}/{1}'.format(episode, episode_num))
        observation = env.reset()
        while True:
            action = rl_model.choose_action(observation)
            observation_, reward, done = env.step(action)
            rl_model.learn(observation, action, observation_, reward, done)
            observation = observation_
            if done:
                break
    print('Train Over')
    rl_model.save_table()
    return rl_model


def model_test(env, rl_model, try_num):
    rl_model.set_greedy()
    for episode in range(try_num):
        observation = env.reset()
        env.render()
        while True:
            action = rl_model.choose_action(str(observation))
            observation_, _, done = env.step(action)
            observation = observation_
            env.render()
            if done:
                break


def player_train(env, player1, player2, episode_num=100):
    player1_win = 0
    player2_win = 0
    tie = 0
    for episode in range(episode_num):
        print('Train {0}/{1}'.format(episode, episode_num))
        player1.clean_trace()
        player2.clean_trace()
        current_player = player1
        opponent_player = player2
        observation = env.reset()
        while True:
            action = current_player.choose_action(observation)
            observation_, reward, done = env.step(action, observation, current_player.role)
            current_player.feed_state(observation, action)
            if done:
                if reward == 1:
                    if current_player is player1:
                        player1_win += 1
                    else:
                        player2_win += 1
                    current_player.learn(reward)
                    opponent_player.learn(-reward)
                else:
                    tie += 1
                break
            observation = observation_
            current_player, opponent_player = opponent_player, current_player
    print('Train Over, player1 win{0}, player2 win{1}, tie{2}'.
          format(player1_win / episode_num, player2_win / episode_num, tie / episode_num))
    player1.save_table()
    player2.save_table()
    return player1, player2


def player_compete(env, player1, player2, compete_num):
    player1.set_greedy()
    player2.set_greedy()
    player1_win = 0
    player2_win = 0
    tie = 0
    for episode in range(compete_num):
        print('Compete round ' + str(episode))
        player1.clean_trace()
        player2.clean_trace()
        current_player = player1
        opponent_player = player2
        observation = env.reset()
        env.render()
        while True:
            action = current_player.choose_action(observation)
            observation_, reward, done = env.step(action, observation, current_player.role)
            env.render()
            if done:
                if reward == 1:
                    if current_player is player1:
                        player1_win += 1
                    else:
                        player2_win += 1
                else:
                    tie += 1
                break
            observation = observation_
            current_player, opponent_player = opponent_player, current_player
    print('Train Over, player1 win{0}, player2 win{1}, tie{2}'.
          format(player1_win / compete_num, player2_win / compete_num, tie / compete_num))


def grid_main():
    env = Grid()
    qlearn_model = QLearningTable(env.action_space)
    sarsa_model = SarsaTable(env.action_space, trace_decay=0.9)
    # sarsa_model = sarsa_train(env, sarsa_model, 100)
    # qlearn_model = qlearn_train(env, qlearn_model, 100)
    qlearn_model.load_table()
    sarsa_model.load_table()
    model_test(env, sarsa_model, 5)


def play_main():
    env = FlagWorld()
    player1 = PlayerRL(env.actions, 'player1', role=1)
    player2 = PlayerRL(env.actions, 'player2', role=-1)
    # player1, player2 = player_train(env, player1, player2, 20000)
    player1.load_table()
    player2.load_table()
    player_compete(env, player1, player2, 20)


if __name__ == '__main__':
    play_main()
