import gym
import random
import logging

logger = logging.getLogger(__name__)


class GridEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        # define  MDP<S, A, R, P, gamma>
        self.states = [1, 2, 3, 4, 5, 6, 7, 8]
        self.action_space = ['n', 'e', 'w', 's']
        self.rewards = {'1_s': -1, '5_s': -1, '3_s': 1}
        self.t = {'1_s': 6, '1_e': 2, '2_w': 1, '2_e': 3, '3_e': 4, '3_w': 2,
                  '3_s': 7, '4_w': 3, '4_e': 5, '5_s': 8, '5_w': 4}
        self.gamma = 0.8

        self.terminate_states = (6, 7, 8)
        self.state = None
        self.viewer = None

        self.x = [140, 220, 300, 380, 460, 140, 300, 460]
        self.y = [250, 250, 250, 250, 250, 150, 150, 150]

    def _step(self, action):
        state = self.state
        if state in self.terminate_states:
            return state, 0, True, {}
        key = '{}_{}'.format(state, action)
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state
        self.state = next_state
        is_terminal = next_state in self.terminate_states
        reward = self.rewards[key] if key in self.rewards else 0.0
        return next_state, reward, is_terminal, {}

    def _reset(self):
        self.state = random.choice(self.states)
        return self.state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.line1 = rendering.Line((100, 300), (500, 300))
            self.line2 = rendering.Line((100, 200), (500, 200))
            self.line3 = rendering.Line((100, 300), (100, 100))
            self.line4 = rendering.Line((180, 300), (180, 100))
            self.line5 = rendering.Line((260, 300), (260, 100))
            self.line6 = rendering.Line((340, 300), (340, 100))
            self.line7 = rendering.Line((420, 300), (420, 100))
            self.line8 = rendering.Line((500, 300), (500, 100))
            self.line9 = rendering.Line((100, 100), (180, 100))
            self.line10 = rendering.Line((260, 100), (340, 100))
            self.line11 = rendering.Line((420, 100), (500, 100))

            self.kulo1 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(140, 150))
            self.kulo1.add_attr(self.circletrans)
            self.kulo1.set_color(0, 0, 0)
            self.kulo2 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(460, 150))
            self.kulo2.add_attr(self.circletrans)
            self.kulo2.set_color(0, 0, 0)

            self.gold = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(300, 150))
            self.gold.add_attr(self.circletrans)
            self.gold.set_color(1, 0.9, 0)

            self.robot = rendering.make_circle(30)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0.8, 0.6, 0.4)

            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)
            self.line11.set_color(0, 0, 0)

            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.kulo1)
            self.viewer.add_geom(self.kulo2)
            self.viewer.add_geom(self.gold)
            self.viewer.add_geom(self.robot)


        if self.state is None:
            return None
        self.robotrans.set_translation(self.x[self.state - 1], self.y[self.state - 1])
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


if __name__ == '__main__':
    env = gym.make('GridWorld-v0')
    state = env.reset()
    print(state)
    print(env.action_space)
    for i in range(1000):
        env.render()

