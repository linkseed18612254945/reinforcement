from gym import spaces
from gym.utils import seeding
import numpy as np
import gym
from gym.envs.classic_control import rendering

class Grid(object):
    def __init__(self, x: int=None, y: int=None,
                 grid_type: int=0, reward: int=0):
        self.x = x
        self.y = y
        self.type = type
        self.reward = reward
        self.grid_type = grid_type
        self.value = 0

    @property
    def grid_name(self):
        return 'X{0}-Y{1}'.format(self.x, self.y)


class GridMatrix(object):
    def __init__(self, width: int, height: int, default_type: int=0, default_reward: int=0):
        self.grids = None
        self.width = width
        self.height = height
        self.len = width * height
        self.default_type = default_type
        self.default_reward = default_reward

        self.reset()

    def reset(self):
        self.grids = []
        for x in range(self.height):
            for y in range(self.width):
                self.grids.append(Grid(x, y, self.default_type, self.default_reward))

    def get_grid(self, x, y=None):
        if isinstance(x, int):
            xx, yy = x, y
        elif isinstance(x, tuple) or isinstance(x, list):
            xx, yy = x[0], x[1]
        else:
            return None
        index = yy * self.width + xx
        return self.grids[index]

    def set_reward(self, x, y, reward):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.reward = reward
        else:
            raise "Wrong Pos!"

    def set_value(self, x, y, value):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.value = value
        else:
            raise "Wrong Pos!"

    def set_type(self, x, y, type):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.type = type
        else:
            raise "Wrong Pos!"

    def get_reward(self, x, y):
        grid = self.get_grid(x, y)
        if grid is not None:
            return grid.reward
        else:
            return None

    def get_value(self, x, y):
        grid = self.get_grid(x, y)
        if grid is not None:
            return grid.value
        else:
            return None

    def get_type(self, x, y):
        grid = self.get_grid(x, y)
        if grid is not None:
            return grid.type
        else:
            return None


class GridWorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
        }
    def __init__(self, width: int=10, height: int=7, u_size=40, default_reward=0, default_type=0, windy=False):
        self.width = width
        self.height = height
        self.u_size = u_size
        self.default_reward = default_reward
        self.default_type = default_type
        self.windy = windy

        self.width_size = width * u_size
        self.height_size = height * u_size

        self.grids = GridMatrix(self.width, self.height, self.default_type, self.default_reward)
        self.reward = 0
        self.action = None

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.height * self.width)

        self.ends = [(7, 3)]
        self.start = (0, 3)
        self.types = []
        self.rewards = []
        self.refresh_setting()
        self.viewer = None
        self._seed()
        self.reset()

    def _state_to_xy(self, s):
        x = s % self.width
        y = int((s - x) / self.width)
        return x, y

    def _xy_to_state(self, x, y=None):
        if isinstance(x, tuple):
            return x[0] + self.width_size * x[1]
        return x[0] + self.width * y

    def _is_end_state(self, x, y):
        for end in self.ends:
            if x == end[0] and y == end[1]:
                return True
        return False

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self.action = action
        old_x, old_y = self._state_to_xy(self.state)
        new_x, new_y = old_x, old_y

        if self.windy:
            if new_x in [3, 4, 5, 8]:
                new_y += 1
            elif new_x in [6, 7]:
                new_y += 2

        if action == 0:
            new_x -= 1  # left
        elif action == 1:
            new_x += 1  # right
        elif action == 2:
            new_y += 1  # up
        elif action == 3:
            new_y -= 1  # down

        elif action == 4:
            new_x, new_y = new_x - 1, new_y - 1
        elif action == 5:
            new_x, new_y = new_x + 1, new_y - 1
        elif action == 6:
            new_x, new_y = new_x + 1, new_y - 1
        elif action == 7:
            new_x, new_y = new_x + 1, new_y + 1
        # boundary effect
        if new_x < 0: new_x = 0
        if new_x >= self.width: new_x = self.width - 1
        if new_y < 0: new_y = 0
        if new_y >= self.height: new_y = self.height - 1

        # wall effect:
        if self.grids.get_type(new_x, new_y) == 1:
            new_x, new_y = old_x, old_y

        self.reward = self.grids.get_reward(new_x, new_y)

        done = self._is_end_state(new_x, new_y)
        self.state = self._xy_to_state(new_x, new_y)

        # 提供格子世界所有的信息在info内
        info = {"x": new_x, "y": new_y, "grids": self.grids}
        return self.state, self.reward, done, info

    def _reset(self):
        self.state = self._xy_to_state(self.start)
        return self.state

    def refresh_setting(self):
        for x, y, r in self.rewards:
            self.grids.set_reward(x, y, r)
        for x, y, t in self.types:
            self.grids.set_type(x, y, t)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        zero = (0, 0)
        u_size = self.u_size
        m = 2  # 格子之间的间隙尺寸

        # 如果还没有设定屏幕对象，则初始化整个屏幕具备的元素。
        if self.viewer is None:

            self.viewer = rendering.Viewer(self.width_size, self.height_size)

            # 绘制格子
            for x in range(self.width):
                for y in range(self.height):
                    v = [(x * u_size + m, y * u_size + m),
                         ((x + 1) * u_size - m, y * u_size + m),
                         ((x + 1) * u_size - m, (y + 1) * u_size - m),
                         (x * u_size + m, (y + 1) * u_size - m)]

                    rect = rendering.FilledPolygon(v)
                    r = self.grids.get_reward(x, y) / 10
                    if r < 0:
                        rect.set_color(0.9 - r, 0.9 + r, 0.9 + r)
                    elif r > 0:
                        rect.set_color(0.3, 0.5 + r, 0.3)
                    else:
                        rect.set_color(0.9, 0.9, 0.9)
                    self.viewer.add_geom(rect)
                    # 绘制边框
                    v_outline = [(x * u_size + m, y * u_size + m),
                                 ((x + 1) * u_size - m, y * u_size + m),
                                 ((x + 1) * u_size - m, (y + 1) * u_size - m),
                                 (x * u_size + m, (y + 1) * u_size - m)]
                    outline = rendering.make_polygon(v_outline, False)
                    outline.set_linewidth(3)

                    if self._is_end_state(x, y):
                        # 给终点方格添加金黄色边框
                        outline.set_color(0.9, 0.9, 0)
                        self.viewer.add_geom(outline)
                    if self.start[0] == x and self.start[1] == y:
                        outline.set_color(0.5, 0.5, 0.8)
                        self.viewer.add_geom(outline)
                    if self.grids.get_type(x, y) == 1:  # 障碍格子用深灰色表示
                        rect.set_color(0.3, 0.3, 0.3)
                    else:
                        pass
            # 绘制个体
            self.agent = rendering.make_circle(u_size / 4, 30, True)
            self.agent.set_color(1.0, 1.0, 0.0)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)

        # 更新个体位置
        x, y = self._state_to_xy(self.state)
        self.agent_trans.set_translation((x + 0.5) * u_size, (y + 0.5) * u_size)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

def LargeGridWorld():
    env = GridWorldEnv(width=10, height=10, u_size=40, default_reward=0,
                       default_type=0, windy=False)
    env.start = (0, 9)
    env.ends = [(5, 4)]
    env.types = [(4, 2, 1), (4, 3, 1), (4, 4, 1), (4, 5, 1), (4, 6, 1), (4, 7, 1),
                 (1, 7, 1), (2, 7, 1), (3, 7, 1), (4, 7, 1), (6, 7, 1), (7, 7, 1),
                 (8, 7, 1)]
    env.rewards = [(3, 2, -1), (3, 6, -1), (5, 2, -1), (6, 2, -1), (8, 3, -1),
                   (8, 4, -1), (5, 4, 1), (6, 4, -1), (5, 5, -1), (6, 5, -1)]
    env.refresh_setting()
    return env

if __name__ == '__main__':
    env = LargeGridWorld()
    start_state = env.reset()
    nfs = env.observation_space
    nfa = env.action_space
    print("nfs:%s; nfa:%s" % (nfs, nfa))
    print(env.observation_space)
    print(env.action_space)
    print(env.state)
    env.render()