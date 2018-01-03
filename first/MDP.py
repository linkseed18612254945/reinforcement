import numpy as np


class Grid:
    def __init__(self, size=4, ld=1):
        self.grid = np.ones((size, size)) * -1
        self.size = size
        self.ld = ld
        self.grid[0][0] = 0
        self.grid[size - 1][size - 1] = 0
        self.values = np.zeros((size, size))
        self.actions = {'l':(-1, 0), 'r':(1, 0), 'u':(0, -1), 'd':(0, 1)}

    def performOneIter(self):
        temp_values = self.values.copy()
        for i in range(self.size):
            for j in range(self.size):
                temp_values[i, j] = self.update_value((i, j))
        self.values = temp_values

    def update_value(self, s):
        successor = self.getSuccessors(s)
        new_value = 0
        num = 4
        reward = self.grid[s]
        for next_state in successor:
            new_value += 1.00 / num * (reward + self.ld * self.values[next_state])
        return new_value

    def isTerminate(self, s):
        return (s[0] == 0 and s[1] == 0) or (s[0] == self.size - 1 and s[1] == self.size - 1)

    def nextState(self, s, a):
        next_x = s[0] + a[0]
        next_y = s[1] + a[1]
        if 0 <= next_x < self.size and 0 <= next_y < self.size:
            return next_x, next_y
        else:
            return s

    def getSuccessors(self, s):
        successors = []
        if self.isTerminate(s):
            return successors
        for a in self.actions.values():
            successors.append(self.nextState(s, a))
        return successors

def greedy_policy(pre_value, state_reward, i, j):
    value = 0
    actions = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
    bestPos = actions[0]
    best_count = 0
    for pos in actions:
        if pre_value[pos] > pre_value[bestPos]:
            value = state_reward + pre_value[pos]
            bestPos = pos
            best_count = 0
        elif pre_value[pos] == pre_value[bestPos]:
            best_count += 1
            value += state_reward + pre_value[pos]
    return value / best_count

if __name__ == '__main__':
    grid = Grid()
    for i in range(200):
        print(str(i) + '------------------------------------------------')
        grid.performOneIter()
        print(grid.values)