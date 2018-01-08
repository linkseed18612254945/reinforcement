import numpy as np
import pickle
import time

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS


class State:
    def __init__(self):
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None
        self.end = None

    def winner_end_check(self):
        if self.end is not None:
            return self.end
        results = []
        for i in range(0, BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))
        for i in range(0, BOARD_COLS):
            results.append(np.sum(self.data[:, i]))

        results.append(0)
        for i in range(0, BOARD_ROWS):
            results[-1] += self.data[i, i]
        results.append(0)
        for i in range(0, BOARD_ROWS):
            results[-1] += self.data[i, BOARD_COLS - i - 1]

        for result in results:
            if result == BOARD_ROWS:
                self.winner = 1
                self.end = True
                return self.end
            if result == -BOARD_ROWS:
                self.winner = -1
                self.end = True
                return self.end

        if np.sum(np.abs(self.data)) == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end

        self.end = False
        return self.end

    def next_state(self, i, j, symbol):
        if self.end:
            return self
        newState = State()
        newState.data = np.copy(self.data)
        newState.data[i, j] = symbol
        return newState

    def show(self):
        for i in range(0, BOARD_ROWS):
            print('----------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                token = ''
                if self.data[i, j] == 1:
                    token = 'x'
                if self.data[i, j] == 0:
                    token = ' '
                if self.data[i, j] == -1:
                    token = 'o'
                out += token + ' |'
            print(out)
        print('-----------------')

    def __hash__(self):
        hashVal = 0
        for i in self.data.reshape(BOARD_ROWS * BOARD_COLS):
            if i == -1:
                i = 2
            hashVal = hashVal * 3 + i
        return int(hashVal)


def getAllStatesImpl(currentState, currentSymbol, allStates):
    currentState.show()
    for i in range(0, BOARD_ROWS):
        for j in range(0, BOARD_COLS):
            if currentState.data[i, j] == 0:
                newState = currentState.next_state(i, j, currentSymbol)
                if hash(newState) not in allStates:
                    is_end = newState.winner_end_check()
                    allStates[hash(newState)] = (newState, is_end)
                    if not is_end:
                        getAllStatesImpl(newState, -currentSymbol, allStates)


def build_states():
    current_symbol = 1
    current_state = State()
    all_states = {hash(current_state): (current_state, current_state.winner_end_check())}
    getAllStatesImpl(current_state, current_symbol, all_states)
    return all_states


class Player:
    def __init__(self, step_size=0.1, explore_rate=0.1):
        self.explore_rate = explore_rate
        self.step_size = step_size
        self.estimations = {}
        self.states = []

    def reset(self):
        self.states = []

    def set_symbol(self, symbol):
        self.symbol = symbol
        for state_hash in ALL_STATES:
            state, is_end = ALL_STATES[state_hash]
            if is_end:
                if state.winner == symbol:
                    self.estimations[state_hash] = 1
                else:
                    self.estimations[state_hash] = 0
            else:
                self.estimations[state_hash] = 0.5

    def feed_state(self, state):
        self.states.append(state)

    def feed_reward(self, reward):
        if len(self.states) == 0:
            return
        self.states = [hash(state) for state in self.states]
        target = reward
        for latestState in reversed(self.states):
            value = self.estimations[latestState] + self.step_size * (target - self.estimations[latestState])
            self.estimations[latestState] = value
            target = value
        self.states = []

    def take_action(self):
        state = self.states[-1]
        nextStates = []
        nextPositions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    nextPositions.append([i, j])
                    nextStates.append(hash(state.next_state(i, j, self.symbol)))
        if np.random.binomial(1, self.explore_rate):
            self.states = []
            np.random.shuffle(nextPositions)
            action = nextPositions[0]
            action.append(self.symbol)
            return action
        values = []
        for state, pos in zip(nextStates, nextPositions):
            values.append((self.estimations[state], pos))
        values.sort(key=lambda x: x[0], reverse=True)
        action = values[0][1]
        action.append(self.symbol)
        return action

    def save_policy(self):
        with open('save_policy.pkl', 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open('save_policy.pkl', 'rb') as f:
            self.estimations = pickle.load(f)


class HumanPlayer:
    def __init__(self, stepSize = 0.1, exploreRate=0.1):
        self.symbol = None
        self.currentState = None
        return

    def reset(self):
        return

    def set_symbol(self, symbol):
        self.symbol = symbol
        return

    def feed_state(self, state):
        self.currentState = state
        return

    def feed_reward(self, reward):
        return

    def take_action(self):
        data = int(input("Input your position:"))
        data -= 1
        i = data // int(BOARD_COLS)
        j = data % BOARD_COLS
        if self.currentState.data[i, j] != 0:
            return self.take_action()
        return i, j, self.symbol


class Judger:
    def __init__(self, player1, player2, feedback=True):
        self.p1 = player1
        self.p2 = player2
        self.feedback = feedback
        self.currentPlayer = None
        self.p1Symbol = 1
        self.p2Symbol = -1
        self.p1.set_symbol(self.p1Symbol)
        self.p2.set_symbol(self.p2Symbol)
        self.currentState = State()

    def give_reward(self):
        if self.currentState.winner == self.p1Symbol:
            self.p1.feed_reward(1)
            self.p2.feed_reward(0)
        elif self.currentState.winner == self.p2Symbol:
            self.p1.feed_reward(0)
            self.p2.feed_reward(1)
        else:
            self.p1.feed_reward(0.1)
            self.p2.feed_reward(0.5)

    def feed_currentState(self):
        self.p1.feed_state(self.currentState)
        self.p2.feed_state(self.currentState)

    def reset(self):
        self.p1.reset()
        self.p2.reset()
        self.currentState = State()
        self.currentPlayer = None

    def play(self, show=True):
        self.reset()
        self.feed_currentState()
        while True:
            if self.currentPlayer == self.p1:
                self.currentPlayer = self.p2
            else:
                self.currentPlayer = self.p1
            if show:
                self.currentState.show()
                time.sleep(1)
            [i, j, symbol] = self.currentPlayer.take_action()
            self.currentState = self.currentState.next_state(i, j, symbol)
            self.currentState, is_end = ALL_STATES[hash(self.currentState)]
            self.feed_currentState()
            if is_end:
                if show:
                    self.currentState.show()
                    time.sleep(1)
                if self.feedback:
                    self.give_reward()
                return self.currentState.winner


def train(epochs=20000):
    player1 = Player()
    player2 = Player()
    judger = Judger(player1, player2)
    player1Win = 0.0
    player2Win = 0.0
    for i in range(0, epochs):
        print("Epoch:", i)
        winner = judger.play(show=False)
        if winner == 1:
            player1Win += 1
        if winner == -1:
            player2Win += 1
        judger.reset()
    print(player1Win / epochs)
    print(player2Win / epochs)
    player1.save_policy()
    player2.save_policy()
    return player1, player2


def compete(turns=500):
    player1 = Player(explore_rate=0)
    player2 = Player(explore_rate=0)
    judger = Judger(player1, player2, False)
    player1.load_policy()
    player2.load_policy()
    player1Win = 0.0
    player2Win = 0.0
    for i in range(0, turns):
        print("Epoch", i)
        winner = judger.play(show=True)
        if winner == 1:
            player1Win += 1
        if winner == -1:
            player2Win += 1
        judger.reset()
    print(player1Win / turns)
    print(player2Win / turns)


def play():
    while True:
        player1 = Player(explore_rate=0)
        player2 = HumanPlayer()
        judger = Judger(player1, player2, False)
        player1.load_policy()
        winner = judger.play(True)
        if winner == player2.symbol:
            print("Win!")
        elif winner == player1.symbol:
            print("Lose!")
        else:
            print("Tie!")


if __name__ == '__main__':
    ALL_STATES = build_states()
    print('Start Training...')
    train(20000)
    print('Start Compete...')
    compete()
    play()