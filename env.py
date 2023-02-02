import torch


class TicTacToe:
    def __init__(self):
        self.player = {1: 'X', -1: 'O'}
        self.r_win = 10.0
        self.r_draw = 1.0

    @staticmethod
    def init_state():
        return torch.zeros(3, 3, dtype=torch.int32)

    # return next_state, reward, done
    def step(self, state, action):
        s2d = torch.clone(state)
        s = s2d.view(9)
        if self.is_full(s2d):
            raise ValueError('Board is full')

        if s[action] != 0:
            return s2d, self.r_win * -1, True

        s[action] = self.get_turn(s2d)

        if self.check_win(s2d):
            return s2d, self.r_win, True

        if self.is_full(s2d):
            return s2d, self.r_draw, True

        else:
            return s2d, 0, False

    @staticmethod
    def valid_actions(state):
        s = state.view(9)
        return torch.nonzero(s == 0).view(-1)

    @staticmethod
    def random_action(state):
        s = state.view(9)
        available_actions = torch.nonzero(s == 0)
        actions = available_actions.size()[0]
        if actions == 0:
            raise ValueError('No actions available')

        rand = torch.randint(actions, (1,)).item()
        random_action = available_actions[rand].item()
        return random_action

    @staticmethod
    def render(state):
        print(state)

    @staticmethod
    def get_turn(state):
        return 1 if state.sum() == 0 else -1

    @staticmethod
    def is_full(state):
        return torch.nonzero(state).size(0) == 9

    @staticmethod
    def check_win(state):
        # Check rows
        for i in range(3):
            if abs(state[i, :].sum()) == 3:
                return True

        # Check columns
        for i in range(3):
            if abs(state[:, i].sum()) == 3:
                return True

        # Check diagonals
        if abs(state.diagonal().sum()) == 3:
            return True
        if abs(state.flipud().diagonal().sum()) == 3:
            return True

        return False
