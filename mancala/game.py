import random
import math

# ゲーム状態
class State:
    def __init__(self, pieces=None, enemy_pieces=None, turn=0):
        self.pieces = pieces if pieces != None else [4, 4, 4, 4, 4, 4, 0]
        self.enemy_pieces = enemy_pieces if enemy_pieces != None else [4, 4, 4, 4, 4, 4, 0]
        self.turn = turn
        self.check_winner()
        self.is_done = (sum(self.pieces[:6]) == 0 or sum(self.enemy_pieces[:6]) == 0)

    def check_winner(self):
        score = sum(self.pieces)
        enemy_score = sum(self.enemy_pieces)
        self.is_draw = (score == enemy_score)
        self.is_lose = (score < enemy_score)

    def next(self, action):
        pieces = [self.pieces.copy(), self.enemy_pieces.copy()]
        side = 0
        pos = action
        rest = pieces[side][pos]
        pieces[side][pos] = 0
        while rest > 0:
            pos += 1
            if pos == 7 - side:
                pos = 0
                side = 1 - side
            pieces[side][pos] += 1
            rest -= 1
        if side == 0 and pos <= 5 and pieces[0][pos] == 1 and pieces[1][5-pos] > 0:
            pieces[0][6] += pieces[0][pos] + pieces[1][5-pos]
            pieces[0][pos] = 0
            pieces[1][5-pos] = 0
        if pos == 6:
            return State(pieces[0], pieces[1], self.turn)
        return State(pieces[1], pieces[0], 1 - self.turn)

    def legal_actions(self):
        return [i for i in range(6) if self.pieces[i] != 0]

    def is_first_player(self):
        return self.turn == 0

    def __str__(self):
        p = (self.enemy_pieces, self.pieces) if self.is_first_player() else (self.pieces, self.enemy_pieces)
        res = ''
        res += f'   {p[0][5]:2d} {p[0][4]:2d} {p[0][3]:2d} {p[0][2]:2d} {p[0][1]:2d} {p[0][0]:2d}\n'
        res += f'{p[0][6]:2d}                   {p[1][6]:2d}\n'
        res += f'   {p[1][0]:2d} {p[1][1]:2d} {p[1][2]:2d} {p[1][3]:2d} {p[1][4]:2d} {p[1][5]:2d}\n'
        return res

# ランダムで行動選択
def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions)-1)]

# for test
if __name__ == '__main__':
    state = State()
    while True:
        if state.is_done:
            break
        state = state.next(random_action(state))
        print(state)
        print()