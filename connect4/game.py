import random
import math

# ゲーム状態
class State:
    def __init__(self, pieces=None, enemy_pieces=None):
        self.pieces = pieces if pieces != None else [0] * 42
        self.enemy_pieces = enemy_pieces if enemy_pieces != None else [0] * 42
        self.mem_lose = None
        self.mem_draw = None
        self.mem_done = None

    def piece_count(self, pieces):
        return sum(pieces)

    def is_lose(self):
        if self.mem_lose is None:
            self.mem_lose = self.is_lose_impl()
        return self.mem_lose

    def is_lose_impl(self):
        def is_comp(x, y, dx, dy):
            for k in range(3):
                x, y = x + dx, y + dy
                if x < 0 or 7 <= x or y < 0 or 6 <= y or self.enemy_pieces[7 * y + x] == 0:
                    return False
            return True
        for j in range(6):
            for i in range(7):
                if self.enemy_pieces[7*j+i] == 0:
                    continue
                for di, dj in [(1,0),(0,1),(1,1),(1,-1)]:
                    if is_comp(i, j, di, dj):
                        return True
        return False

    def is_draw(self):
        if self.mem_draw is None:
            self.mem_draw = self.is_draw_impl()
        return self.mem_draw

    def is_draw_impl(self):
        return self.piece_count(self.pieces) + self.piece_count(self.enemy_pieces) == 42

    def is_done(self):
        if self.mem_done is None:
            self.mem_done = (self.is_draw() or self.is_lose())
        return self.mem_done

    def next(self, action):
        pieces = self.pieces.copy()
        for j in range(5, -1, -1):
            if self.pieces[7*j+action] == 0 and self.enemy_pieces[7*j+action] == 0:
                pieces[7*j+action] = 1
                break
        return State(self.enemy_pieces, pieces)

    def legal_actions(self):
        actions = []
        for i in range(7):
            if self.pieces[i] == 0 and self.enemy_pieces[i] == 0:
                actions.append(i)
        return actions

    def is_first_player(self):
        return self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)

    def __str__(self):
        ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
        res = ''
        for i in range(42):
            if self.pieces[i] == 1:
                res += ox[0]
            elif self.enemy_pieces[i] == 1:
                res += ox[1]
            else:
                res += '-'
            if i % 7 == 6:
                res += '\n'
        return res

# ランダムで行動選択
def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions)-1)]

# for test
if __name__ == '__main__':
    state = State()
    while True:
        if state.is_done():
            break
        state = state.next(random_action(state))
        print(state)
        print()