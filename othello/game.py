import random
import math

BOARD_SIZE = 8

# ゲーム状態
class State:
    def __init__(self, pieces=None, enemy_pieces=None, depth=0):
        self.depth = depth
        self.dxy = ((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))
        self.pass_end = False
        self.pieces = pieces
        self.enemy_pieces = enemy_pieces
        if pieces is None or enemy_pieces is None:
            self.pieces = [0] * BOARD_SIZE * BOARD_SIZE
            self.enemy_pieces = [0] * BOARD_SIZE * BOARD_SIZE
            half = BOARD_SIZE // 2
            self.pieces[(half-1)*BOARD_SIZE+half] = self.pieces[half*BOARD_SIZE+half-1] = 1
            self.enemy_pieces[(half-1)*BOARD_SIZE+half-1] = self.enemy_pieces[half*BOARD_SIZE+half] = 1
        self.mem_lose = None
        self.mem_draw = None
        self.mem_done = None
        self.actions = []

    def piece_count(self, pieces):
        return sum(pieces)

    def is_lose(self):
        if self.mem_lose is None:
            self.mem_lose = self.is_lose_impl()
        return self.mem_lose

    def is_lose_impl(self):
        return self.is_done() and sum(self.pieces) < sum(self.enemy_pieces)

    def is_draw(self):
        if self.mem_draw is None:
            self.mem_draw = self.is_draw_impl()
        return self.mem_draw

    def is_draw_impl(self):
        return self.is_done() and sum(self.pieces) == sum(self.enemy_pieces)

    def is_done(self):
        if self.mem_done is None:
            self.mem_done = (sum(self.pieces) + sum(self.enemy_pieces) == BOARD_SIZE * BOARD_SIZE or self.pass_end)
        return self.mem_done

    def next(self, action):
        state = State(self.pieces.copy(), self.enemy_pieces.copy(), 1-self.depth)
        if action != BOARD_SIZE * BOARD_SIZE:
            state.is_legal_action_xy(action%BOARD_SIZE, action//BOARD_SIZE, True)
        w = state.pieces
        state.pieces = state.enemy_pieces
        state.enemy_pieces = w

        if action == BOARD_SIZE * BOARD_SIZE and state.legal_actions() == [BOARD_SIZE * BOARD_SIZE]:
            state.pass_end = True
        return state

    def legal_actions(self):
        if self.actions:
            return self.actions
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.is_legal_action_xy(i, j):
                    self.actions.append(i+j*BOARD_SIZE)
        if len(self.actions) == 0:
            self.actions.append(BOARD_SIZE * BOARD_SIZE)
        return self.actions

    def is_legal_action_xy(self, x, y, flip=False):
        def is_legal_action_xy_dxy(x, y, dx, dy):
            x, y = x+dx, y+dy
            if y < 0 or BOARD_SIZE - 1 < y or x < 0 or BOARD_SIZE - 1 < x or self.enemy_pieces[x+y*BOARD_SIZE] != 1:
                return False
            for j in range(BOARD_SIZE):
                x, y = x+dx, y+dy
                if y < 0 or BOARD_SIZE - 1 < y or x < 0 or BOARD_SIZE - 1 < x or (self.enemy_pieces[x+y*BOARD_SIZE] == 0 and self.pieces[x+y*BOARD_SIZE] == 0):
                    return False
                if self.pieces[x+y*BOARD_SIZE] == 1:
                    if flip:
                        for i in range(BOARD_SIZE):
                            x, y = x-dx, y-dy
                            if self.pieces[x+y*BOARD_SIZE] == 1:
                                return True
                            self.pieces[x+y*BOARD_SIZE] = 1
                            self.enemy_pieces[x+y*BOARD_SIZE] = 0
                    return True
            return False

        if self.enemy_pieces[x+y*BOARD_SIZE] == 1 or self.pieces[x+y*BOARD_SIZE] == 1:
            return False
        if flip:
            self.pieces[x+y*BOARD_SIZE] = 1
        flag = False
        for dx, dy in self.dxy:
            if is_legal_action_xy_dxy(x, y, dx, dy):
                flag = True
        return flag


    def is_first_player(self):
        return self.depth == 0

    def __str__(self):
        ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
        res = ''
        for i in range(BOARD_SIZE * BOARD_SIZE):
            if self.pieces[i] == 1:
                res += ox[0]
            elif self.enemy_pieces[i] == 1:
                res += ox[1]
            else:
                res += '-'
            if i % BOARD_SIZE == BOARD_SIZE - 1:
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
