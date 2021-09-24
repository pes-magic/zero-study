from game import State
from pv_mcts import pv_mcts_action
from tensorflow.keras.models import load_model
from threading import Thread
import tkinter as tk
import sys

class GameUI(tk.Frame):
    def __init__(self, master=None, model=None, human_turn=0):
        tk.Frame.__init__(self, master)
        self.master.title('オセロ')

        self.state = State()
        self.next_action = pv_mcts_action(model, 0.0)
        self.human_turn = human_turn

        self.c = tk.Canvas(self, width = 320, height = 320, highlightthickness = 0)
        self.c.bind('<Button-1>', self.turn_of_human)
        self.c.pack()

        self.turn_of_ai()

    def is_human_turn(self):
        if self.human_turn == 0:
            return self.state.is_first_player()
        else:
            return not self.state.is_first_player()

    def turn_of_human(self, event):
        # ゲーム終了
        if self.state.is_done():
            self.state = State()
            self.turn_of_ai()
            return

        # 人間のターンでない
        if not self.is_human_turn():
            return

        x = event.x//40
        y = event.y//40
        if x < 0 or 7 < x or y < 0 or 7 < y:
            return # 範囲外
        action = x + y * 8
        # 合法手チェック
        legal_actions = self.state.legal_actions()
        if legal_actions == [64]:
            action = 64
        if action != 64 and not (action in legal_actions):
            return

        self.state = self.state.next(action)
        self.on_draw()
        self.master.after(1, self.turn_of_ai)

    def turn_of_ai(self):
        if self.state.is_done():
            return

        self.on_draw()

        while not self.is_human_turn():
            action = self.next_action(self.state)
            self.state = self.state.next(action)
            self.on_draw()

    def draw_piece(self, index, first_player):
        x = (index%8)*40+5
        y = (index//8)*40+5
        self.c.create_oval(x, y, x+30, y+30, width=1.0, fill='#000000' if first_player else '#FFFFFF')

    def on_draw(self):
        self.c.delete('all')
        self.c.create_rectangle(0, 0, 320, 320, width=0.0, fill='#C69C6C')
        for i in range(1,8):
            self.c.create_line(0, i*40, 320, i*40, width=1.0, fill='#000000')
            self.c.create_line(i*40, 0, i*40, 320, width=1.0, fill='#000000')

        for i in range(64):
            if self.state.pieces[i] == 1:
                self.draw_piece(i, self.state.is_first_player())
            if self.state.enemy_pieces[i] == 1:
                self.draw_piece(i, not self.state.is_first_player())

if __name__ == '__main__':
    human_turn = 0
    if len(sys.argv) >= 2:
        human_turn = int(sys.argv[1])
    model = load_model('./model/best.h5')
    f = GameUI(model=model, human_turn=human_turn)
    f.pack()
    f.mainloop()
    del model