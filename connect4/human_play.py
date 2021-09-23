from game import State
from pv_mcts import pv_mcts_action
from tensorflow.keras.models import load_model
from threading import Thread
import tkinter as tk
import sys

class GameUI(tk.Frame):
    def __init__(self, master=None, model=None, human_turn=0):
        tk.Frame.__init__(self, master)
        self.master.title('コネクトフォー')

        self.state = State()
        self.next_action = pv_mcts_action(model, 0.0)
        self.human_turn = human_turn

        self.c = tk.Canvas(self, width = 280, height = 240, highlightthickness = 0)
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

        action = int(event.x/40)
        if action < 0 or 7 <= action:
            return # 範囲外
        # 合法手チェック
        if not (action in self.state.legal_actions()):
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
        x = (index%7)*40+5
        y = (index//7)*40+5
        self.c.create_oval(x, y, x+30, y+30, width=1.0, fill='#FF0000' if first_player else '#FFFF00')

    def on_draw(self):
        self.c.delete('all')
        self.c.create_rectangle(0, 0, 280, 240, width=0.0, fill='#00A0FF')
        for i in range(42):
            x = (i%7)*40+5
            y = (i//7)*40+5
            self.c.create_oval(x, y, x+30, y+30, width=1.0, fill='#FFFFFF')

        for i in range(42):
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