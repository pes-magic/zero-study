from game import State
from pv_mcts import pv_mcts_action
from tensorflow.keras.models import load_model
import sys

if __name__ == '__main__':
    human_turn = 0
    if len(sys.argv) >= 2:
        human_turn = int(sys.argv[1])
    if human_turn != 0:
        human_turn = 1
    model = load_model('./model/best.h5')
    state = State()
    next_action = pv_mcts_action(model, 0.0)

    while not state.is_done:
        print(state)
        if state.turn == human_turn:
            legal_actions = state.legal_actions()
            while True:
                print("Your Turn: ", end='')
                try:
                    val = int(input())
                    if val not in legal_actions:
                        continue
                    state = state.next(val)
                    break
                except:
                    state = State()
                    break
        else:
            action = next_action(state)
            print(f'AI Action: {action}')
            state = state.next(action)

    print(state)
    if state.is_draw:
        print('Draw')
    elif state.is_lose:
        print('Second Win' if state.is_first_player() else 'First Win')
    else:
        print('First Win' if state.is_first_player() else 'Second Win')
