from game import State
from pv_mcts import pv_mcts_scores, pv_mcts_scores_list
from dual_network import DN_OUTPUT_SIZE
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
import pickle
import os
import itertools

SP_GAME_COUNT = 500 # セルフプレイを行うゲーム数
SP_TEMPERATURE = 1.0 # ボルツマン分布の温度パラメータ

# 先手プレイヤーの価値
def first_player_value(ended_state: State):
    if ended_state.is_draw:
        return 0
    if ended_state.is_lose:
        return -1 if ended_state.is_first_player() else 1
    return 1 if ended_state.is_first_player() else -1

# 学習データの保存
def write_data(history):
    now = datetime.now()
    os.makedirs('./data/', exist_ok=True)
    path = f'./data/{now.year:04}{now.month:02}{now.day:02}{now.hour:02}{now.minute:02}{now.second:02}.history'
    with open(path, mode='wb') as f:
        pickle.dump(history, f)

# 1ゲームの試行
def play(model):
    history = []
    state = State()
    while True:
        # ゲーム終了
        if state.is_done:
            break
        # 合法手の確率分布
        scores = pv_mcts_scores(model, state, SP_TEMPERATURE)
        # 学習データの状態と方策
        policies = [0] * DN_OUTPUT_SIZE
        for action, policy in zip(state.legal_actions(), scores):
            policies[action] = policy
        history.append([[state.pieces, state.enemy_pieces], policies, None, state.turn])

        # 行動
        action = np.random.choice(state.legal_actions(), p=scores)
        state = state.next(action)
    # 学習データに価値を追加
    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value if history[i][3] == 0 else -value
    return history

# Nゲームの試行
def play_parallel(model, play_num:int):
    history = [[] for i in range(play_num)]
    state = [State()] * play_num
    while True:
        # ゲーム終了
        if all(s.is_done for s in state):
            break
        # 合法手の確率分布
        scores = pv_mcts_scores_list(model, state, SP_TEMPERATURE)
        # 学習データの状態と方策
        policies = [[0] * DN_OUTPUT_SIZE for i in range(play_num)]
        for i in range(play_num):
            if state[i].is_done:
                continue
            for action, policy in zip(state[i].legal_actions(), scores[i]):
                policies[i][action] = policy
            history[i].append([[state[i].pieces, state[i].enemy_pieces], policies[i], None, state[i].turn])

            # 行動
            action = np.random.choice(state[i].legal_actions(), p=scores[i])
            state[i] = state[i].next(action)

    # 学習データに価値を追加
    for i in range(play_num):
        value = first_player_value(state[i])
        for j in range(len(history[i])):
            history[i][j][2] = value if history[i][j][3] == 0 else -value
    return history


# セルフプレイ
def self_play():
    # history = []
    model = load_model('./model/best.h5')
    # for i in range(SP_GAME_COUNT):
    #     h = play(model)
    #     history.extend(h)
    #     print(f'\rSelfPlay {i+1}/{SP_GAME_COUNT}', end='')
    # print('')
    history = list(itertools.chain.from_iterable(play_parallel(model, SP_GAME_COUNT)))
    # 学習データの保存
    write_data(history)
    # モデルの破棄
    K.clear_session()
    del model

if __name__ == '__main__':
    model = load_model('./model/best.h5')

    import time

    start1 = time.time()
    for i in range(10):
        h = play(model)
        print(f'game {i}')
        for turn, e in enumerate(h):
            print(f'turn {turn}: {e[3]}, {e[2]}')
            print(f'{State(e[0][0], e[0][1], e[3])}')
        print(f'first player valud = {first_player_value(State(h[-1][0][0], h[-1][0][1], h[-1][3]))}')
    elapsed_time1 = time.time() - start1

    print("======================")

    start2 = time.time()
    # h = play_parallel(model, 100)
    # for idx, g in enumerate(h):
    #     print(f'game {idx}')
    #     for turn, e in enumerate(g):
    #         print(f'turn {turn} {e[2]}')
    #         print(f'{State(e[0][0], e[0][1])}')
    elapsed_time2 = time.time() - start2
    print(f'time1:{elapsed_time1}')
    print(f'time2:{elapsed_time2}')

    # self_play()