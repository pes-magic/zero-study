from game import State
from pv_mcts import pv_mcts_action
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
from shutil import copy
import numpy as np
import time

EN_GAME_COUNT = 200 # 1評価あたりのゲーム数
EN_TEMPERATURE = 1.0 # ボルツマン分布の温度パラメータ

# 先手プレイヤのポイント
def first_player_point(ended_state: State):
    if ended_state.is_draw:
        return 0.5
    if ended_state.is_lose:
        return 0 if ended_state.is_first_player() else 1
    return 1 if ended_state.is_first_player() else 0

# 1ゲームの実行
def play(next_actions):
    state = State()
    while True:
        if state.is_done:
            break
        next_action = next_actions[0 if state.is_first_player() else 1]
        action = next_action(state)
        state = state.next(action)

    return first_player_point(state)

# Nゲームの実行
def play_parallel(next_actions, play_num:int):
    state = [State()] * play_num
    turn = 0
    while True:
        if all(s.is_done for s in state):
            break
        if turn == 0:
            if all(s.is_done or not s.is_first_player() for s in state):
                turn = 1
        else:
            if all(s.is_done or s.is_first_player() for s in state):
                turn = 0
        actions = next_actions[turn](state)
        for i in range(play_num):
            if state[i].is_done:
                continue
            if turn == 0 and not state[i].is_first_player():
                continue
            if turn == 1 and state[i].is_first_player():
                continue
            state[i] = state[i].next(actions[i])

    cnt = [0] * 3
    for s in state:
        if s.is_lose:
            cnt[0 if s.is_first_player() else 2] += 1
        else:
            cnt[1] += 1
    print(cnt)

    return sum([first_player_point(s) for s in state])

def play_parallel2(next_actions, play_num:int):
    state = [State()] * play_num
    turn = 0
    while True:
        if all(s.is_done for s in state):
            break
        if all(s.is_done or ((idx%2 == turn) ^ (s.is_first_player())) for idx, s in enumerate(state)):
            turn = 1 - turn
        actions = next_actions[turn](state)
        for i in range(play_num):
            if state[i].is_done:
                continue
            if (i%2 == turn) ^ state[i].is_first_player():
                continue
            state[i] = state[i].next(actions[i])

    cnt = [0] * 3
    for idx, s in enumerate(state):
        if s.is_draw:
            cnt[1] += 1
        elif s.is_lose:
            if idx % 2 == 0:
                cnt[2 if s.is_first_player() else 0] += 1
            else:
                cnt[0 if s.is_first_player() else 2] += 1
        else:
            if idx % 2 == 0:
                cnt[0 if s.is_first_player() else 2] += 1
            else:
                cnt[2 if s.is_first_player() else 0] += 1

    print(cnt)

    return sum([first_player_point(s) if idx%2 == 0 else 1 - first_player_point(s) for idx, s in enumerate(state)])

# ベストプレイヤの交代
def update_best_player():
    copy('./model/latest.h5', './model/best.h5')
    print('Change BestPlayer')

# ネットワークの評価
def evaluate_network():
    # 最新プレイヤのモデル読み込み
    model0 = load_model('./model/latest.h5')
    # ベストプレイヤのモデル読み込み
    model1 = load_model('./model/best.h5')
    # PV MCTS で行動選択を行う関数
    next_action0 = pv_mcts_action(model0, EN_TEMPERATURE)
    next_action1 = pv_mcts_action(model1, EN_TEMPERATURE)
    next_actions = (next_action0, next_action1)

    # 対戦を繰り返す
    # start = time.time()
    # total_point = 0.0
    # for i in range(EN_GAME_COUNT):
    #     if i%2 == 0:
    #         total_point += play(next_actions)
    #     else:
    #         total_point += 1 - play(list(reversed(next_actions)))
    #     print(f'\rEvaluate {i+1}/{EN_GAME_COUNT}', end='')
    # print('')

    # elapsed_time = time.time() - start
    # print(f'elapsed_time:{elapsed_time}[sec]')

    # # 平均ポイント
    # average_point = total_point / EN_GAME_COUNT
    # print('AveragePoint', average_point)

    # total_point = 0.0
    # total_point += play_parallel(next_actions, EN_GAME_COUNT//2)
    # total_point += EN_GAME_COUNT//2 - play_parallel(list(reversed(next_actions)), EN_GAME_COUNT//2)

    total_point = play_parallel2(next_actions, EN_GAME_COUNT)

    # 平均ポイント
    average_point = total_point / EN_GAME_COUNT
    print('AveragePoint', average_point)

    # モデル破棄
    K.clear_session()
    del model0
    del model1

    # ベストプレイヤの交代
    if average_point > 0.5:
        update_best_player()
        return True
    else:
        return False

if __name__ == '__main__':
    start = time.time()
    evaluate_network()
    elapsed_time = time.time() - start
    print(f'elapsed_time:{elapsed_time}[sec]')
