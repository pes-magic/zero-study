from game import State
from dual_network import DN_INPUT_SHAPE
from math import sqrt
from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np
import time

PV_EVALUATE_COUNT = 50 # 1推論あたりのシミュレーション回数

def predict(model, state):
    if not isinstance(state, list):
        p, v = predict(model, [state])
        return p[0], v[0]

    # 推論のための入力データのシェイプ変換
    a, b, c = DN_INPUT_SHAPE
    x = np.array([[s.pieces, s.enemy_pieces] for s in state])
    x = x.reshape(len(state), c, a, b).transpose(0, 2, 3, 1).reshape(len(state), a, b, c)

    #推論
    y = model.predict(x, batch_size=256)

    #方策の取得
    policies = []
    value = []

    for i in range(len(state)):
        p = y[0][i][list(state[i].legal_actions())]
        p /= sum(p) if sum(p) else 1
        policies.append(p)

        # 価値の取得
        value.append(y[1][i][0])

    return policies, value

# ノードのリストを試行回数リストに変換
def nodes_to_scores(nodes):
    return [c.n for c in nodes]

def boltzman(xs, temperature):
    xs = [x ** (1/temperature) for x in xs]
    return [x / sum(xs) for x in xs]

# 温度を考慮した試行回数からスコア分布への変換
def nodes_to_temperature_scores(nodes, temperature: float):
    if nodes is None:
        return None
    scores = nodes_to_scores(nodes)
    if temperature == 0:
        # 温度0なら最大値をとる行動を確率1に
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:
        # ボルツマン分布でばらつきを付加
        scores = boltzman(scores, temperature)
    return scores


# モンテカルロ木探索のスコア取得
def pv_mcts_scores(model, state, temperature):
    class Node:
        def __init__(self, state, p):
            self.state = state # 状態
            self.p = p # 方策
            self.w = 0 # 累計価値
            self.n = 0 # 試行回数
            self.child_nodes = None # 子ノード群

        # 局面の価値計算
        def evaluate(self):
            if self.state.is_done:
                value = -1 if self.state.is_lose else 0 if self.state.is_draw else 1
                self.w += value
                self.n += 1
                return value
            if not self.child_nodes:
                policies, value = predict(model, self.state)
                self.w += value
                self.n += 1
                # 子ノード展開
                self.child_nodes = []
                for action, policy in zip(self.state.legal_actions(), policies):
                    self.child_nodes.append(Node(self.state.next(action), policy))
                return value
            else:
                # アーク評価値が最大の子ノードの評価で価値を取得
                next_node = self.next_child_node()
                value = next_node.evaluate()
                if self.state.turn != next_node.state.turn:
                    value = -value # 相手から見た価値なので反転
                self.w += value
                self.n += 1
                return value

        # アーク評価値が最大の子ノードを取得
        def next_child_node(self):
            C_PUCT = 1.0
            t = sum(nodes_to_scores(self.child_nodes))
            pucb_values = []
            for child_node in self.child_nodes:
                pucb_values.append(((-child_node.w if self.state.turn != child_node.state.turn else child_node.w) / child_node.n if child_node.n else 0.0) + C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))
            return self.child_nodes[np.argmax(pucb_values)]


    # 現局面のノード
    root_node = Node(state, 0)

    # 設定した回数の評価
    for _ in range(PV_EVALUATE_COUNT):
        root_node.evaluate()

    # 合法手の確率分布
    return nodes_to_temperature_scores(root_node.child_nodes, temperature)

class Node:
    def __init__(self, state, p):
        self.state = state # 状態
        self.p = p # 方策
        self.w = 0 # 累計価値
        self.n = 0 # 試行回数
        self.child_nodes = None # 子ノード群

    # modelにより評価する状態の取得
    def get_evaluate(self):
        if self.state.is_done:
            return None
        if not self.child_nodes:
            return self.state
        else:
            # アーク評価値が最大の子ノードをチェック
            return self.next_child_node().get_evaluate()

    # 局面の価値計算
    def evaluate(self, predict_policy, predict_value):
        if self.state.is_done:
            value = -1 if self.state.is_lose else 0 if self.state.is_draw else 1
            self.w += value
            self.n += 1
            return value
        if not self.child_nodes:
            self.w += predict_value
            self.n += 1
            # 子ノード展開
            self.child_nodes = [Node(self.state.next(action), policy) for action, policy in zip(self.state.legal_actions(), predict_policy)]
            return predict_value
        else:
            # アーク評価値が最大の子ノードの評価で価値を取得
            value = self.child_nodes[self.mem_next_child].evaluate(predict_policy, predict_value)
            if self.state.turn != self.child_nodes[self.mem_next_child].state.turn:
                value = -value # 相手から見た価値なので反転
            self.w += value
            self.n += 1
            return value

    # アーク評価値が最大の子ノードを取得
    def next_child_node(self):
        C_PUCT = 1.0
        sqrt_t = sqrt(sum(nodes_to_scores(self.child_nodes)))
        pucb_values = [((-child_node.w if self.state.turn != child_node.state.turn else child_node.w) / child_node.n if child_node.n else 0.0) + C_PUCT * child_node.p * sqrt_t / (1 + child_node.n) for child_node in self.child_nodes]
        self.mem_next_child = np.argmax(pucb_values)
        return self.child_nodes[self.mem_next_child]


# list[State]用 (並列シミュレーション)
def pv_mcts_scores_list(model, state, temperature):
    if not isinstance(state, list):
        return pv_mcts_scores(model, state, temperature)

    predict_state = [State()] * len(state)
    predict_policies = []
    predict_value = []

    # 現局面のノード
    root_nodes = [Node(s, 0) for s in state]

    def call_get_evaluate(node: Node):
        return node.get_evaluate()

    def call_evaluate(node: Node):
        node.evaluate()

    # 設定した回数の評価
    for _ in range(PV_EVALUATE_COUNT):
        need_predict = False
        for idx, r in enumerate(root_nodes):
            s = r.get_evaluate()
            if s != None:
                need_predict = True
                predict_state[idx] = s
        if need_predict:
            predict_policies, predict_value = predict(model, predict_state)
        for r, policy, value in zip(root_nodes, predict_policies, predict_value):
            r.evaluate(policy, value)

    # 合法手の確率分布
    return [nodes_to_temperature_scores(r.child_nodes, temperature) for r in root_nodes]


# モンテカルロ木探索で行動選択
def pv_mcts_action(model, temperature=0):
    def pv_mcts_action(state):
        if isinstance(state, list):
            scores = pv_mcts_scores_list(model, state, temperature)
            return [np.random.choice(s.legal_actions(), p=p) if p else None for s, p in zip(state, scores)]
        scores = pv_mcts_scores(model, state, temperature)
        return np.random.choice(state.legal_actions(), p=scores)
    return pv_mcts_action

if __name__ == '__main__':
    path = sorted(Path('./model').glob('*.h5'))[-1]
    model = load_model(str(path))

    state = State()
    start = time.time()
    for i in range(20):
        print(f'\rpredict {i+1}/20')
        y = predict(model, state)
    elapsed_time = time.time() - start
    print(y)
    print(f'elapsed_time:{elapsed_time}[sec]')

    state = [State() for i in range(32)]
    start = time.time()
    for i in range(20):
        print(f'\rpredict_list {i+1}/20')
        y = predict(model, state)
    elapsed_time = time.time() - start
    print(y)
    print(f'elapsed_time:{elapsed_time}[sec]')


    # next_action = pv_mcts_action(model, 1.0)

    # while True:
    #     if state.is_done:
    #         break
    #     action = next_action(state)
    #     state = state.next(action)
    #     print(state)
