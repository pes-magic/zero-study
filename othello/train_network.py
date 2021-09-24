from dual_network import DN_INPUT_SHAPE
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
import pickle

RN_EPOCHS = 10 # 学習回数
rotate90 = [
    56, 48, 40, 32, 24, 16,  8,  0,
    57, 49, 41, 33, 25, 17,  9,  1,
    58, 50, 42, 34, 26, 18, 10,  2,
    59, 51, 43, 35, 27, 19, 11,  3,
    60, 52, 44, 36, 28, 20, 12,  4,
    61, 53, 45, 37, 29, 21, 13,  5,
    62, 54, 46, 38, 30, 22, 14,  6,
    63, 55, 47, 39, 31, 23, 15,  7,
    64
]

mirror = [
     7,  6,  5,  4,  3,  2,  1,  0,
    15, 14, 13, 12, 11, 10,  9,  8,
    23, 22, 21, 20, 19, 18, 17, 16,
    31, 30, 29, 28, 27, 26, 25, 24,
    39, 38, 37, 36, 35, 34, 33, 32,
    47, 46, 45, 44, 43, 42, 41, 40,
    55, 54, 53, 52, 51, 50, 49, 48,
    63, 62, 61, 60, 59, 58, 57, 56,
    64
]

# 学習データの読み込み
def load_data():
    history_path = sorted(Path('./data').glob('*.history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)

def copy_history(board, policies, values):
    size = len(board)
    mrx0 = [0] * 64
    mrx1 = [0] * 64
    mry  = [0] * 65
    rx0 = [0] * 64
    rx1 = [0] * 64
    ry  = [0] * 65
    tmp = [0] * 65

    for i in range(size):
        for j in range(64):
            mrx0[j] = board[i][0][mirror[j]]
            mrx1[j] = board[i][1][mirror[j]]
            rx0[j] = board[i][0][j]
            rx1[j] = board[i][1][j]
        for j in range(65):
            mry[j] = policies[i][mirror[j]]
            ry[j] = policies[i][j]
        board.append([mrx0.copy(), mrx1.copy()])
        policies.append(mry.copy())
        values.append(values[i])

        for k in range(3):
            for j in range(64):
                tmp[j] = rx0[rotate90[j]]
            for j in range(64):
                rx0[j] = tmp[j]
            for j in range(64):
                tmp[j] = rx1[rotate90[j]]
            for j in range(64):
                rx1[j] = tmp[j]
            for j in range(65):
                tmp[j] = ry[rotate90[j]]
            for j in range(65):
                ry[j] = tmp[j]
            board.append([rx0.copy(), rx1.copy()])
            policies.append(ry.copy())
            values.append(values[i])
            for j in range(64):
                mrx0[j] = rx0[mirror[j]]
                mrx1[j] = rx1[mirror[j]]
            for j in range(65):
                mry[j] = ry[mirror[j]]
            board.append([mrx0.copy(), mrx1.copy()])
            policies.append(mry.copy())
            values.append(values[i])


# デュアルネットワークの学習
def train_network():
    history = load_data()
    xs, y_policies, y_values = zip(*history)

    xs = list(xs)
    y_policies = list(y_policies)
    y_values = list(y_values)
    copy_history(xs, y_policies, y_values)

    # 学習用に入力データのシェイプ変換
    a, b, c = DN_INPUT_SHAPE
    xs = np.array(xs)
    xs = xs.reshape(len(xs), c, a, b).transpose(0, 2, 3, 1)
    y_policies = np.array(y_policies)
    y_values = np.array(y_values)

    # ベストプレイヤーのモデル読み込み
    model = load_model('./model/best.h5')

    # モデルのコンパイル
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')

    # 学習率
    def step_decay(epoch):
        if epoch >= 8:
            return 0.00025
        if epoch >= 5:
            return 0.0005
        return 0.001
    lr_decay = LearningRateScheduler(step_decay)

    # 出力
    print_callback = LambdaCallback(
        on_epoch_begin=lambda epoch,logs:
            print(f'\rTrain {epoch+1}/{RN_EPOCHS}', end=''))

    # 学習の実行
    model.fit(xs, [y_policies, y_values], batch_size=512, epochs=RN_EPOCHS, verbose=0, callbacks=[lr_decay, print_callback])
    print('')

    # 最新プレイヤーのモデルの保存
    model.save('./model/latest.h5')

    # モデルの破棄
    K.clear_session()
    del model

if __name__ == '__main__':
    train_network()
