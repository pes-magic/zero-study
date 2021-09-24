from dual_network import create_dual_network
from self_play import self_play
from train_network import train_network
from evaluate_network import evaluate_network
import time

create_dual_network()

for i in range(100):
    print(f'Train{i}=======================')
    start = time.time()
    self_play()
    train_network()
    update_best_player = evaluate_network()
    elapsed_time = time.time() - start
    print(f'elapsed_time:{elapsed_time}[sec]')
