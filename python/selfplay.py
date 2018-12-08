from config import config as c
import os
import time
import random
from multiprocessing import Pool
import merge

def get_best_index():
    return int(os.listdir(c.weights_dir + "/best")[0])

def run_selfplay(best_index):
    os.system("../c/build/gomokuer.exe selfplay -w {w_dir}/{w_name}.pkl -r {r} -s {seed} -v 0".format(
        w_dir = c.weights_dir,
        w_name = best_index,
        r = 1,
        seed = random.random()*10000))

def run_selfplay_multiprocessing():
    p = Pool(c.selfplay_process_count)
    for i in range(c.selfplay_target_rounds):
        p.apply_async(run_selfplay, args = (get_best_index(),))
    p.close()
    p.join()

if __name__ == "__main__":
    # run_selfplay("0.pkl")
    print("Selfplaying...")
    run_selfplay_multiprocessing()
    merge.run()
    print("Selfplay Finished!")