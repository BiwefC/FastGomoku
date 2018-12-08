from config import config as c
import model
import os

def init_dirs():
    dirs = [
        "../data/backup",
        "../data/raw",
        "../data/train",
        c.weights_dir + "/current",
        c.weights_dir + "/log",
        c.weights_dir + "/best",
        c.weights_dir + "/eval/queue",
        c.weights_dir + "/eval/result",
    ]

    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def init_weights():
    mod = model.Model()
    mod.save_weight(c.weights_dir + "/0.pkl")
    with open(c.weights_dir + "/current/0", mode = 'w'):
        pass
    with open(c.weights_dir + "/best/0", mode = 'w'):
        pass

if __name__ == "__main__":
    init_dirs()
    init_weights()
