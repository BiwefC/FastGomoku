import numpy as np
import data
import model
import os
from config import config as c

# gen = int(input("current weight gen="))
# n_epoch = int(input("n_epoch="))


def run():
    gen = int(os.listdir(c.weights_dir + "/current")[0])
    mod = model.Model()
    mod.load_weight(c.weights_dir + "/{0}.pkl".format(gen))

    print("preparing data...")
    obsv, prob, result = data.load()
    count = np.size(obsv, 0)
    count_train = int(count * 0.8)
    obsv, prob, result = data.shuffle(obsv, prob, result)

    train_obsv = obsv[0:count_train]
    train_prob = prob[0:count_train]
    train_result = result[0:count_train]

    test_obsv = obsv[count_train:count]
    test_prob = prob[count_train:count]
    test_result = result[count_train:count]

    mod.test(test_obsv, test_prob, test_result)

    for d_gen in range(c.train_count):
        for epoch in range(c.train_epoch):
            print("epoch {0} of {1}".format(epoch + 1, c.train_epoch))

            tr_obsv, tr_prob, tr_result = data.augment(train_obsv, train_prob, train_result)
            tr_obsv, tr_prob, tr_result = data.shuffle(tr_obsv, tr_prob, tr_result)

            mod.train(tr_obsv, tr_prob, tr_result)
            # mod.train(train_obsv, train_prob, train_result)
            mod.test(test_obsv, test_prob, test_result)

            mod.save_weight(c.weights_dir + "/{0}.pkl".format(gen + 1))
            with open(c.weights_dir + "/eval/queue/{0}".format(gen + 1), mode="w"):
                pass
        os.rename(c.weights_dir + "/current/{0}".format(gen), c.weights_dir + "/current/{0}".format(gen + 1))
        gen = gen + 1


if __name__ == "__main__":
    run()