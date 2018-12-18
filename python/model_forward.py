import numpy as np
import torch
from torch.autograd import Variable
import resnet
from config import config as c
import time


class Model:
    def __init__(self, is_train=True):
        self.resnet = resnet.ResNet()
        self.resnet.cuda()

        if is_train:
            self.optimizer = torch.optim.SGD(
                self.resnet.parameters(),
                lr=c.learning_rate,
                momentum=0.9,
                weight_decay=1E-4)
            self.policy_loss_fn = MultiLableCrossEntropy()
            self.value_loss_fn = torch.nn.MSELoss()

    def evaluate(self, observations):
        """
        observation: [None, 2, c.board_size, c.board_size] array

        return policy[None, c.board_size * c.board_size], value[None]
        """

        assert self.weight_ready
        count = np.size(observations, 0)


        with torch.no_grad():
            observations = Variable(
                torch.from_numpy(observations).float().cuda())

        self.resnet.eval()
        p, v = self.resnet.forward(observations)
        p = p.data.cpu().numpy()
        v = v.data.cpu().numpy()

        p = np.reshape(p, [count, 1, c.board_size, c.board_size])
        v = np.reshape(v, [count])

        p = np.reshape(p, [count, c.board_size * c.board_size])

        return p, v

    def load_weight(self, path):
        print("loading {0}".format(path))
        self.resnet.load_state_dict(torch.load(path))
        self.weight_ready = True
        print("loaded")
