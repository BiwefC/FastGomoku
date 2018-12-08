import numpy as np
import torch
from torch.autograd import Variable
import resnet
import data
from config import config as c
# import config as C
# C_BLOCKS = 3  # deprecated
# C_FILTERS = 32
# C_BATCHNORM_DECAY = 0.9


# NCHW!!!!


class MultiLableCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(MultiLableCrossEntropy).__init__()

    def forward(self, input, target):
        return -torch.dot(torch.log(input).reshape(-1), target.reshape(-1)) / input.data.size()[0]
        # print(s)
        # print(1 / 0)
        # return s


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
        actions = np.random.randint(0, 8, count, int)

        for i in range(count):
            observations[i] = data.transform(
                observations[i], actions[i], inverse=False)

        observations = Variable(
            torch.from_numpy(observations).float().cuda(), volatile=True)

        self.resnet.eval()
        p, v = self.resnet.forward(observations)
        p = p.data.cpu().numpy()
        v = v.data.cpu().numpy()

        p = np.reshape(p, [count, 1, c.board_size, c.board_size])
        v = np.reshape(v, [count])
        for i in range(count):
            p[i] = data.transform(p[i], actions[i], inverse=True)
        p = np.reshape(p, [count, c.board_size * c.board_size])

        return p, v

    def train(self, observation, prob, result):
        """
        observation: [None, 2, c.board_size, c.board_size] array
        prob: [None, c.board_size * c.board_size] array, searched probabalities
        result: [None] array, {-1, +1} represents loss or win

        """
        assert self.weight_ready

        n_samples = np.size(observation, 0)
        n_batches = int(n_samples / c.batch_size)

        observation = torch.from_numpy(observation).float().cuda()
        prob = torch.from_numpy(prob).float().cuda()
        result = torch.from_numpy(result).float().cuda()

        train_policy_loss, train_value_loss = 0.0, 0.0

        self.resnet.train()
        for k in range(0, n_batches):
            if (k + 1) % 100 == 0:
                print("train batch {0} of {1}".format(k + 1, n_batches))

            start, end = k * c.batch_size, (k + 1) * c.batch_size
            obsv_var = Variable(observation[start:end])
            prob_var = Variable(prob[start:end])
            result_var = Variable(result[start:end])

            self.optimizer.zero_grad()
            policy, value = self.resnet.forward(obsv_var)
            policy_loss = self.policy_loss_fn.forward(policy, prob_var)
            value_loss = c.value_loss_factor * self.value_loss_fn.forward(value, result_var.reshape(-1, 1))

            train_policy_loss += policy_loss.data.item()
            train_value_loss += value_loss.data.item()

            loss = policy_loss + value_loss
            # policy_loss.backward(retain_graph=True)
            # value_loss.backward()
            loss.backward()

            self.optimizer.step()

        print("train   policy_loss={0}, value_loss={1}".format(
            train_policy_loss / n_batches,
            train_value_loss / n_batches))

    def test(self, observation, prob, result):
        """
        observation: [None, 2, c.board_size, c.board_size] array
        prob: [None, c.board_size * c.board_size] array, searched probabalities
        result: [None] array, {-1, +1} represents loss or win

        """
        assert self.weight_ready
        print("testing...")
        n_samples = np.size(observation, 0)
        n_batches = int(n_samples / c.batch_size)

        observation = torch.from_numpy(observation).float().cuda()
        prob = torch.from_numpy(prob).float().cuda()
        result = torch.from_numpy(result).float().cuda()

        test_policy_loss, test_value_loss = 0.0, 0.0

        self.resnet.eval()
        for k in range(0, n_batches):
            start, end = k * c.batch_size, (k + 1) * c.batch_size
            
            with torch.no_grad():
                obsv_var = Variable(observation[start:end])
                prob_var = Variable(prob[start:end])
                result_var = Variable(result[start:end])

            policy, value = self.resnet.forward(obsv_var)
            policy_loss = self.policy_loss_fn.forward(policy, prob_var)
            value_loss = c.value_loss_factor * self.value_loss_fn.forward(value, result_var.reshape(-1, 1))

            test_policy_loss += policy_loss.data.item()
            test_value_loss += value_loss.data.item()

        print("test    policy_loss={0}, value_loss={1}".format(
            test_policy_loss / n_batches,
            test_value_loss / n_batches))

    def load_weight(self, path):
        print("loading {0}".format(path))
        self.resnet.load_state_dict(torch.load(path))
        self.weight_ready = True
        print("loaded")

    def random_init_weight(self):
        # self.sess.run(tf.global_variables_initializer())
        self.weight_ready = True

    def save_weight(self, path):
        torch.save(self.resnet.state_dict(), path)
        # saver = tf.train.Saver()
        # saver.save(self.sess, path)
