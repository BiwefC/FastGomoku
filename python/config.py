class Config(object):
    def __init__(self):
        # board
        self.board_size = 15

        # weight
        self.weights_dir = "../weights"

        # hyper
        self.filters = 32
        self.value_hidden_units = 128
        self.value_loss_factor = 1.0
        self.batchnorm_momentum = 0.1

        # train
        self.recent_count = 10000
        self.batch_size = 256
        self.learning_rate = 5e-4
        self.train_count = 2
        self.train_epoch = 1

        # selfplay
        self.selfplay_process_count = 3
        self.selfplay_target_rounds = 500

        # evaluate
        self.eval_process_count = 3
        self.eval_target_rounds = 50

config = Config()