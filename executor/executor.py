
from trainers.trainer import Trainer
import config as cfg


class Executor(object):
    """
    Class for running main class methods which run whole algorithm.
    """
    @staticmethod
    def run():
        trainer = Trainer(cfg)
        trainer.train()


if __name__ == '__main__':
    executor = Executor()
    executor.run()
