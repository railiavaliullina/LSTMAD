from torch import nn
import torch
import numpy as np
import os
import time
import warnings
from scipy.stats import multivariate_normal

from torchvision.utils import save_image

from dataloader.dataloader import get_dataloaders
from models.LSTMAD import LSTM_AD
from utils.logging import Logger


class Trainer(object):
    def __init__(self, cfg):
        """
        Class for initializing and performing training procedure.
        :param cfg: train config
        """
        self.cfg = cfg
        self.dl_train, self.dl_valid = get_dataloaders()
        self.model = self.get_model()
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()
        self.logger = Logger(self.cfg)

    def get_model(self):
        model = LSTM_AD(self.cfg)
        return model  # .cuda()

    @staticmethod
    def get_criterion():
        """
        Gets criterion.
        :return: criterion
        """
        criterion = nn.MSELoss()
        return criterion

    def get_optimizer(self):
        """
        Gets optimizer.
        :return: optimizer
        """
        optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.cfg.lr,
                                       'weight_decay': self.cfg.weight_decay}])
        return optimizer

    def restore_model(self):
        """
        Restores saved model.
        """
        if self.cfg.load_saved_model:
            print(f'Trying to load checkpoint from epoch {self.cfg.epoch_to_load}...')
            try:
                checkpoint = torch.load(self.cfg.checkpoints_dir + f'/checkpoint_{self.cfg.epoch_to_load}.pth')
                load_state_dict = checkpoint['model']
                self.model.load_state_dict(load_state_dict)
                self.start_epoch = checkpoint['epoch'] + 1
                self.global_step = checkpoint['global_step'] + 1
                self.optimizer.load_state_dict(checkpoint['opt'])
                print(f'Loaded checkpoint from epoch {self.cfg.epoch_to_load}.')
            except FileNotFoundError:
                print('Checkpoint not found')

    def save_model(self):
        """
        Saves model.
        """
        if self.cfg.save_model and self.epoch % self.cfg.epochs_saving_freq == 0:
            print('Saving current model...')
            state = {
                'model': self.model.state_dict(),
                'epoch': self.epoch,
                'global_step': self.global_step,
                'opt': self.optimizer.state_dict()
            }
            if not os.path.exists(self.cfg.checkpoints_dir):
                os.makedirs(self.cfg.checkpoints_dir)

            path_to_save = os.path.join(self.cfg.checkpoints_dir, f'checkpoint_{self.epoch}.pth')
            torch.save(state, path_to_save)
            print(f'Saved model to {path_to_save}.')

    def evaluate(self, dl, set_type):
        """
        Evaluates model performance. Calculates and logs model accuracy on given data set.
        :param dl: train or test dataloader
        :param set_type: 'train' or 'test' data type
        """
        # if not os.path.exists(self.cfg.eval_plots_dir + f'eval_{set_type}'):
        #     os.makedirs(self.cfg.eval_plots_dir + f'eval_{set_type}')
        all_predictions, losses, all_inputs = [], [], []

        self.model.eval()
        with torch.no_grad():
            print(f'Evaluating on {set_type} data...')
            eval_start_time = time.time()

            dl_len = len(dl)
            for i, batch in enumerate(dl):
                b_size, seq_len, _ = batch[0].size()
                loss, preds = self.make_training_step(batch, make_update=False, return_preds=True)
                losses.append(loss)
                all_predictions.append(preds)
                all_inputs.append(batch[0].squeeze(-1))

                if i % 50 == 0:
                    print(f'iter: {i}/{dl_len}')

            mean_loss = np.mean(losses)
            print(f'Loss on {set_type} data: {mean_loss}')

            self.logger.log_metrics(names=[f'eval/{set_type}/loss'], metrics=[mean_loss], step=self.epoch)
            print(f'Evaluating time: {round((time.time() - eval_start_time) / 60, 3)} min')

            all_predictions = torch.cat(all_predictions, 0)
            all_inputs = torch.cat(all_inputs, 0)
            errors = [[] for _ in range(seq_len)]

            for i in range(seq_len):
                if i > 0:
                    cur_input = all_inputs[:, i]
                    ind = 0
                    for j in range(i - 1, max(-1, i - self.cfg.l - 1), -1):
                        errors[i].append(all_predictions[:, i - j, ind] - cur_input)
                        ind += 1

            samples_num = all_predictions.size(0)
            mu = torch.zeros((self.cfg.l, samples_num))
            for e in errors[1:]:
                for idx, comp in enumerate(e):
                    mu[idx] += comp

            mu = mu.mean(1)
            diff = all_predictions.reshape(-1, self.cfg.l) - mu
            sigma = torch.matmul(diff.T, diff) / diff.size(0)
            mu, sigma = mu.numpy(), sigma.numpy()

        n = mu.shape[0]
        p = [[[] for _ in range(seq_len - 10)] for _ in range(samples_num)]
        anomaly_scores = [[[] for _ in range(seq_len - 10)] for _ in range(samples_num)]
        errors_ = torch.stack([torch.stack(e) for e in errors[10:]]).transpose(2, 0).transpose(2, 1)#.reshape(-1, 10)

        # aa = multivariate_normal.logpdf(errors_, mean=mu, cov=sigma)

        for i in range(1, samples_num):
            for j in range(seq_len - 10):
                x = errors_[i][j].numpy()
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        p[i][j] = (1 / (((2 * np.pi) ** (n / 2)) * (np.linalg.det(sigma) + 1e-6) ** 0.5)) * \
                                  np.exp(-0.5 * ((x - mu).T @ (np.linalg.inv(sigma)) @ (x - mu)))
                    except Warning as e:
                        print('error found:', e)
                        a=1
                anomaly_scores[i][j] = 1 / (p[i][j] + 1e-6)
        self.model.train()
        return np.stack([np.asarray(a) for a in anomaly_scores[1:]]), mu, sigma

    def make_training_step(self, batch, make_update=True, return_preds=False):
        """
        Makes single training step.
        :param batch: current batch containing input vector and it`s label
        :return: loss on current batch
        """
        x, _ = batch
        b_size, seq_len, _ = x.size()
        outputs = self.model(x).view(b_size, seq_len, -1)

        targets = []
        for i in range(seq_len):
            target = x[:, i + 1: min(i + 10 + 1, seq_len)].squeeze(-1)
            if target.size(1) < 10:
                target = torch.cat([target, torch.ones((b_size, 10 - target.size(1))) * -1], 1)
            targets.append(target)
        targets = torch.stack(targets).transpose(1, 0)

        losses = []
        for i in range(b_size):
            outputs_, targets_ = outputs[i].reshape(-1), targets[i].reshape(-1)
            loss_ = self.criterion(outputs_[targets_ != -1], targets_[targets_ != -1])
            losses.append(loss_)

        loss = torch.stack(losses).mean()  # sum()

        if make_update:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if return_preds:
            return loss.item(), outputs
        return loss.item()

    def train(self):
        """
        Runs training procedure.
        """
        total_training_start_time = time.time()
        self.start_epoch, self.epoch, self.global_step = 0, -1, 0

        # restore model if necessary
        self.restore_model()

        # evaluate on test data
        if self.cfg.evaluate_before_training:
            anomaly_scores, mu, sigma = self.evaluate(self.dl_valid, set_type='valid')

        # start training
        print(f'Starting training...')
        iter_num = len(self.dl_train)
        for epoch in range(self.start_epoch, self.cfg.epochs):
            epoch_start_time = time.time()
            self.epoch = epoch
            print(f'Epoch: {self.epoch}/{self.cfg.epochs}')

            losses = []
            for iter_, batch in enumerate(self.dl_train):
                loss = self.make_training_step(batch)
                self.logger.log_metrics(names=['train/loss'], metrics=[loss], step=self.global_step)

                losses.append(loss)
                self.global_step += 1

                if iter_ % 10 == 0:
                    mean_loss = np.mean(losses[-50:]) if len(losses) > 50 else np.mean(losses)
                    print(f'iter: {iter_}/{iter_num}, loss: {mean_loss}')

            self.logger.log_metrics(names=['train/mean_loss_per_epoch'], metrics=[np.mean(losses)], step=self.epoch)

            # save model
            self.save_model()

            # evaluate on test data
            anomaly_scores, mu, sigma = self.evaluate(self.dl_valid, set_type='valid')

            print(f'Epoch total time: {round((time.time() - epoch_start_time) / 60, 3)} min')

        print(f'Training time: {round((time.time() - total_training_start_time) / 60, 3)} min')
