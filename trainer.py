import torch
import torch.nn.functional as F
import numpy as np
import os

from tqdm import tqdm
from utils import logger

class Trainer:
    def __init__(self, train_dataloader, dev_dataloader, model, optimizer):
        self.train_data = train_dataloader
        self.dev_data = dev_dataloader
        self.model = model
        self.optimizer = optimizer

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def calculate_result(self, pred, truth):
        pred = torch.argmax(pred, dim=-1).cpu()
        truth = truth.cpu()
        acc = (pred == truth).sum().item() / truth.shape[0]

        def calcu(string_p, string_t, pred, truth):
            if len(torch.where(eval(string_p))[0]) == 0 or len(torch.where(eval(string_t))[0]) == 0:
                return 0
            records_array = torch.cat([torch.where(eval(string_p))[0], torch.where(eval(string_t))[0]], dim=0).numpy()
            vals, inverse, count = np.unique(records_array, return_inverse=True,
                                             return_counts=True)
            return (len(np.where(count > 1)[0]))

        def CALCU(cate, pred, truth):  # int
            TP = calcu('pred==' + str(cate), 'truth==' + str(cate), pred, truth)
            FP = calcu('pred==' + str(cate), 'truth!=' + str(cate), pred, truth)
            TN = calcu('pred!=' + str(cate), 'truth!=' + str(cate), pred, truth)
            FN = calcu('pred!=' + str(cate), 'truth==' + str(cate), pred, truth)
            precision = 1 if TP + FP == 0 else TP / (TP + FP)
            recall = 1 if TP + FN == 0 else TP / (TP + FN)
            f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            if f1 > 1:
                print(1)
            return np.array((precision, recall, f1))

        p, r, f1 = tuple((CALCU(1, pred, truth) + CALCU(2, pred, truth) + CALCU(0, pred, truth))/3)

        return p, r, f1, acc

    def train_epoch(self, epoch):
        logger.info('Epoch: %2d: Training Model...' % epoch)
        pbar = tqdm(total = len(self.train_data))
        self.model.train()

        losses, precise, recall, f1s, accs = [], [], [], [], []
        for batch in self.train_data:
            idx, x, y = map(lambda i: i.to(self.device), batch)

            self.optimizer.zero_grad()
            out = self.model(x)
            loss = F.nll_loss(out, y)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            p, r, f1, acc = self.calculate_result(out, y)
            precise.append(p)
            recall.append(r)
            f1s.append(f1)
            accs.append(acc)

            pbar.set_description('Epoch: %2d | LOSS: %2.3f | F1: %1.3f | ACC: %1.3f' % (epoch, np.mean(losses), np.mean(f1s), np.mean(accs)))
            pbar.update(1)
        pbar.close()
        logger.info('Epoch: %2d | LOSS: %2.3f | F1: %1.3f | ACC: %1.3f | PRECISION: %1.3f | RECALL: %1.3f' %
                    (epoch, np.mean(losses), np.mean(f1s), np.mean(accs), np.mean(precise), np.mean(recall)))

    def evaluate_epoch(self, epoch):
        # step1: eval p model
        logger.info('Epoch %2d: Evaluating Model...' % epoch)
        self.model.eval()

        losses, precise, recall, f1s, accs = [], [], [], [], []
        for batch in self.dev_data:
            idx, x, y = map(lambda i: i.to(self.device), batch)

            with torch.no_grad():
                out = self.model(x)
            loss = F.nll_loss(out, y)

            p, r, f1, acc = self.calculate_result(out, y)
            losses.append(loss.item())
            precise.append(p)
            recall.append(r)
            f1s.append(f1)
            accs.append(acc)

        logger.info('Epoch: %2d | LOSS: %2.3f | F1: %1.3f | ACC: %1.3f | PRECISION: %1.3f | RECALL: %1.3f' %
                    (epoch, np.mean(losses), np.mean(f1s), np.mean(accs), np.mean(precise), np.mean(recall)))

    def train(self, num_epoch, save_path):
        for epoch in range(num_epoch):
            self.train_epoch(epoch)
            self.evaluate_epoch(epoch)

            # save state dict
            path = os.path.join(save_path, 'state_%d_epoch.pt' % epoch)
            self.save_dict(path)

    def save_dict(self, save_path):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        torch.save(state_dict, save_path)

    def load_dict(self, path):
        state_dict = torch.load(path)

        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
