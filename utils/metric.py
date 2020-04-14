# Author: Jingxiao Gu
# Description: Metric Code for AliProducts Recognition Competition

import torch

def GAP(predicts: torch.Tensor, confs: torch.Tensor, targets: torch.Tensor) -> float:
    ''' Simplified GAP@1 metric: only one prediction per sample is supported '''
    assert len(predicts.shape) == 1
    assert len(confs.shape) == 1
    assert len(targets.shape) == 1
    assert predicts.shape == confs.shape and confs.shape == targets.shape

    _, indices = torch.sort(confs, descending=True)

    confs = confs.cpu().numpy()
    predicts = predicts[indices].cpu().numpy()
    targets = targets[indices].cpu().numpy()

    res, true_pos = 0.0, 0

    for i, (c, p, t) in enumerate(zip(confs, predicts, targets)):
        rel = int(p == t)
        true_pos += rel

        res += true_pos / (i + 1) * rel

    res /= targets.shape[0]
    return res

def MAP(predicts: torch.Tensor, confs: torch.Tensor, targets: torch.Tensor, num_class: int) -> (list, list):
    ''' Simplified GAP@1 metric: only one prediction per sample is supported '''
    assert len(predicts.shape) == 1
    assert len(confs.shape) == 1
    assert len(targets.shape) == 1
    assert predicts.shape == confs.shape and confs.shape == targets.shape

    _, indices = torch.sort(confs, descending=True)

    confs = confs.cpu().numpy()
    predicts = predicts[indices].cpu().numpy()
    targets = targets[indices].cpu().numpy()

    true_list = [0] * num_class
    pred_list = [0] * num_class

    for i, (c, p, t) in enumerate(zip(confs, predicts, targets)):
        true_list[t] += 1.0
        pred_list[t] += int(p == t)

    return pred_list, true_list

class AverageMeter:
    ''' Computes and stores the average and current value '''
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MapAverageMeter:
    ''' Computes and stores the average and current value '''
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.pred_map = [0.0] * self.num_classes
        self.true_map = [0.0] * self.num_classes
        self.simple_map = [0.0] * self.num_classes
        self.avg_map = 0.0
        self.sum = 0.0
        self.error = 0.0
        self.count = 0

    def update(self, pred_list: list, true_list: list) -> None:
        for idx in range(len(pred_list)):
            if true_list[idx] > 0:
                self.pred_map[idx] += pred_list[idx]
                self.true_map[idx] += true_list[idx]

        self.count = 0
        self.sum = 0
        for idx in range(len(pred_list)):
            if self.true_map[idx] > 0:
                self.count += 1
                self.simple_map[idx] = self.pred_map[idx] / self.true_map[idx]
                self.sum += self.simple_map[idx]
        self.avg_map = self.sum / self.count
        self.error = 1.0 - self.avg_map