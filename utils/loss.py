import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, labels):
        logp = self.ce(inputs, labels)
        prob = torch.exp(-logp)
        loss = (self.alpha * (torch.pow((1 - prob), self.gamma)) * logp).mean()
        return loss

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, labels):
        loss = self.ce(inputs, labels)
        return loss