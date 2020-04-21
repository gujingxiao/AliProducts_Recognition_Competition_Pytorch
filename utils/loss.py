import torch
import torch.nn as nn
import numbers
from torch.nn import functional as F
from utils.common import euclidean_dist, cosine_dist
from utils.common import get_mask

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, labels):
        loss = self.ce(inputs, labels)
        return loss

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

class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=255,
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        label[[a, torch.arange(label.size(1)), *b]] = 0

        if self.reduction == 'mean':
            loss = -torch.sum(torch.sum(logs*label, dim=1)) / n_valid
        elif self.reduction == 'none':
            loss = -torch.sum(logs*label, dim=1)
        return loss


# From: https://github.com/CHENGY12/DMML/blob/master/loss/triplet.py
class TripletLoss(nn.Module):
    """
    Batch hard triplet loss.
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        if not (isinstance(margin, numbers.Real) or margin == 'soft'):
            raise Exception('Invalid margin parameter for triplet loss.')
        self.margin = margin

    def forward(self, feature, label):
        distance = euclidean_dist(feature, feature, squared=False)

        positive_mask = get_mask(label, 'positive')
        hardest_positive = (distance * positive_mask.float()).max(dim=1)[0]

        negative_mask = get_mask(label, 'negative')
        max_distance = distance.max(dim=1)[0]
        not_negative_mask = ~(negative_mask.data)
        negative_distance = distance + max_distance * (not_negative_mask.float())
        hardest_negative = negative_distance.min(dim=1)[0]

        diff = hardest_positive - hardest_negative
        if isinstance(self.margin, numbers.Real):
            tri_loss = (self.margin + diff).clamp(min=0).mean()
        else:
            tri_loss = F.softplus(diff).mean()

        return tri_loss

# From: https://github.com/CHENGY12/DMML/blob/master/loss/dmml.py
class DMMLLoss(nn.Module):
    """
    DMML loss with center support distance and hard mining distance.
    Args:
        num_support: the number of support samples per class.
        distance_mode: 'center_support' or 'hard_mining'.
    """
    def __init__(self, num_support, distance_mode='hard_mining', margin=0.4, gid=None):
        super().__init__()

        if not distance_mode in ['center_support', 'hard_mining']:
            raise Exception('Invalid distance mode for DMML loss.')
        if not isinstance(margin, numbers.Real):
            raise Exception('Invalid margin parameter for DMML loss.')

        self.num_support = num_support
        self.distance_mode = distance_mode
        self.margin = margin
        self.gid = gid

    def forward(self, feature, label):
        feature = feature.cpu()
        label = label.cpu()
        classes = torch.unique(label)  # torch.unique() is cpu-only in pytorch 0.4
        if self.gid is not None:
            feature, label, classes = feature.cuda(self.gid), label.cuda(self.gid), classes.cuda(self.gid)
        num_classes = len(classes)
        num_query = label.eq(classes[0]).sum() - self.num_support

        support_inds_list = list(map(
            lambda c: label.eq(c).nonzero()[:self.num_support].squeeze(1), classes))
        query_inds = torch.stack(list(map(
            lambda c: label.eq(c).nonzero()[self.num_support:], classes))).view(-1)
        query_samples = feature[query_inds]

        if self.distance_mode == 'center_support':
            center_points = torch.stack([torch.mean(feature[support_inds], dim=0)
                for support_inds in support_inds_list])
            dists = euclidean_dist(query_samples, center_points)
        elif self.distance_mode == 'hard_mining':
            dists = []
            max_self_dists = []
            for i, support_inds in enumerate(support_inds_list):
                # dist_all = euclidean_dist(query_samples, feature[support_inds])
                dist_all = cosine_dist(query_samples, feature[support_inds])
                max_dist, _ = torch.max(dist_all[i*num_query:(i+1)*num_query], dim=1)
                min_dist, _ = torch.min(dist_all, dim=1)
                dists.append(min_dist)
                max_self_dists.append(max_dist)
            dists = torch.stack(dists).t()
            # dists = torch.clamp(torch.stack(dists).t() - self.margin, min=0.0)
            for i in range(num_classes):
                dists[i*num_query:(i+1)*num_query, i] = max_self_dists[i]

        log_prob = F.log_softmax(-dists, dim=1).view(num_classes, num_query, -1)

        target_inds = torch.arange(0, num_classes)
        if self.gid is not None:
            target_inds = target_inds.cuda(self.gid)
        target_inds = target_inds.view(num_classes, 1, 1).expand(num_classes, num_query, 1).long()

        dmml_loss = -log_prob.gather(2, target_inds).squeeze().view(-1).mean()

        batch_size = feature.size(0)
        l2_loss = torch.sum(feature ** 2) / batch_size
        dmml_loss += 0.002 * 0.25 * l2_loss

        return dmml_loss