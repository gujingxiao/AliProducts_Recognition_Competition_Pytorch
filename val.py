# Author: Jingxiao Gu
# Description: Val Code for AliProducts Recognition Competition

import time
import torch.nn as nn
import torchvision
from reader.dataloader import *
from utils.metric import *

def val(val_loader: Any, model: Any, criterion: Any,) -> None:
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.eval()
    activation = nn.Softmax(dim=1)
    num_steps = len(val_loader)

    print('val total batches: {}'.format(num_steps))
    end = time.time()

    with torch.no_grad():
        for i, (input_, target) in enumerate(val_loader):
            if i >= num_steps:
                break

            output = model(input_.cuda())

            output = activation(output)
            loss = criterion(output, target.cuda())
            confs, predicts = torch.max(output.detach(), dim=1)
            avg_score.update(GAP(predicts, confs, target))

            losses.update(loss.data.item(), input_.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % LOG_FREQ == 0:
                print('[{}/{}]\t time {:.3f} ({:.3f})\t loss {:.4f} ({:.4f})\t GAP {:.4f} ({:.4f})\t'.format(
                    i, num_steps, batch_time.val, batch_time.avg, losses.val, losses.avg, avg_score.val, avg_score.avg))

    print(' * average GAP on val {:.4f}'.format(avg_score.avg))

if __name__ == '__main__':
    global_start_time = time.time()
    train_loader, val_loader, label_encoder_train, label_encoder_val, num_classes = load_data('val')
    # single GPU: [0]; Multi GPU: [0, 1, ...]
    device_ids = DEVICE_ID

    # Backbone
    # TODO: Add more backbones
    if BACKBONE == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif BACKBONE == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif BACKBONE == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    else:
        model = torchvision.models.resnet50(pretrained=True)

    # Set False for evaluation
    for param in model.parameters():
        param.requires_grad = False

    # Change last two layers to adapt for the dataset and classification
    # This configuration only for resnet, other models should be different
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    criterion = nn.CrossEntropyLoss()
    model.cuda(device_ids[0])
    model = nn.DataParallel(model, device_ids=device_ids)

    checkpoint = torch.load(CHECKPOINT)
    model.load_state_dict(checkpoint, strict=True)

    val(val_loader, model, criterion)
