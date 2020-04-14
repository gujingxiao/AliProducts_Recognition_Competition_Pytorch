# Author: Jingxiao Gu
# Description: Train Code for AliProducts Recognition Competition

import time
from reader.dataloader import *
from utils.metric import *
from utils.margin import *
from utils.loss import *
from model import modelzoo
from optimizer.adam_gc import Adam_GC
from pytorch_metric_learning import losses, miners


def train(train_loader: Any, model: Any, margin: Any, criterion: Any, optimizer: Any, epoch: int, num_classes: Any) -> None:
    print('epoch {}'.format(epoch))
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()
    map_score = MapAverageMeter(num_classes)

    model.train()
    num_steps = min(len(train_loader), MAX_STEPS_PER_EPOCH)

    print('train total batches: {}'.format(num_steps))

    end = time.time()

    for i, (input_, target) in enumerate(train_loader):
        if i >= num_steps:
            break

        output = model(input_.cuda())
        output = margin(output, target.cuda())

        loss = criterion(output, target.cuda())
        confs, predicts = torch.max(output.detach(), dim=1)

        # TODO: Add MAP for evaluation
        avg_score.update(GAP(predicts, confs, target))

        pred_list, true_list = MAP(predicts, confs, target, num_classes)
        map_score.update( pred_list, true_list)

        losses.update(loss.data.item(), input_.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % LOG_FREQ == 0:
            print('Epoch: {} [{}/{}]\t time: {:.3f} ({:.3f})\t loss: {:.4f} ({:.4f})\t GAP: {:.4f} ({:.4f})\t MAP: {:.4f}\t LR: {:.6}'.format(
                epoch, i, num_steps, batch_time.val, batch_time.avg, losses.val, losses.avg,
                avg_score.val, avg_score.avg, map_score.avg_map, optimizer.state_dict()['param_groups'][0]['lr']))

        if i % SAVE_FREQ == 0 and i > 0:
            print('Save weights every {} steps.'.format(SAVE_FREQ))
            torch.save(model.state_dict(), 'weights/aliproducts_recognition_{}_backbone_{}.pkl'.format(BACKBONE, epoch))
            torch.save(margin.state_dict(), 'weights/aliproducts_recognition_{}_margin_{}_{}.pkl'.format(BACKBONE, MARGIN_TYPE, epoch))


        if has_time_run_out():
            break

    print(' * average GAP on train {:.4f}'.format(avg_score.avg))


def val(val_loader: Any, model: Any, margin: Any, criterion: Any, num_classes: Any) -> None:
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()
    map_score = MapAverageMeter(num_classes)

    model.eval()
    num_steps = len(val_loader)

    print('val total batches: {}'.format(num_steps))
    end = time.time()

    with torch.no_grad():
        for i, (input_, target) in enumerate(val_loader):
            if i >= num_steps:
                break

            output = model(input_.cuda())
            output = margin(output, target.cuda())

            loss = criterion(output, target.cuda())
            confs, predicts = torch.max(output.detach(), dim=1)
            avg_score.update(GAP(predicts, confs, target))

            pred_list, true_list = MAP(predicts, confs, target, num_classes)
            map_score.update(pred_list, true_list)

            losses.update(loss.data.item(), input_.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % LOG_FREQ == 0:
                print('[{}/{}]\t time {:.3f} ({:.3f})\t loss {:.4f} ({:.4f})\t GAP {:.4f} ({:.4f})\t MAP: {:.4f}\t'.format(
                    i, num_steps, batch_time.val, batch_time.avg, losses.val, losses.avg, avg_score.val, avg_score.avg, map_score.avg_map))


    print(' * on val, average GAP:{:.4f}\t MAP:{:.4f}\t MEAN ERROR:{:.4f}'.format(avg_score.avg, map_score.avg_map, map_score.error))

def has_time_run_out() -> bool:
    return time.time() - global_start_time > TIME_LIMIT - 500

if __name__ == '__main__':
    global_start_time = time.time()
    train_loader, val_loader, label_encoder_train, label_encoder_val, num_classes = load_data('train')

    # single GPU: [0]; Multi GPU: [0, 1, ...]
    device_ids = DEVICE_ID

    # Backbone
    # TODO: Add more backbones
    model = modelzoo.get_model(BACKBONE, num_classes)
    print('Backbone: {}'.format(BACKBONE))
    # Change last two layers to adapt for the dataset and classification
    # This configuration only for resnet, other models should be different
    model.avg_pool = nn.AdaptiveAvgPool2d(1)

    # Set True for training
    # Consider freeze some layers while finetuning
    if FREEZE_BACKBONE == False:
        if FREEZE_PARTIAL == False:
            for param in model.parameters():
                param.requires_grad = True
        else:
            freeze_count = 0
            for param in model.parameters():
                freeze_count += 1
                if freeze_count < PARTIAL_NUMBER:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            print('{} has {} layers. {} layers will be frozen.'.format(BACKBONE, freeze_count, PARTIAL_NUMBER))
    else:
        for param in model.parameters():
            param.requires_grad = False


    # Make sure out feature dim could match the classifier
    if BACKBONE == 'resnet34':
        in_feature = 512
    else:
        in_feature = 2048

    if MARGIN_TYPE == 'arcMargin':
        margin = losses.ArcFaceLoss
        # margin = ArcMarginProduct(in_feature, num_classes)
    elif MARGIN_TYPE == 'inner':
        margin = InnerProduct(in_feature, num_classes)
    elif MARGIN_TYPE == 'tripletMargin':
        margin = losses.TripletMarginLoss(margin=0.1)

    for param in margin.parameters():
        param.requires_grad = True

    # Make model in device 0
    # Set parallel. Single GPU is also okay.
    model.cuda(device_ids[0])
    margin.cuda(device_ids[0])
    model = nn.DataParallel(model, device_ids=device_ids)
    margin = nn.DataParallel(margin, device_ids=device_ids)

    # if checkpoint is not none, load it as pretrained
    if BACKBONE_CHECKPOINT != '':
        checkpoint = torch.load(BACKBONE_CHECKPOINT)
        model.load_state_dict(checkpoint, strict=False)

    if MARGIN_CHECKPOINT != '':
        checkpoint = torch.load(MARGIN_CHECKPOINT)
        margin.load_state_dict(checkpoint, strict=False)

    # Loss function
    # TODO: Add more loss function
    if LOSS_FUNC == 'focalLoss':
        criterion = FocalLoss()
    else:
        criterion = CELoss()

    # Set optimizer and learning strategy
    if OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': margin.parameters()}], lr=LEARNING_RATE)
    elif OPTIMIZER == 'adam_gc':
        optimizer = Adam_GC([{'params': model.parameters()}, {'params': margin.parameters()}], lr=LEARNING_RATE)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_FACTOR)

    # Training
    for epoch in range(1, NUM_EPOCHS + 1):
        print('-' * 80)
        train(train_loader, model, margin, criterion, optimizer, epoch, num_classes)
        torch.save(model.state_dict(), 'weights/aliproducts_recognition_{}_backbone_{}.pkl'.format(BACKBONE, epoch))
        torch.save(margin.state_dict(), 'weights/aliproducts_recognition_{}_margin_{}_{}.pkl'.format(BACKBONE, MARGIN_TYPE, epoch))
        lr_scheduler.step(epoch)

        val(val_loader, model, margin, criterion, num_classes)

        if has_time_run_out():
            break