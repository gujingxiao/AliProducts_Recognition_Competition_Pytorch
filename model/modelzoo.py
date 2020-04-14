from model import resnet
from model import xception
from model import senet
from .res2net_v1b import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s
from .res2net import res2net50_26w_4s, res2net101_26w_4s
from .res2next import res2next50

def get_model(model_type='resnet50', num_classes=1000):
    # TODO: Add more backbones
    if model_type == 'resnet34':
        model = resnet.resnet34(pretrained=True)
    elif model_type == 'resnet50':
        model = resnet.resnet50(pretrained=True)
    elif model_type == 'resnet101':
        model = resnet.resnet101(pretrained=True)
    elif model_type == 'resnet152':
        model = resnet.resnet152(pretrained=True)
    elif model_type == 'resnext50_32x4d':
        model = resnet.resnext50_32x4d(pretrained=True)
    elif model_type == 'resnext101_32x8d':
        model = resnet.resnext101_32x8d(pretrained=True)
    elif model_type == 'res2net_v1b_50':
        model = res2net50_v1b_26w_4s(pretrained=True)
    elif model_type == 'res2net_v1b_101':
        model = res2net101_v1b_26w_4s(pretrained=True)
    elif model_type == 'res2net50_26w_4s':
        model = res2net50_26w_4s(pretrained=True)
    elif model_type == 'res2net101_26w_4s':
        model = res2net101_26w_4s(pretrained=True)
    elif model_type == 'res2next50':
        model = res2next50(pretrained=True)
    elif model_type == 'senet154':
        model = senet.senet154(num_classes=num_classes, pretrained='imagenet')
    else:
        model = resnet.resnet50(pretrained=True)

    return model