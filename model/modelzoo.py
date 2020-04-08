from model import resnet
from model import xception
from model import senet

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
    elif model_type == 'senet154':
        model = senet.senet154(num_classes=num_classes, pretrained='imagenet')
    elif model_type == 'se_resnet50':
        model = senet.se_resnet50(num_classes=num_classes, pretrained='imagenet')
    elif model_type == 'se_resnet101':
        model = senet.se_resnet101(num_classes=num_classes, pretrained='imagenet')
    elif model_type == 'se_resnet152':
        model = senet.se_resnet152(num_classes=num_classes, pretrained='imagenet')
    elif model_type == 'se_resnext50_32x4d':
        model = senet.se_resnext50_32x4d(num_classes=num_classes, pretrained='imagenet')
    elif model_type == 'se_resnext101_32x4d':
        model = senet.se_resnext101_32x4d(num_classes=num_classes, pretrained='imagenet')
    else:
        model = resnet.resnet50(pretrained=True)

    return model