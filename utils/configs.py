# Author: Jingxiao Gu
# Description: config Code for AliProducts Recognition Competition

# 目前支持的模型 Support following models
# ['resnet34'  'resnet50'  'resnet101'  'resnet152'  'resnext50_32x4d'  'resnext101_32x8d']
# ['senet154'  'res2net_v1b_50'  'res2net_v1b_101'  'res2net50_26w_4s'  'res2net101_26w_4s'  'res2next50']
BACKBONE = 'resnest101'
MARGIN_TYPE = 'inner'    # margin类型 Choose margin type ['inner'  'arcMargin'  'tripletMargin']
LOSS_FUNC = 'ceLoss'  # 损失函数类型 Choose loss function ['ceLoss', LabelSmooth, 'focalLoss']
OPTIMIZER = 'adam_gc'    # 优化器类型 Choose optimizer type ['adam'  'adam_gc'  'sgd'  'sgd_gc']
USE_HIERARCHY = False

BACKBONE_CHECKPOINT = ''#'weights/resnest101-22405ba7.pth'#'weights/aliproducts_{}_backbone_4_fp16.pkl'.format(BACKBONE)
MARGIN_CHECKPOINT = ''#'weights/aliproducts_{}_margin_inner_4_fp16.pkl'.format(BACKBONE)
# MARGIN3_CHECKPOINT = 'weights/aliproducts_{}_margin3_inner_1.pkl'.format(BACKBONE)
FREEZE_BACKBONE = True   # 是否冻结主干网络  Whether freeze backbone
FREEZE_PARTIAL = False    # 是否冻结部分网络  Whether freeze part of backbone
PARTIAL_NUMBER = 30      # 如果冻结部分，需要冻结前多少层  if freeze part of backbone, how many layers will be frozen

DEVICE_ID = [0, 1]           # 选择GPU  single GPU: [0]; Multi GPU: [0, 1, ...]
MIXED_PRECISION_TRAIN = True #
MAX_SAMPLES_PER_CLASS = 100  # 类别包含最大数目 train data only less than this number
MIN_SAMPLES_PER_CLASS = 80    # 类别包含最小数目 train data only more than this number

IMAGE_SIZE = 224           # 训练分辨率
DATA_AUGMENTATION = False  # 是否使用数据增强，默认为False
BATCH_SIZE = 1024           # 批大小 batch size
LEARNING_RATE = 0.001      # 初始学习率，默认为0.001  Default 0.001 on beginning
LR_STEP = 2                # 衰减的步长  decay in every LR_STEP
LR_FACTOR = 0.5            # 衰减因子  decay factor in every decay

NUM_WORKERS = 4               # 读取数据的线程数  number of threads for dataloader
MAX_STEPS_PER_EPOCH = 500000  # 每个epoch的最大步数  max steps per epoch
NUM_EPOCHS = 15               # 训练的轮数  total epochs

LOG_FREQ = 10               # 打印频率  log every N steps
SAVE_FREQ = 500             # 保存模型频率  save weights every N steps
TIME_LIMIT = 50 * 60 * 60   # 每个epoch的最大训练时间  time limit for every epoch

DATASET_PATH = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/AliProducts/'  # 数据源文件夹  base path of the dataset
TRAIN_PATH = DATASET_PATH + 'train/'                  # 训练图片路径  path of train
VAL_PATH = DATASET_PATH + 'val/'                      # 验证图片路径  path of val
TEST_PATH = DATASET_PATH + 'test/'                    # 测试图片路径  path of test
TRAIN_JSON = DATASET_PATH + 'train.json'
VAL_JSON = DATASET_PATH + 'val.json'
TRAIN_CSV = DATASET_PATH + 'train.csv'
VAL_CSV = DATASET_PATH + 'val.csv'
HIERARCHY_VAL_CSV = DATASET_PATH + 'val_hierarchy.csv'
TREE_FILE = DATASET_PATH + 'product_tree.json'
TREE_CSV = DATASET_PATH + 'product_tree.csv'
CONF_CLASS = DATASET_PATH + 'conf_each_class.csv'
RELIABLE_TRAIN = DATASET_PATH + 'reliable_train.csv'
