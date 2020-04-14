# Author: Jingxiao Gu
# Description: config Code for AliProducts Recognition Competition

# Support following models
# ['resnet34'  'resnet50'  'resnet101'  'resnet152'  'resnext50_32x4d'  'resnext101_32x8d']
# ['senet154'  'res2net_v1b_50'  'res2net_v1b_101'  'res2net50_26w_4s'  'res2net101_26w_4s'  'res2next50']
BACKBONE = 'res2net_v1b_50'
MARGIN_TYPE = 'inner'  # Choose margin type ['inner'  'arcMargin'  'tripletMargin']
LOSS_FUNC = 'focalLoss' # Choose loss function ['ceLoss'  'focalLoss']
OPTIMIZER = 'adam_gc'  # Choose optimizer type ['adam'  'adam_gc'  'sgd'  'sgd_gc']
GENERATE_SELECT = False  # From all classes, select part of classes which accuracy is no less than 0.75%
USE_SELECT = False     # Whether to use select classes

BACKBONE_CHECKPOINT = 'weights/aliproducts_recognition_{}_backbone_32845.pkl'.format(BACKBONE)
MARGIN_CHECKPOINT = ''#'weights/aliproducts_recognition_{}_margin_inner_1.pkl'.format(BACKBONE)

DEVICE_ID = [0, 1] # single GPU: [0]; Multi GPU: [0, 1, ...]
# [MAX_SAMPLES_PER_CLASS=100, MIN_SAMPLES_PER_CLASS=80 : 2345 classes]
# [MAX_SAMPLES_PER_CLASS=100000, MIN_SAMPLES_PER_CLASS=0 : 50030 classes]
MAX_SAMPLES_PER_CLASS = 500  # train data only less than this number
MIN_SAMPLES_PER_CLASS = 0  # train data only more than this number
IMAGE_SIZE = 224
DATA_AUGMENTATION = False

BATCH_SIZE = 640  # batch size
LEARNING_RATE = 0.001 # Default 0.001 on beginning
LR_STEP = 2  # decay in every LR_STEP
LR_FACTOR = 0.4  # decay factor in every decay

NUM_WORKERS = 4  # number of threads for dataloader
MAX_STEPS_PER_EPOCH = 500000  # max steps per epoch
NUM_EPOCHS = 10  # total epochs

LOG_FREQ = 50  # log every N steps
SAVE_FREQ = 500  # save weights every N steps
TIME_LIMIT = 50 * 60 * 60   # time limit for every epoch

FREEZE_BACKBONE = True  # Whether freeze backbone
FREEZE_PARTIAL = True  # Whether freeze part of backbone
PARTIAL_NUMBER = 90   # if freeze part of backbone, how many layers will be frozen

DATASET_PATH = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/AliProducts/'  # base path for dataset
TRAIN_PATH = DATASET_PATH + 'train/'
VAL_PATH = DATASET_PATH + 'val/'
TRAIN_JSON = DATASET_PATH + 'train.json'
TRAIN_CSV = DATASET_PATH + 'balance_train.csv'
VAL_JSON = DATASET_PATH + 'val.json'
VAL_CSV = DATASET_PATH + 'val.csv'
TREE_FILE = DATASET_PATH + 'product_tree.json'
SELECT_CLASS = DATASET_PATH + 'conf_each_class.csv'