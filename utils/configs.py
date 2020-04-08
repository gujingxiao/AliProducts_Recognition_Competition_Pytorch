# Author: Jingxiao Gu
# Description: config Code for AliProducts Recognition Competition

# ['resnet34'  'resnet50'  'resnet101'  'resnet152'  'resnext50_32x4d'  'resnext101_32x8d']
# ['senet154'  'se_resnet50'  'se_resnet101'  'se_resnet152'  'se_resnext50_32x4d'  'se_resnext101_32x4d']
BACKBONE = 'resnet50'
BACKBONE_CHECKPOINT = 'weights/aliproducts_recognition_resnet50_backbone_2.pkl'
MARGIN_CHECKPOINT = 'weights/aliproducts_recognition_resnet50_margin_inner_2.pkl'

DEVICE_ID = [0, 1] # single GPU: [0]; Multi GPU: [0, 1, ...]
MAX_SAMPLES_PER_CLASS = 200  # train data only less than this number
MIN_SAMPLES_PER_CLASS = 100  # train data only more than this number

BATCH_SIZE = 128  # batch size
LEARNING_RATE = 0.001 # Default 0.001 on beginning
LR_STEP = 4  # decay in every LR_STEP
LR_FACTOR = 0.5  # decay factor in every decay

NUM_WORKERS = 4  # number of threads for dataloader
MAX_STEPS_PER_EPOCH = 50000  # max steps per epoch
NUM_EPOCHS = 120  # total epochs

LOG_FREQ = 20  # log every N steps
TIME_LIMIT = 50 * 60 * 60   # time limit for every epoch

MARGIN_TYPE = 'inner'  # Choose margin type ['inner'  'arcMargin']

DATASET_PATH = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/AliProducts/'  # base path for dataset
TRAIN_PATH = DATASET_PATH + 'train/'
VAL_PATH = DATASET_PATH + 'val/'
TRAIN_JSON = DATASET_PATH + 'train.json'
TRAIN_CSV = DATASET_PATH + 'train.csv'
VAL_JSON = DATASET_PATH + 'val.json'
VAL_CSV = DATASET_PATH + 'val.csv'
TREE_FILE = DATASET_PATH + 'product_tree.json'