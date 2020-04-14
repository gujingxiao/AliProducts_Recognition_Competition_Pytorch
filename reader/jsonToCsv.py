# Author: Jingxiao Gu
# Description: jsonToCsv Code for AliProducts Recognition Competition

import json
import pandas as pd
from utils.configs import *

# Train Json to CSV
f = open(TRAIN_JSON, encoding='utf-8')
content = f.read()
train = json.loads(content)

class_ids = []
image_ids = []
label_ids = []

class_id = train['images']

for idx in range(len(class_id)):
    class_ids.append(class_id[idx]['class_id'])
    image_ids.append(class_id[idx]['image_id'])
    label_ids.append(class_id[idx]['class_id'])

train_data = {'image_id':image_ids, 'class_id': class_ids, 'label_id':label_ids}
data_df = pd.DataFrame(train_data)
data_df.to_csv(TRAIN_CSV, index=False, header=True)
f.close()

# Val Json to CSV
f = open(VAL_JSON, encoding='utf-8')
content = f.read()
val = json.loads(content)

class_ids = []
image_ids = []
label_ids = []

class_id = val['images']

for idx in range(len(class_id)):
    class_ids.append(class_id[idx]['class_id'])
    image_ids.append(class_id[idx]['image_id'])
    label_ids.append(class_id[idx]['class_id'])

val_data = {'image_id':image_ids, 'class_id': class_ids, 'label_id':label_ids}
data_df = pd.DataFrame(val_data)
data_df.to_csv(VAL_CSV, index=False, header=True)
f.close()

# Balance Data
import numpy as np

train_df = pd.read_csv(TRAIN_CSV)
total_count = 0

class_ids = []
image_ids = []
label_ids = []

for index in range(50030):
    train_select_df = train_df.loc[train_df.class_id.isin([index])].copy()
    train_select_np = np.array(train_select_df)

    # print(train_select_np)
    # np.random.shuffle(train_select_np)
    # print(train_select_np)

    if len(train_select_np) >= 400:
        np.random.shuffle(train_select_np)
        for idx in range(400):
            class_ids.append(train_select_np[idx][0])
            image_ids.append(train_select_np[idx][1])
            label_ids.append(train_select_np[idx][2])
        total_count += 400
    elif len(train_select_np) >= 200:
        np.random.shuffle(train_select_np)
        for idx in range(200):
            class_ids.append(train_select_np[idx][0])
            image_ids.append(train_select_np[idx][1])
            label_ids.append(train_select_np[idx][2])
        total_count += 200
    elif len(train_select_np) >= 100:
        np.random.shuffle(train_select_np)
        for idx in range(100):
            class_ids.append(train_select_np[idx][0])
            image_ids.append(train_select_np[idx][1])
            label_ids.append(train_select_np[idx][2])
        total_count += 100
    elif len(train_select_np) >= 60:
        for idx in range(len(train_select_np)):
            class_ids.append(train_select_np[idx][0])
            image_ids.append(train_select_np[idx][1])
            label_ids.append(train_select_np[idx][2])
        total_count += len(train_select_np)
    elif len(train_select_np) >= 40:
        for idx in range(len(train_select_np)):
            class_ids.append(train_select_np[idx][0])
            image_ids.append(train_select_np[idx][1])
            label_ids.append(train_select_np[idx][2])
        # np.random.shuffle(train_select_np)
        # for idx in range(int(len(train_select_np) * 0.5)):
        #     class_ids.append(train_select_np[idx][0])
        #     image_ids.append(train_select_np[idx][1])
        #     label_ids.append(train_select_np[idx][2])
        total_count += len(train_select_np)#int(len(train_select_np) * 0.5) + len(train_select_np)
    elif len(train_select_np) >= 25:
        factor = 1
        for round_shuffle in range(factor):
            np.random.shuffle(train_select_np)
            for idx in range(len(train_select_np)):
                class_ids.append(train_select_np[idx][0])
                image_ids.append(train_select_np[idx][1])
                label_ids.append(train_select_np[idx][2])
        total_count += len(train_select_np) * factor
    elif len(train_select_np) >= 10:
        factor = 1
        for round_shuffle in range(factor):
            np.random.shuffle(train_select_np)
            for idx in range(len(train_select_np)):
                class_ids.append(train_select_np[idx][0])
                image_ids.append(train_select_np[idx][1])
                label_ids.append(train_select_np[idx][2])
        total_count += len(train_select_np) * factor
    elif len(train_select_np) >= 0:
        factor = 1
        for round_shuffle in range(factor):
            np.random.shuffle(train_select_np)
            for idx in range(len(train_select_np)):
                class_ids.append(train_select_np[idx][0])
                image_ids.append(train_select_np[idx][1])
                label_ids.append(train_select_np[idx][2])
        total_count += len(train_select_np) * factor

    print(index, len(class_ids))

train_data = {'image_id':image_ids, 'class_id': class_ids, 'label_id':label_ids}
data_df = pd.DataFrame(train_data)
data_df.to_csv(DATASET_PATH + 'balance_train.csv', index=False, header=True)

