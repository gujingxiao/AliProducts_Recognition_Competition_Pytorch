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
# train_df = pd.read_csv(TRAIN_CSV)
# counts = train_df.class_id.value_counts()
# selected_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index
# print(counts)

