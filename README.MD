
# AliProducts Recognition

## System Requirement
1. python 3.5 or 3.6
2. torch 1.1.0
3. torchvision 0.3.0
4. pandas 0.23.4

## Data Process
1. Edit configs.py. Change dataset path, backbone and devide_ids.       
2. Run jsonToCsv.py to generate csv files.      

## Training
1. Make sure configs.py suitable for your configeration.        
2. Run train.py     

## Validation
1. Make sure configs.py suitable for your configeration.        
2. Run val.py   

## Baseline
|Models|Loss Function|Base LR|Batch Size|Class Number|Min Sample Number|Resolution|GAP|
|:---|:---|:---|:---|:---|:---|:---|:---|
|resnet50|CrossEntropyLoss|0.001|128|4566|100|224 x 224|0.8859| 
