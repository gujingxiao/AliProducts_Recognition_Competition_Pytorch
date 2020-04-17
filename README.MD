
# AliProducts Recognition

## 系統要求 System Requirement		
1. python 3.5 or 3.6
2. CUDA10.1 + CUDNN 7.6.5
3. torch 1.4.0
4. torchvision 0.5.0
5. pandas 0.23.4
6. imgaug 0.4.0
7. NVIDIA APEX (https://github.com/NVIDIA/apex)
8. cupy

## 数据转换 Data Convert
1. 运行jsonTocsv脚本，可以将train.json和val.json转换为对应的csv保存。这里进行转换主要为了处理方便。 		
2. 脚本还会生成一个balance train文件，主要是将数目较多的类别进行一个随机选择，来平衡一下样本。		

## 配置参数 Configuration
1. 所有参数配置在utils下configs脚本中。这里需要配置数据集路径、GPU和训练相关参数。		
2. 当前还有arcface和senet154存在问题，不可用。		

## 训练 Training
1. 完成修改后，使用train脚本进行训练。     
2. 训练时每个epoch会自动保存模型。另外还可以从configs中配置N steps来保存模型，以防止意外宕机。		
3. 每个epoch后都会进行evaluation，评价指标将通过loss，GAP和MAP进行评价。比赛所使用的为MAP。		

## 验证 Validation
1. 使用val脚本进行evaluation，生成指标与train中保持一致。   	
2. 通过configs中GENETATE_SELECT来控制是否需要生成选择列表，用于后续选择稳定类别训练使用。		

## 图像处理 Image Process
1. 由于图像大小比例不一致，为了保持图像原有比例不变，以最长边为准，短边加pad，将图像居中到一个黑色正方形中		
2. 当前训练采用分辨率224x224		
3. 采用imgaug的库进行数，数据显示等采用opencv		

## 训练加速 Training Acceleration
1. 使用NVIDIA APEX库进行混合精度训练   
2. 将所有训练中numpy的处理转换为cupy    

## 基线 Baseline   
* 基线不使用数据增强 No data augmentations		
* 基线训练采用imagenet预训练权重  Use imagenet pretrained weights		
* 基线采用全连接作为分类器  Inner linear classifier		
* 本轮使用train.csv中未经过数据平衡的数据列表		
* 选取类别样本数在[80, 100]范围内的样本，共计2345类，来训练特征提取器		
* 训练共计3轮，第一轮为warm up，第二轮为训练特征提取器		
* 第三轮使用第二轮最优模型并冻结权重进行50030类的训练		

|R|Models|Loss|Opt|Freeze|Epochs|Base LR|LR_STEP|LR_FACTOR|Batch Size|Class Number|Input|GAP|MAP
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|1|res2net_v1b_50|CELoss|Adam|Freeze Backbone|2|0.001|4|0.5|512|2345|224x224|0.3876|0.4562| 
|2|res2net_v1b_50|CELoss|Adam|None|10|0.001|4|0.5|96|2345|224x224|0.9231|0.9298|
|3|res2net_v1b_50|FocalLoss|Adam GC|Freeze Backbone|8|0.001|3|0.4|512|50030|224x224|0.5413|0.6568| 

## 第一版优化 Version#1
* V1初始权重采用基线中第二轮最好的模型  Use imagenet pretrained weights		
* V1依然采用全连接作为分类器  Inner linear classifier		
* V1使用balance_train.csv中经过数据均衡的数据列表		
* 选取类别样本数在[50, 500]范围内的样本，共计14132类，先进行一个epoch的warm up，然后正常训练新的特征提取器		
* 采用新的可靠训练集，使用最新最优的权重finetune特征提取器		
* 最后使用finetune得到的最优模型并冻结权重进行50030类的训练		

|R|Models|Loss|Opt|Freeze|Epochs|Base LR|LR_STEP|LR_FACTOR|Batch Size|Class Number|Input|GAP|MAP
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|1|res2net_v1b_50|CELoss|Adam GC|Freeze Backbone|1|0.001|4|0.5|768|14132|224x224|0.6753|0.7117| 
|2|res2net_v1b_50|CELoss|Adam GC|None|10|0.001|4|0.4|96|14132|224x224|0.7972|0.8586|
|3|res2net_v1b_50|FocalLoss|Adam GC|Freeze Backbone|8|0.001|3|0.4|512|50030|224x224|0.6730|0.7561| 

## 第二版优化 Version#2
* V2初始权重采用V1中第二轮最好的模型 		
* V2依然采用全连接作为分类器  Inner linear classifier	
* 根据规则，不能使用val作为数据清洗的依据，那么这里单纯使用train数据来进行自动清洗    
* 使用第一版最优的模型，将balance_train.csv中的数据进行inference，使用csv列表保存image_id, confidence, predict和label。
* 统计上述csv中每个类别的信息，如果某一类准确率≥0.9，那么这个类别所有数据保留；如果准确率≥0.6，那么这个类别内分类错误的数据将被删除掉；如果准确率＜0.6，那么不仅删除分类错误的数据，分类正确但置信度低于0.6的也将被删除（如果某一个类别所有数据都将被删除，这时需要做一个保留：如果有正确的，保留分数最高的，如果没有正确的，随机保留一个）
* 重新生成新的balance_train.csv，并且将数目小于等于5的类别，都扩充到5个
* 先进行一个epoch的warm up，后使用v1中的backbone，并freeze backbone，训练分类器，查看效果		
* 先进行一个epoch的warm up，后freeze 一部分layers，然后进行训练，查看效果  		

|R|Models|Loss|Opt|Freeze|Epochs|Base LR|LR_STEP|LR_FACTOR|Batch Size|Class Number|Input|GAP|MAP
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|1|res2net_v1b_50|FocalLoss|Adam GC|Freeze Backbone|1|0.001|4|0.5|512|32845|224x224|0.6528|0.7144| 
|2|res2net_v1b_50|FocalLoss|Adam GC|Freeze 90 layers|10|0.001|4|0.4|192|32845|224x224|0.8761|0.9174|
|3|res2net_v1b_50|FocalLoss|Adam GC|Freeze Backbone|8|0.001|3|0.4|512|50030|224x224|-|-| 

## 多层级标签训练 Hierarchical Label Training
* 在上述基础上，还需要使用数据集的多层级标签训练   
* 标签共有4级，第4级就是50030类全子标签；第3级有100多个类别，可以从网络再拉一个branch进行第三级标签训练，并且通过加权因子将第三次标签的loss与第四级标签的loss进行求和，再一起进行反向传播    
* 这里需要设计一个比较新颖的网络结构   
* 这部分可能会成为一个比较大的突破    

## 考虑使用决策树做高层级标签分类 Consider Decision Tree
* 可以考虑使用extractor出来的特征向量，结合xgboost进行训练，可查资料较少，目前不知道效果如何   

## 引用 Reference
1. res2net: https://github.com/Res2Net/Res2Net-PretrainedModels [Res2Net: A New Multi-scale Backbone Architecture]
2. Gradient-Centralization: https://github.com/Yonghongwei/Gradient-Centralization [Gradient Centralization: A New Optimization Technique for Deep Neural Networks]
