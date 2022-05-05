# EDSR
![7d97d659ac7303d1785949db09a5b3f9](https://user-images.githubusercontent.com/61419255/166881235-317f9e3e-a5ee-436a-ac83-0bfe571e641a.png)

EDSR的结构由开头一个卷积层，若干个残差层，上采样层，结尾一个卷积层构成。

残差层由卷积层，ReLU激活函数，卷积层构成，并形成shortcut跳线连接。
上采样层由卷积层，PixelShuffle层构成，并根据超分倍率堆叠不同数量的上采样层。

EDSR是对SRResNet的优化改进，可以看到残差层中去掉了BatchNorm层，可以降低内存消耗从而堆叠更多的残差层。

# 文件结构
1. dataset.py 定义了数据集
2. edsr.py    定义了EDSR网络模型
3. train.py   网络训练
4. main.py    对单张图片进行超分

# 注意事项
1. 请将各文件中的路径更换（训练集、验证集、checkpoint、tensorboard log 等等）
2. 模型训练支持重新训练和从checkpoint中恢复训练
3. 图片超分需要在main.py中自行更改图片路径
4. 需要gpu环境

# 效果
训练100个epoch后，在div2k验证集上，峰值信噪比可以达到31db。
