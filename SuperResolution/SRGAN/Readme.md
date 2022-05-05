# SRGAN
SRGAN是一种对抗生成网络。由Generator网络生成超分重构图片，由Discriminator判别图片是否是真实图片。二者对抗训练，目标是最终生成可以骗过Discriminator网络的超分辨率图片

## Generaotr网络
生成网络一般就选用SRResNet。
![image](https://user-images.githubusercontent.com/61419255/166888227-ac8de200-ece0-40d3-b165-c961f1beeb39.png)

参数在图片中已经比较详细的给出（是4倍放大）。

## Discriminator网络
![image](https://user-images.githubusercontent.com/61419255/166888228-ae4af3cb-f8d3-434f-bb7b-dc17ceaed5ad.png)

判别器网络是基于VGG的网络，用于判断输入是真实图片还是生成器生成的超分图片。

## Loss损失函数
不同于其他相对简单的SR网络采用MESLoss，SRGAN的损失函数较为复杂，由若干个部分组成。

**生成网络的损失：**

1. 重构损失，SR图像与HR图像的MES损失
2. 感知损失，由对抗损失和内容损失构成
3. 内容损失，SR图像与HR图像送入截断自VGG网络中第i层第j个卷积核输出的feature map的MES损失
4. 对抗损失，LR图像送入G网后的结果送入D网的负log和

总损失就是上述重构损失和感知损失的加权和

**对抗网络的损失：**

二分类交叉熵损失函数BCELoss

# 文件结构
datasets.py   定义了数据集

models.py     定义了G网和D网

loss.py       定义了各种损失函数

pretrain.py   单独训练G网，即SRResNet网络

train.py      训练SRGAN网络

main.py       对单图进行超分辨率重构

# 注意事项
1. 相关路径仍需要自己修改
2. 支持重新训练和从checkpoint中恢复训练
3. SRGAN网络训练困难，可以先单独预训练一下G网，然后再进行对抗训练
4. 需要GPU环境


# 重构效果
SRResNet训练100个epoch左右，在DIV2k验证集上可以达到31.4db。
SRGAN训练100个epoch左右，在DIV2K验证集上可以达到28db，而且有更真实的细节表现。

双立方插值，SRResNet，SRGAN图片对比如下


![bicubic](https://user-images.githubusercontent.com/61419255/166958345-47bc7bfe-451e-428d-8a4f-8a0b051bcaf6.jpg)    ![srresnet](https://user-images.githubusercontent.com/61419255/166958359-549f7a38-acfa-4818-bdc1-6f2050e47fa8.jpg)    ![srgan](https://user-images.githubusercontent.com/61419255/166958366-c340648e-c905-4d43-b124-85b2d25ad4da.jpg)



