# SOUP

SOUP: Singing-Oriented Unreliable Pitcher. 

~自动跑调~

## 概述

是时候放弃循规蹈矩的古典音乐了！SOUP可以将输入人声的音调扭曲，以取得偏离的f0，这将激发你的创作灵感，向着未来音乐进发！

接下来将为你展示本项目的效果，在实验中我们使用了[Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC)（但我们更建议使用[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)）项目，在f0提取器后插入了SOUP：

[输入源](source/2018000728.wav)

[原始输出](source/test03292215.wav)

[SOUP输出](source/test03292215_SOUP.wav)

## 使用

在python3.8下进行过测试，建议使用conda构建的虚拟环境下进行炒作

### 1.安装环境

建议先行前往https://pytorch.org/ 安装你所需要版本的torch

```
pip install -r requirements.txt
```

### 2.下载预训练模型

本项目需要下载两个与训练模型`rmvpe`到`pretrain/rmvpe`与`SOME`到`pretrain/SOME`

rmvpe：[https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip](https://github.com/yxlllc/RMVPE/releases/tag/230917)

SOME：[https://github.com/xunmengshe/OpenUtau/releases/download/0.0.0.0/some-0.0.1.oudep](https://github.com/xunmengshe/OpenUtau/releases/tag/0.0.0.0)

### 3.预处理

将**切片好的**音频文件复制到`data/train/audio`中，运行`python draw.py`，之后运行：

```
python preprocess.py -c config.yaml
```

一定要是**切片好的**音频，~因为我没写SOME的切片推理~

### 4.训练

运行:

```
python train.py -c config.yaml
```

训练速度非常快，100k就差不多了

## 结语

这是一个愚人节项目，可能没什么实际的使用价值，或许在未来我们需要跑调的负样本会用得到？

基于项目：

https://github.com/DiffBeautifier/svbcode

https://github.com/openvpi/SOME

https://github.com/yxlllc/RMVPE
