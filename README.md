# SOUP
[中文文档](README_cn.md)
SOUP: Singing-Oriented Unreliable Pitcher.
~Automatic Tuning~
## Overview
It's time to abandon the conventional classical music! SOUP can twist the pitch of the input human voice to obtain a deviated f0, which will inspire your creative inspiration and move towards the future of music!
Next, we will show you the effect of this project. In the experiment, we used the [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) project (but we recommend using [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)), and inserted SOUP after the f0 extractor:
[Input Source](source/2018000728.wav)
[Original Output](source/test03292215.wav)
[SOUP Output](source/test03292215_SOUP.wav)
## Usage
Tested under python3.8, it is recommended to use the virtual environment built by conda for operation
### 1. Install the environment
It is recommended to go to https://pytorch.org/ first to install the version of torch you need
```
pip install -r requirements.txt
```
### 2. Download pre-trained models
This project requires downloading two pre-trained models `rmvpe` to `pretrain/rmvpe` and `SOME` to `pretrain/SOME`
rmvpe: [https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip](https://github.com/yxlllc/RMVPE/releases/tag/230917)
SOME: [https://github.com/xunmengshe/OpenUtau/releases/download/0.0.0.0/some-0.0.1.oudep](https://github.com/xunmengshe/OpenUtau/releases/tag/0.0.0.0)
### 3. Preprocessing
Copy the sliced audio file to `data/train/audio`, run `python draw.py`, and then run:
```
python preprocess.py -c config.yaml
```
It must be **sliced audio**, ~because I didn't write the slice inference for SOME~
### 4. Training
Run:
```
python train.py -c config.yaml
```
The training speed is very fast, 100k is almost enough
## Conclusion
This is an April Fool's project, which may not have much practical value, but maybe we will need deviated negative samples in the future?

Base on：

https://github.com/DiffBeautifier/svbcode

https://github.com/openvpi/SOME

https://github.com/yxlllc/RMVPE
