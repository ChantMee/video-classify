## 0.创建虚拟环境
conda create -n pytorch python=3.8.10

## 1.安装pytorch、
没有GPU就不要安装
低版本cuda
# CUDA 11.1
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# CUDA 10.2
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html
# CPU only
pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html

高版本cuda
pip --default-timeout=300 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

## 2.安装第三方库
pip install tqdm scikit-learn pandas scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
报错提升少哪些库就安装哪些库

## 3. run

```
Train: python train.py
Test: python test.py
``` 
