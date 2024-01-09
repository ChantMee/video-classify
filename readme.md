## 0.Create a virtual environment
conda create -n pytorch python=3.8.10

## 1.Install pytorch
Don’t install a lower version of cuda if you don’t have a GPU
# CUDA 11.1
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# CUDA 10.2
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html
# CPU only
pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html

High version cuda
pip --default-timeout=300 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

## 2. Install third-party libraries
pip install tqdm scikit-learn pandas scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
If the error message indicates that there are fewer libraries, install those libraries.

## 3. run

```
Train: python train.py
Test: python test.py

``` 
