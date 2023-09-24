# バリデーションも含めながら自分で実装してみる

import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torchsummary import summary
from neural_net import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np


###################
# データセットの準備 #
###################
training_data = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=ToTensor()
)

# train:val:test = 8:1:1（大体）にする
num_val = int(len(test_data) * 0.5)
num_test = len(test_data) - num_val
torch.manual_seed(0) # test_dataからvalidationとtestへの分割の再現性の確保
validation_data, test_data = random_split(test_data, [num_val, num_test])

print(f"\nDataset: {training_data.__class__.__name__}")
print(f"    Training data  : {len(training_data)}")
print(f"    Validation data: {len(validation_data)}")
print(f"    Test data      : {len(test_data)}\n")

# データローダーの作成
batch_size = 128 # バッチサイズ
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

img_size = 0
channels = 0
for X, y in train_dataloader:
    print(f"Batch size             : {batch_size}")
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y             : {y.shape}\n")
    img_size = X.shape[2]
    channels = X.shape[1]
    break

classes = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

# 学習データの表示
plt.figure(figsize=(8, 8))
for i in range(9):
    ax = plt.subplot(3, 3, i+1)
    image, label = training_data[i]
    img = image.permute(1, 2, 0)  # 軸の入れ替え (C,H,W) -> (H,W,C)
    plt.imshow(img)
    ax.set_title(classes[label])
    # 枠線消し
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

##############
# モデルの定義 #
##############
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device\n")
# 乱数シードの固定
if device == "cuda":
    torch.cuda.manual_seed(0)
elif device == "mps":
    torch.mps.manual_seed(0)

model = NeuralNetwork(img_size=img_size).to(device)
# モデル構造の確認
if device != "mps":
    summary(model, (channels, img_size, img_size), batch_size=batch_size, device=device)
else:
    print(model)

loss_fn = nn.CrossEntropyLoss()  # 損失関数の定義
optimizer = torch.optim.Adam(model.parameters())  # 最適化アルゴリズムの定義

########################
# モデルの学習用関数の定義 #
########################
def train(dataloader, model, loss_fn, optimizer):
    """
    Method for train
    """
    num_train_data = len(dataloader.dataset)  # 学習データの総数
    num_batches    = len(dataloader)  # バッチの総数
    

def validation(dataloader, model, loss_fn):
    """
    Method for validation
    """

def test(dataloader, model, loss_fn):
    """
    Method for test
    """

#####################
# モデルの学習フェーズ #
#####################