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

import time
from datetime import datetime
import os
import platform


###################
# データセットの準備 #
###################
training_data = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
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
    print(f"Shape of y             : {y.shape} {y.dtype}\n")
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


def show_dataset_sample(data: datasets, classes: dict, show_fig: bool = True) -> None:
    """
    学習データの表示。3✕3個のサンプルを表示する
    """
    plt.figure(figsize=(8, 8))
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        image, label = data[i]
        img = image.permute(1, 2, 0)  # 軸の入れ替え (C,H,W) -> (H,W,C)
        plt.imshow(img)
        ax.set_title(classes[label])
        # 枠線消し
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if show_fig:
        plt.show()


show_dataset_sample(training_data, classes, show_fig=False)

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

model = NeuralNetwork(img_size=img_size, channels=channels).to(device)
# モデル構造の確認
if device != "mps":
    summary(model, (channels, img_size, img_size), batch_size=batch_size, device=device)
    print()
else:
    print(model)

loss_fn = nn.CrossEntropyLoss()  # 損失関数の定義
# optimizer = torch.optim.Adam(model.parameters())  # 最適化アルゴリズムの定義
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

########################
# モデルの学習用関数の定義 #
########################
def train(dataloader, model, loss_fn, optimizer):  
    """
    学習用関数。1エポックだけのサイクルとなる
    """
    model.train()  # 学習モードに移行

    num_train_data = len(dataloader.dataset)  # 学習データの総数
    num_batches    = len(dataloader)  # バッチの総数

    total_correct = 0  # 現エポックにおける正解数
    total_loss    = 0  # 現エポックにおける全バッチのLossの合計値
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        num_correct = (pred.argmax(1) == y).type(torch.float).sum().item()  # 現バッチにおける総正解数

        total_correct += num_correct
        total_loss    += loss.item()
        
        # 誤差逆伝播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch % len(X) == 0) or (batch+1 == num_batches):
            current_acc = num_correct / len(X)
            print(f"    [{batch+1:>3d}/{num_batches:3d} batches] Loss: {loss.item():>5.4f} - Accuracy: {current_acc:>5.4f}")
    
    avg_acc  = total_correct / num_train_data
    avg_loss = total_loss / num_batches
    return avg_acc, avg_loss

def validation(dataloader, model, loss_fn):
    """
    検証用関数。`train()`後に配置
    """
    model.eval()  # 検証用モード

    num_val_data = len(dataloader.dataset)  # 検証データの総数
    num_batches  = len(dataloader)  # バッチの総数

    total_correct = 0
    total_loss    = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            num_correct = (pred.argmax(1) == y).type(torch.float).sum().item()

            total_correct += num_correct
            total_loss += loss.item()
    
    avg_acc = total_correct / num_val_data
    avg_loss = total_loss / num_batches
    return avg_acc, avg_loss

def test(dataloader, model, loss_fn):
    """
    テスト用関数。学習終了後に配置。`validation()`を流用
    """
    avg_acc, avg_loss = validation(dataloader, model, loss_fn)
    return avg_acc, avg_loss

#####################
# モデルの学習フェーズ #
#####################
epochs = 100

train_acc_list  = []
train_loss_list = []

val_acc_list  = []
val_loss_list = []

print("\033[44mTraining Step\033[0m")
for t in range(epochs):
    time_start = time.perf_counter()

    print(f"Epoch {t+1}\n----------------------------------------------------------------")
    
    print("\033[34mTrain\033[0m")
    avg_train_acc, avg_train_loss = train(train_dataloader, model, loss_fn, optimizer)
    train_acc_list.append(avg_train_acc)
    train_loss_list.append(avg_train_loss)

    print("\033[34mValidation\033[0m")
    avg_val_acc, avg_val_loss = validation(validation_dataloader, model, loss_fn)
    print(f"    Avg validation loss: {avg_val_loss:>5.4f}, Avg validation accuracy: {avg_val_acc:>5.4f}")
    val_acc_list.append(avg_val_acc)
    val_loss_list.append(avg_val_loss)

    time_end = time.perf_counter()
    elapsed_per_epoch = time_end - time_start
    print(f"\033[34mStats of Train in Epoch {t+1}\033[0m\n    Avg loss: {avg_train_loss:>5.4f}, Avg accuracy: {avg_train_acc:>5.4f} (Duration: {elapsed_per_epoch:.2f}s)\n")

print("\033[44mTest Step\033[0m")
avg_test_acc, avg_test_loss = test(test_dataloader, model, loss_fn)
print(f"    Avg test loss: {avg_test_loss:>5.4f}, Avg test accuracy: {avg_test_acc:>5.4f}")

# 学習結果保存用のディレクトリ作成
if not(platform.system() == "Windows"):
    date_now = datetime.now().isoformat(timespec='seconds')
else:
    date_now = datetime.now().strftime("%Y%m%dT%H-%M-%S")

os.makedirs(f"results/{date_now}")

def save_learning_curve_of_loss(train_loss_list: list, val_loss_list: list, date_now: str) -> None:
    """
    損失の学習曲線を表示。学習時と検証時とを同時に表示する
    """
    epochs = len(train_loss_list)
    
    fig, ax = plt.subplots(figsize=(16,9), dpi=120)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Learning Curve (Loss)")

    ax.plot(train_loss_list, label="Train loss")
    ax.plot(val_loss_list, label="Val loss")

    ax.set_xticks(np.concatenate([np.array([0]), np.arange(4, epochs, 5)]))
    ax.set_xticklabels(np.concatenate([np.array([1]), np.arange(5, epochs+1, 5)], dtype="unicode"))

    ax.set_ylim(0)

    ax.grid()
    ax.legend()
    fig.tight_layout()

    plt.savefig(f"./results/{date_now}/LC_loss_{date_now}.png")

def save_learning_curve_of_acc(train_acc_list: list, val_acc_list: list, date_now: str) -> None:
    """
    Accuracyの学習曲線を表示。学習時と検証時とを同時に表示する
    """
    epochs = len(train_acc_list)

    fig, ax = plt.subplots(figsize=(16,9), dpi=120)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve (Accuracy)")

    ax.plot(train_acc_list, label="Train acc")
    ax.plot(val_acc_list, label="Val acc")

    ax.set_xticks(np.concatenate([np.array([0]), np.arange(4, epochs, 5)]))
    ax.set_xticklabels(np.concatenate([np.array([1]), np.arange(5, epochs+1, 5)], dtype="unicode"))

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    # y軸のみにminor ticksを表示する
    ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    ax.xaxis.set_tick_params(which="minor", bottom=False)

    ax.grid()
    ax.legend()
    fig.tight_layout()

    plt.savefig(f"./results/{date_now}/LC_acc_{date_now}.png")

save_learning_curve_of_acc(train_acc_list, val_acc_list, date_now)
save_learning_curve_of_loss(train_loss_list, val_loss_list, date_now)
