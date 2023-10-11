import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Never, Literal
import time
from datetime import datetime
import os
import platform
import pandas as pd
import csv

from neural_net import NeuralNetwork


def set_device() -> str:
    """
    使用するデバイスを指定する
    """
    # デバイスの指定
    device = (
        "cuda" 
        if torch.cuda.is_available() else "mps"
        if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} device\n")
    return device


def fix_seed(device: str, seed: int = 0) -> None:
    """
    各種乱数シードの固定
    """
    torch.manual_seed(seed)

    if device == "cuda":
        torch.cuda.manual_seed(seed)
    elif device == "mps":
        torch.mps.manual_seed(seed)


def confirm_dataset(train_data: datasets, val_data: datasets, test_data: datasets) -> None:
    """
    学習データ、検証データ、テストデータのサイズ確認
    """
    print(f"\nDataset: {train_data.__class__.__name__}")
    print(f"    Training data  : {len(train_data)}")
    print(f"    Validation data: {len(val_data)}")
    print(f"    Test data      : {len(test_data)}\n")


def get_img_info(dataloader: DataLoader) -> Tuple[int, int] | Never:
    """
    データローダーの形状を表示する。返り値としてチャンネル数と画像サイズを返す。
    """
    class NotSquareImgError(Exception):
        def __str__(self):
            return f"{NotSquareImgError.__name__}: Image height and width don't match!"

    for X, y in dataloader:
        print("Shape of X")
        print(f"    Batch size: {X.shape[0]}")
        print(f"    Channels  : {X.shape[1]}")
        print(f"    Height    : {X.shape[2]}")
        print(f"    Width     : {X.shape[3]}")
        print(f"Shape of y : {y.shape} {y.dtype}\n")
        channels = X.shape[1]
        img_size = X.shape[2]
        break

    try:
        if X.shape[2] != X.shape[3]:
            raise NotSquareImgError
    except NotSquareImgError as e:
        print(e)

    return channels, img_size


def show_dataset_sample(data: datasets, classes: dict, show_fig: bool = True) -> None:
    """
    学習データの表示。9個のサンプルを表示する。
    """
    if not(show_fig):
        return
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
    plt.show()


def confirm_scheduler(scheduler, show_fig: bool = True) -> None:
    """
    スケジューラによる学習率の変化の確認用
    """
    if not(show_fig):
        return
    lrs = []
    for i in range(scheduler.t_initial):
        if i == 0:
            print(f"warmup_lr_init: {scheduler._get_lr(i)[0]}")
        lrs.append(scheduler._get_lr(i))
        if i == 30:
            print(f"finish warmup : {scheduler._get_lr(i)[0]}")
        elif i == scheduler.t_initial - 1:
            print(f"final lr      : {scheduler._get_lr(i)[0]}\n")
    plt.plot(lrs)
    plt.show()


def train(model, criterion, optimizer, dataloader: DataLoader, device: str) -> Tuple[float, float]:
    """
    学習用関数。1エポック間の学習について記述する。
    """
    model.train()  # 学習モードに移行

    num_train_data = len(dataloader.dataset)  # 学習データの総数
    iterations     = dataloader.__len__()  # イテレーション数

    total_correct = 0 # エポックにおける、各イテレーションの正解数の合計
    total_loss    = 0 # エポックにおける、各イテレーションの損失の合計

    for iteration, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        batch_size = len(X)  # バッチサイズ

        pred = model(X)
        loss = criterion(pred, y)

        # 現在のイテレーションにおける、バッチ内の総正解数（最大：バッチサイズ）
        num_correct = (pred.argmax(1) == y).type(torch.float).sum().item()

        total_correct += num_correct
        total_loss    += loss.item()

        optimizer.zero_grad()  # モデルの全パラメータの勾配を初期化
        loss.backward()  # 誤差逆伝搬
        optimizer.step()  # パラメータの更新

        # 学習状況の表示
        if (iteration % batch_size == 0) or (iteration+1 == iterations):
            acc_in_this_iteration = num_correct / batch_size
            print(f"    [{iteration+1:3d}/{iterations:3d} iterations] Loss: {loss.item():>5.4f} - Accuracy: {acc_in_this_iteration:>5.4f}")
    

    avg_acc  = total_correct / num_train_data  # 本エポックにおけるAccuracy
    avg_loss = total_loss / iterations  # 本エポックにおける損失
    return avg_acc, avg_loss 


def validation(model, criterion, dataloader: DataLoader, device: str) -> Tuple[float, float]:
    """
    検証用関数。`train()`後に配置する
    """
    model.eval()  # 検証モードに移行

    num_val_data = len(dataloader.dataset)  # 検証データの総数
    iterations   = dataloader.__len__()  # イテレーション数

    total_correct = 0 # エポックにおける、各イテレーションの正解数の合計
    total_loss    = 0 # エポックにおける、各イテレーションの損失の合計

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = criterion(pred, y)

            # 現在のイテレーションにおける、バッチ内の総正解数（最大：バッチサイズ）
            num_correct = (pred.argmax(1) == y).type(torch.float).sum().item()

            total_correct += num_correct
            total_loss    += loss.item()
    
    avg_acc  = total_correct / num_val_data  # 本エポックにおけるAccuracy
    avg_loss = total_loss / iterations  # 本エポックにおける損失
    return avg_acc, avg_loss

def test(model, criterion, dataloader: DataLoader, device: str) -> Tuple[float, float]:
    """
    テスト用関数。全エポック終了後に配置する。`validation()`を流用
    """
    avg_acc, avg_loss = validation(model, criterion, dataloader, device)
    return avg_acc, avg_loss


def make_dir_4_deliverables() -> str:
    """
    成果物保存用のディレクトリを作成する。ディレクトリ名は日付+時間。\n
    返り値として、作成したディレクトリパスを返す。
    """
    if not(platform.system() == "Windows"):
        date_now = datetime.now().isoformat(timespec="seconds")
    else:
        date_now = datetime.now().strftime("%Y%m%dT%H-%M-%S")
    
    os.makedirs(f"results/{date_now}/")
    return f"./results/{date_now}"


class ListsOfDifferntLengths(Exception):
    def __str__(self):
        return f"{ListsOfDifferntLengths.__name__}: The two lists are different lengths!"


class SaveLearningProgress:
    def __init__(self, dir_path: str):
        self.dir_path = dir_path

    def save_lc(self, train_list: list, val_list: list, lc_type: Literal["Accuracy", "Loss"]):
        try:
            if len(train_list) != len(val_list):
                raise ListsOfDifferntLengths
        except ListsOfDifferntLengths as e:
            print(e)

        epochs = len(train_list)

        fig, ax = plt.subplots(figsize=(16, 9), dpi=120)

        ax.set_xlabel("Epochs")
        ax.set_ylabel(f"{lc_type}")
        ax.set_title(f"Learning Curve ({lc_type})")

        ax.plot(train_list, label=f"Train {lc_type}")
        ax.plot(val_list, label=f"Validation {lc_type}")

        step = 5 if epochs < 100 else 10

        if (epochs > 1) and abs((train_list[0] - train_list[epochs-1]) > 0.1):
            ax.set_yscale("log")

        ax.set_xticks(np.concatenate([np.array([0]), np.arange(step-1, epochs, step)]))
        ax.set_xticklabels(np.concatenate([np.array([1]), np.arange(step, epochs+1, step)], dtype="unicode"))

        ax.grid()
        ax.legend()
        fig.tight_layout()

        plt.savefig(f"{self.dir_path}/LC_{lc_type}_{self.dir_path.replace('./results/', '')}.png")
        plt.close()

    def save_csv(self, list_1: list, list_2: list, data_type: Literal["learn_acc", "learn_loss", "test"]):
        try:
            if len(list_1) != len(list_2):
                raise ListsOfDifferntLengths
        except ListsOfDifferntLengths as e:
            print(e)
        
        index = ["epoch_" + str(i + 1) for i in range(len(list_1))]

        match data_type:
            case "learn_acc":
                df = pd.DataFrame(
                    {
                        "train_acc": list_1,
                        "val_acc": list_2
                    },
                    index=index
                )
                df.to_csv(
                    f"{dir_path}/acc_in_learn_{dir_path.replace('./results/', '')}.csv",
                    index=False, 
                    encoding="utf-8", 
                    quoting=csv.QUOTE_ALL
                )
            case "learn_loss":
                df = pd.DataFrame(
                    {
                        "train_loss": list_1,
                        "val_loss": list_2
                    },
                    index=index
                )
                df.to_csv(
                    f"{dir_path}/loss_in_learn_{dir_path.replace('./results/', '')}.csv",
                    index=False, 
                    encoding="utf-8", 
                    quoting=csv.QUOTE_ALL
                )
            case "test":
                index = ["test"]
                df = pd.DataFrame(
                    {
                        "test_acc": list_1,
                        "test_loss": list_2
                    },
                    index=index
                )
                df.to_csv(
                    f"{dir_path}/stats_in_test_{dir_path.replace('./results/', '')}.csv",
                    index=False, 
                    encoding="utf-8", 
                    quoting=csv.QUOTE_ALL
                )
            case _:
                try:
                    raise ValueError("SaveLearningProgress.save_csv(): Unknown data type!")
                except ValueError as e:
                    print(e)


class EarlyStopping:
    def __init__(self, dir_path: str, model_name: str, patience: int = 5):
        """
        `dir_path` (str): モデルの保存先ディレクトリ
        `patience` (str): 訓練が停止するまでの残りエポック数
        """
        self.patience = patience  # 訓練が停止するまでの残りエポック数
        self.counter = 0  # エポック数の現在のカウンター値
        self.best_score = None  # 損失のベストスコア
        self.early_stop = False  # Early stopping のフラグ
        self.path = f"{dir_path}/{dir_path.replace('./results/', '')}_{model_name}.pth"  # モデル保存パス

    def __call__(self, model, val_loss: float):
        """
        最小の損失が更新されたかの計算
        """
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_model(model)
        elif score > self.best_score:
            # ベストスコアを更新できなかった場合
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # ベストスコアを更新した場合
            self.best_score = score  # ベストスコアの更新
            self.save_model(model)
            self.counter = 0

    def save_model(self, model):
        torch.save(model.state_dict(), self.path)


if __name__ == "__main__":
    #-------------------------#
    #          諸準備          #
    #-------------------------#
    device = set_device()  # デバイスの指定
    fix_seed(device=device)  # 乱数シードの固定

    # 学習結果保存用のディレクトリを作成
    dir_path = make_dir_4_deliverables()  # ./results/[something]

    #------------------------------#
    #          モデルの定義          #
    #------------------------------#
    model = NeuralNetwork(img_size=28, channels=1).to(device)
    model_name = NeuralNetwork.__name__

    epochs = 1000 
    batch_size = 64

    learning_rate = 0.01
    weight_decay = 0

    label_smoothing_epsilon = 0.

    # モデル構造の確認
    summary(model=model, input_size=(batch_size, 1, 28, 28))
    print()

    optimizer = torch.optim.SGD(
        params=model.parameters(), 
        lr=learning_rate, 
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing_epsilon)  # Label Smoothingありの損失関数


    #----------------------------------#
    #          データセットの用意         #
    #----------------------------------#
    train_data = datasets.FashionMNIST(root="./data", train=True, download=True, transform=ToTensor())
    test_data  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=ToTensor())
    # データセットの分割
    # train:val:test = 8:1:1
    num_val  = int(len(test_data) * 0.5)
    num_test = len(test_data) - num_val
    val_data, test_data = random_split(test_data, [num_val, num_test])
    
    # データローダーの作成
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    confirm_dataset(train_data, val_data, test_data)
    channels, img_size = get_img_info(train_dataloader)

    # CIFAR 10の全クラス
    classes = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }
    show_dataset_sample(train_data, classes, show_fig=False)


    #######################################
    #          ファインチューニング          #
    #######################################
    # 以下のリストの要素数はエポック数となる
    train_acc_list  = []
    train_loss_list = []
    val_acc_list  = []
    val_loss_list = []

    earlystopping = EarlyStopping(dir_path, model_name, patience=3)
    save_progress = SaveLearningProgress(dir_path=dir_path)

    print("\033[44mTraining Step\033[0m")

    for t in range(epochs):
        time_start = time.perf_counter()  # エポック内の処理時間の計測

        print(f"Epoch {t+1}\n----------------------------------------------------------------", flush=True)

        print("\033[34mTrain\033[0m", flush=True)
        train_acc, train_loss = train(model, criterion, optimizer, train_dataloader, device)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        print("\033[34mValidation\033[0m", flush=True)
        val_acc, val_loss = validation(model, criterion, val_dataloader, device)
        print(f"    Avg val loss: {val_loss:>5.4f}, Avg val acc: {val_acc:>5.4f}")
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        time_end = time.perf_counter()
        elapsed_per_epoch = time_end - time_start

        print(f"\033[34mStats of Train in Epoch {t+1}\033[0m\n    Avg loss: {train_loss:>5.4f}, Avg acc: {train_acc:>5.4f} (Duration: {elapsed_per_epoch:.2f}s)\n", flush=True)

        # AccuracyとLossの保存
        save_progress.save_csv(train_acc_list, val_acc_list, data_type="learn_acc")
        save_progress.save_csv(train_loss_list, val_loss_list, data_type="learn_loss")
        save_progress.save_lc(train_acc_list, val_acc_list, lc_type="Accuracy")
        save_progress.save_lc(train_loss_list, val_loss_list, lc_type="Loss")

        # Early Stoppingの判定
        earlystopping(model, val_loss)
        if earlystopping.early_stop:
            print(f"\nEarly Stopping (Saved model path: {earlystopping.path})")
            break

    print("\033[44mTest Step\033[0m", flush=True)
    test_acc, test_loss = test(model, criterion, test_dataloader, device)
    print(f"    Avg test loss: {test_loss:>5.4f}, Avg test acc: {test_acc:>5.4f}", flush=True)
    save_progress.save_csv([test_acc], [test_loss], data_type="test")
