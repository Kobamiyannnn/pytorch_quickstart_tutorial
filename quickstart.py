import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time
from neural_net import NeuralNetwork
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

#####################
# Working with data #
#####################
print("< Working with data >")

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 512  # ミニバッチ学習のバッチサイズ

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")  # N: Batch size
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# FashionMNISTの全クラス
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# 学習データのの表示
plt.figure()
for i in range(10):
    ax = plt.subplot(2, 5, i+1)
    image, label = training_data[i]
    img = image.permute(1, 2, 0)  # 軸の入れ替え (C,H,W) -> (H,W,C)
    plt.imshow(img)
    ax.set_title(classes[label])
    # 枠線消し
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

###################
# Creating Models #
###################
print("\n" + "-" * 60 + "\n" + "< Creating Models >")

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device\n")

# Define model
model = NeuralNetwork().to(device)
if device != "mps":
    summary(model, (1, 28, 28), batch_size=batch_size, device=device)
else:
    print(model)

###################################
# Optimizing the Model Parameters #
###################################
print("\n" + "-" * 60 + "\n" + "< Optimizing the Model Parameters >")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_correct = 0
    total_loss = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        total_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            # loss.item(): batch番目のミニバッチにおける損失
            # len(X): バッチサイズと等しい
            loss, current = loss.item(), (batch + 1) * len(X)
            accuracy = correct / len(X)
            print(f"loss: {loss:>5.4f} - accuracy: {accuracy:>5.4f} [{current:>5d}/{size:>5d}]")
    total_correct /= size
    total_loss /= num_batches
    print(f"Average: \n Accuracy: {(100 * total_correct):>0.1f}%,     Loss: {total_loss:>8f}")
    return total_loss, total_correct

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct

# 学習を行う
elapsed_per_epoch = 0  # 1エポックを終えるまでの経過時間
epochs = 200

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

avg_train_loss = 0
avg_train_acc = 0
avg_test_loss = 0
avg_test_acc = 0

for t in range(epochs):
    time_start = time.perf_counter()
    print(f"Epoch {t+1} (Former epoch: {elapsed_per_epoch:.2f}s)\n-------------------------------")
    
    avg_train_loss, avg_train_acc = train(train_dataloader, model, loss_fn, optimizer)
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)

    avg_test_loss, avg_test_acc = test(test_dataloader, model, loss_fn)
    test_loss_list.append(avg_test_loss)
    test_acc_list.append(avg_test_acc)

    time_end = time.perf_counter()
    elapsed_per_epoch = time_end - time_start

print("Done!")
date_now = datetime.now().isoformat(timespec="seconds")
os.makedirs(f"results/{date_now}")

# 学習曲線 (loss)
fig, ax = plt.subplots(figsize=(16,9), dpi=120)
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title("Learning Curve (Loss)")
ax.plot(np.arange(0, epochs), train_loss_list, label="train_loss")
ax.plot(np.arange(0, epochs), test_loss_list, label="test_loss")
ax.set_xticks(np.arange(0, epochs, 5))
ax.legend()
fig.tight_layout()
plt.savefig(f"./results/{date_now}/LC_loss_{date_now}.png")

# 学習曲線 (accuracy)
fig, ax = plt.subplots(figsize=(16,9), dpi=120)
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
ax.set_title("Learning Curve (Accuracy)")
ax.plot(np.arange(0, epochs), train_acc_list, label="train_acc")
ax.plot(np.arange(0, epochs), test_acc_list, label="test_acc")
ax.set_xticks(np.arange(0, epochs, 5))
ax.legend()
fig.tight_layout()
plt.savefig(f"./results/{date_now}/LC_acc_{date_now}.png")


#################
# Saving Models #
#################
print("\n" + "-" * 60 + "\n" + "< Saving Models >")

# torch.save(model.state_dict(), "model.pth")
# torch.save(model.state_dict(), "model.pth")  # model.state_dict()でパラメータのみ保存
model_scripted = torch.jit.script(model)
model_scripted.save(f"./results/{date_now}/model_{date_now}.pth")
print("Saved PyTorch Model State to model.pth")