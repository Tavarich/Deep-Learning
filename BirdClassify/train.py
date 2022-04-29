import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import BirdDataSet, get_image_list, get_kfold
from model import resnet34_cbam
'''
超参数
'''
num_epoch = 20
batch_size = 48
lr = 1e-4


def valid(model, val_set, val_iter):
    model.eval()
    val_acc = 0
    val_loss = 0
    for step, (X, y) in enumerate(val_iter):
        X = X.to(device)
        y = y.to(device)

        y_hat = model(X)
        loss = criterion(y_hat, y)
        pred = torch.argmax(y_hat, dim=1)

        accuracy = torch.sum(pred == y).item()
        val_acc += accuracy
        val_loss += loss.item()

    val_acc /= len(val_set)
    val_loss /= (step + 1)
    print(f"------valid| acc={val_acc:.6f}, loss={val_loss:.6f}-----")

    return val_acc


def train(models, tk, vk, k):
    """
    :param models: 模型列表
    :param tk: 训练集图片列表
    :param vk: 验证集图片列表
    :param k:  交叉验证折数
    :return: None
    """
    for i in range(5):
        acc = 0.0
        train_set = BirdDataSet(tk[i], train_path, mode='train')
        train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6)

        val_set = BirdDataSet(vk[i], train_path, mode='valid')
        val_iter = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=6)

        for epoch in range(num_epoch):
            models[i].train()
            epoch_acc = 0
            epoch_loss = 0
            for step, (X, y) in enumerate(train_iter):
                X = X.to(device)
                y = y.to(device)

                y_hat = models[i](X)
                loss = criterion(y_hat, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred = torch.argmax(y_hat, dim=1)
                accuracy = torch.sum(pred == y).item()
                epoch_acc += accuracy
                epoch_loss += loss.item()

            epoch_acc /= len(train_set)
            epoch_loss /= (step + 1)
            print(f"|model {i} of epoch{epoch:>2d}| acc={epoch_acc:.6f}, loss={epoch_loss:.6f}")

            val_acc = valid(models[i], val_set, val_iter)
            scheduler.step()

            if val_acc > acc:
                state = {'model': [models[i].state_dict() for i in range(len(models))],
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'epoch': epoch}
                torch.save(state, 'checkpoint')
                acc = val_acc


if __name__ == "__main__":

    train_path = "classify-birds/train_set"
    train_list = get_image_list(train_path)

    tk, vk = get_kfold(train_list, k=5)  # 5折交叉验证

    device = torch.device('cuda')
    models = [resnet34_cbam(True) for i in range(5)]
    for model in models:
        model.to(device)

    optimizer = optim.Adam([{"params": mlp.parameters()} for mlp in models], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)
    criterion = nn.CrossEntropyLoss()

    n = input("是否选择重新训练(y/n) : ")

    if n == 'y':
        train(models, tk, vk, k=5)
    else:
        assert os.path.exists('checkpoint')
        checkpoint = torch.load('checkpoint')

        for i in range(len(models)):
            models[i].load_state_dict(checkpoint['model'][i])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        train(models, tk, vk, k=5)
