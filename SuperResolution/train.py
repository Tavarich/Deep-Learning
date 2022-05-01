import os
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from edsr import EDSR
from datasets import SrDataSet
from math import log10

'''
超参数
'''
crop_size = 120
scaling_size = 2

batch_size = 32
num_epoch = 500
lr = 5e-4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter('/root/tf-logs/edsr')  # tensorboard 可视化

model = EDSR()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
save_path = '/root/autodl-tmp/Models/SuperResolution/edsr_v2'


def valid(model, val_iter, epoch):
    model.eval()
    val_loss = 0
    val_psnr = 0
    with torch.no_grad():
        for step, (X, y) in enumerate(val_iter):
            lr = X.to(device)
            hr = y.to(device)
            sr = model(lr)

            loss = criterion(sr, hr)
            psnr = 10 * log10(1 / loss.item())
            val_psnr += psnr
            val_loss += loss.item()

        if step == len(val_iter) - 1:
            writer.add_image(f'EDSR/epoch {epoch} sr',
                             make_grid(sr[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)
            writer.add_image(f'EDSR/epoch {epoch} hr',
                             make_grid(hr[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)

    val_loss /= (step + 1)
    val_psnr /= (step + 1)
    print(f'[epoch{epoch:>2d}] valid loss={val_loss:.6f}, valid pnsr={val_psnr:.6f}Db')
    print("-" * 55)

    return val_psnr, val_loss


def train(model, train_set, valid_set, start=0, maxv=0):
    for epoch in range(start, num_epoch):
        model.train()
        epoch_loss = 0
        epoch_psnr = 0
        for step, (X, y) in enumerate(train_iter):
            lr = X.to(device)
            hr = y.to(device)
            sr = model(lr)

            optimizer.zero_grad()
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()

            psnr = 10 * log10(1 / loss.item())
            epoch_loss += loss.item()
            epoch_psnr += psnr

        epoch_loss /= (step + 1)
        epoch_psnr /= (step + 1)
        print(f'[epoch{epoch:>2d}] train loss={epoch_loss:.6f}, train pnsr={epoch_psnr:.6f}Db')

        val_psnr, val_loss = valid(model, valid_iter, epoch)
        if val_psnr > maxv:
            maxv = val_psnr
            checkpoint = {'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict(),
                          'epoch': epoch,
                          'pnsr': val_psnr}

            torch.save(checkpoint, save_path)
            print(f'model has been saved in epoch{epoch}!')

        writer.add_scalars('EDSR/Loss', {
            'train_loss': epoch_loss,
            'valid_loss': val_loss,
        }, epoch)

        writer.add_scalars('EDSR/PSNR', {
            'train_psnr': epoch_psnr,
            'valid_psnr': val_psnr,
        }, epoch)


if __name__ == "__main__":

    train_path = "/root/autodl-tmp/DIV2K/DIV2K_train_HR"
    valid_path = "/root/autodl-tmp/DIV2K/DIV2K_valid_HR"

    train_set = SrDataSet(train_path, crop_size, scaling_size)
    valid_set = SrDataSet(valid_path, crop_size, scaling_size)

    train_iter = DataLoader(train_set, batch_size, shuffle=True, num_workers=8)
    valid_iter = DataLoader(valid_set, batch_size, shuffle=False, num_workers=8)

    device = torch.device('cuda')
    model.to(device)

    n = input('是否重新开始训练(y/n): ')
    if n == 'y':
        print('加载预训练模型')
        train(model, train_iter, valid_iter)
    else:
        assert os.path.exists(save_path)

        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        start = checkpoint['epoch']
        pnsr = checkpoint['pnsr']

        print('加载已训练模型')
        train(model, train_iter, valid_iter, start=start, maxv=pnsr)
