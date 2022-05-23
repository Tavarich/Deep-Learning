import os
import random
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
from model import UNet, UNetPP
from datasets import SegDataSet

RANDOM_SEED = 42 # any random number
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)       # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)   # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False     # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现




def cal_dice(inputs, target, smooth=1e-5):
    intersect = (inputs * target).sum()
    union = torch.sum(inputs) + torch.sum(target)
    Dice = (2 * intersect + smooth) / (union + smooth)
    return Dice.item()


def valid(net, valid_iter, epoch, mod):
    net.eval()
    valid_loss = 0
    valid_dice = 0

    with torch.no_grad():
        for step, (image, labels) in enumerate(valid_iter):
            image = image.to(device).float()
            labels = labels.to(device).float()
            pred = net(image)
            mask = pred.argmax(dim=1, keepdim=True)
            
            loss = class_cerition(pred, labels.squeeze(1).long()) + dice_cerition(pred, labels.squeeze(1).long())
           
            valid_loss += loss.item()
            valid_dice += cal_dice(mask, labels.long())

            if step == len(valid_iter) - 1:
                writer.add_image(f'{mod}/epoch {epoch} label',
                                 make_grid(image[:4, :3, :, :].cpu(), nrow=4, padding=4), epoch)
                writer.add_image(f'{mod}/epoch {epoch} predict',
                                 make_grid(mask[:4, :1, :, :].cpu(), nrow=4, padding=4), epoch)

        valid_loss /= (step + 1)
        valid_dice /= (step + 1)

        print(f"valid:  loss = {valid_loss:.6f}, dice = {valid_dice:.6f}")
        print("-" * 50)

        return valid_loss, valid_dice


def train(net, train_iter, valid_iter, mod, start=0, maxv=0):
    for epoch in range(start, num_epoch):
        net.train()
        train_loss = 0
        train_dice = 0

        for step, (image, labels) in enumerate(train_iter):
            image = image.to(device).float()
            labels = labels.to(device).float()
            
            pred = net(image)
            mask = pred.argmax(dim=1, keepdim=True).float()
            optimizer.zero_grad()

            loss = class_cerition(pred, labels.squeeze(1).long()) + dice_cerition(pred, labels.squeeze(1).long())
          
            loss.backward()
            optimizer.step()

            train_dice += cal_dice(mask, labels)
            train_loss += loss.item()


        train_loss /= (step + 1)
        train_dice /= (step + 1)
        
        scheduler.step()
        print(f"epoch{epoch:>2d}: loss = {train_loss:.6f}, dice = {train_dice:.6f}")

        valid_loss, valid_dice = valid(net, valid_iter, epoch, mod)

        if valid_dice > maxv:
            maxv = valid_dice
            checkpoint = {'model': net.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict(),
                          'epoch': epoch,
                          'maxv': maxv}

            torch.save(checkpoint, path)
            print(f'model has been saved in epoch{epoch}!')

        writer.add_scalars(f'{mod}/loss', {
            'train_loss': train_loss,
            'valid_loss': valid_loss}, epoch)

        writer.add_scalars(f'{mod}/dice', {
            'train_dice': train_dice,
            'valid_dice': valid_dice}, epoch)


if __name__ == "__main__":
    
    set_seed(RANDOM_SEED)
    
    num_epoch = 150
    batch_size = 9
    lr = 3e-4
    
    x = input("1. unet or 2. unetpp?(1/2)")
    if x == '1':
        path = '/root/autodl-tmp/Models/MedicalSegment/unet.pth'
        model = UNet()
    elif x == '2':
        path = '/root/autodl-tmp/Models/MedicalSegment/unetpp.pth'
        model = UNetPP()

    
    # unet = DeepLab(pretrained=False, num_classes=2)
    train_set = SegDataSet([], '/root/autodl-tmp/Blood/train/img', '/root/autodl-tmp/Blood/train/mask', (512, 512))
    valid_set = SegDataSet([], '/root/autodl-tmp/Blood/valid/img', '/root/autodl-tmp/Blood/valid/mask', (512, 512), mode='eval')

    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_iter = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    class_cerition = nn.CrossEntropyLoss()
    dice_cerition = smp.losses.DiceLoss(mode='multiclass')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    writer = SummaryWriter('/root/tf-logs/Unets')

    device = torch.device('cuda')
    model.to(device)

    n = input('是否重新开始训练?(y/n): ')
    if n == 'y':
        print('加载初始模型')
        if x == '1':
            train(model, train_iter, valid_iter, mod='unet')
        elif x == '2':
            train(model, train_iter, valid_iter, mod='unetpp')
    else:
        assert os.path.exists(path)
        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']
        maxv = checkpoint['maxv']
        print('加载已训练模型')
        if x == '1':
            train(model, train_iter, valid_iter, 'unet', epoch, maxv)
        elif x == '2':
            train(model, train_iter, valid_iter, 'unetpp', epoch, maxv)
        

