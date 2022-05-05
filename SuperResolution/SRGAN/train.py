import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from datasets import SrDataSet
from models import Generator, Discriminator
from loss import PerceptualLoss

'''超分参数以及超参数'''
crop_size = 96
scaling_size = 2

num_epoch = 150
batch_size = 16
lr = 1e-4

device = torch.device('cuda')
criterionG = PerceptualLoss(device)
criterionD = nn.BCELoss()

netG = Generator(scaling_size)
netD = Discriminator(crop_size)

optimizerG = optim.Adam(netG.parameters(), lr=lr)
optimizerD = optim.Adam(netD.parameters(), lr=lr)

schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=10, gamma=0.9)
schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=10, gamma=0.9)

netG.to(device)
netD.to(device)

checkpointGan = '/root/autodl-tmp/Models/SuperResolution/srgan'

writer = SummaryWriter('/root/tf-logs/srgan')

# # G网加载已经训练过的参数
# pretrained = '/root/autodl-tmp/Models/SuperResolution/srresnet'
# checkpointG = torch.load(pretrained)
# netG.load_state_dict(checkpointG['model'])


def calculate_psnr(sr_img, hr_img):
    return 10. * torch.log10(1. / torch.mean((hr_img - sr_img) ** 2))


def train(netG, netD, train_iter, valid_iter, start=0, maxv=0):

    real_label = torch.ones([batch_size, 1, 1, 1]).to(device)
    fake_label = torch.zeros([batch_size, 1, 1, 1]).to(device)

    for epoch in range(start, num_epoch):
        netG.train()
        netD.train()
        train_psnr = 0
        train_loss_g = 0
        train_loss_d = 0

        for step, (lr_img, hr_img) in enumerate(train_iter):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            '''训练Generator网络'''
            fake_img = netG(lr_img)
            loss_g = criterionG(fake_img, hr_img, netD(fake_img))
            optimizerG.zero_grad()
            loss_g.backward()
            optimizerG.step()

            train_loss_g += loss_g.item()
            train_psnr += calculate_psnr(fake_img, hr_img).item()
            
            if epoch > 50 and epoch % 3 == 0:   
                '''训练两次Generator网络，然后训练一次Discriminator网络'''
                real_out = netD(hr_img)
                fake_out = netD(fake_img.detach())
                loss_d = criterionD(real_out, real_label) + criterionD(fake_out, fake_label)
                optimizerD.zero_grad()
                loss_d.backward()
                optimizerD.step()

                train_loss_d += loss_d.item()

        train_loss_d /= (step + 1)
        train_loss_g /= (step + 1)
        train_psnr /= (step + 1)

        schedulerD.step()
        schedulerG.step()

        print(f"train [epoch{epoch:>2d}]: psnr = {train_psnr:.6f}, G Loss = {train_loss_g:.6f}, D Loss = {train_loss_d:.6f}")

        valid_loss, valid_psnr = valid(netG, netD, valid_iter, epoch)
        if valid_psnr > maxv or epoch % 50 == 0:
            maxv = valid_psnr
            checkpoint = {
                'modelG': netG.state_dict(),
                'modelD': netD.state_dict(),
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict(),
                'schedulerG': schedulerG.state_dict(),
                'schedulerD': schedulerD.state_dict(),
                'epoch' : epoch,
                'psnr': valid_psnr
            }
            torch.save(checkpoint, checkpointGan)
            print(f'model has been save in epoch{epoch}')
        writer.add_scalars('srgan/g_loss', {
            'train_loss': train_loss_g,
            'valid_loss': valid_loss,
        }, epoch)

        writer.add_scalars('srgan/psnr', {
            'train_psnr': train_psnr,
            'valid_psnr': valid_psnr,
        }, epoch)


def valid(netG, netD, valid_iter, epoch):
    netG.eval()
    netD.eval()

    valid_loss = 0
    valid_psnr = 0

    with torch.no_grad():
        for step, (lr_img, hr_img) in enumerate(valid_iter):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            fake_img = netG(lr_img).clamp(0., 1.)  # 锚定值域
            loss = criterionG(fake_img, hr_img, netD(fake_img))

            valid_loss += loss.item()
            valid_psnr += calculate_psnr(fake_img, hr_img).item()

            if step == len(valid_iter) - 1:
                writer.add_image(f'srgan/epoch {epoch} sr',
                             make_grid(fake_img[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)
                writer.add_image(f'srgan/epoch {epoch} hr',
                             make_grid(hr_img[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)

        valid_loss /= (step + 1)
        valid_psnr /= (step + 1)
        print(f"valid [epoch{epoch:>2d}]: psnr = {valid_psnr:.6f}, G Loss = {valid_loss:.6f}")
        print("-" * 55)
    return valid_loss, valid_psnr


if __name__ == "__main__":

    train_path = '/root/autodl-tmp/DIV2K/DIV2K_train_HR_560'
    valid_path = '/root/autodl-tmp/DIV2K/DIV2K_valid_HR'
    train_set = SrDataSet(train_path, crop_size, scaling_size)
    valid_set = SrDataSet(valid_path, crop_size, scaling_size)

    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_iter = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8)

    n = input('是否从已训练模型种继续训练(y/n): ')

    if n == 'y':
        checkpoint = torch.load(checkpointGan)

        netG.load_state_dict(checkpoint['modelG'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])

        netD.load_state_dict(checkpoint['modelD'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])

        epoch = checkpoint['epoch']
        psnr = checkpoint['psnr']
        train(netG, netD, train_iter, valid_iter, epoch, psnr)

        print('加载已训练模型')

    else:
        print('已加载初始模型')
        train(netG, netD, train_iter, valid_iter)
