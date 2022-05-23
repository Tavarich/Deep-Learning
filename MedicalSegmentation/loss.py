import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        intersect = (input * target).sum()
        union = torch.sum(input) + torch.sum(target)
        dice = (2 * intersect + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        return dice_loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """
    def __init__(self, num_classes = 2):
        super(MulticlassDiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, input, target, weights=None):
        
        # (batch_size, 1, h, w) -> (batch_size, h, w, num_classes) -> (batch_size, num_class, h, w)
        target = F.one_hot(target.long().squeeze(dim=1), num_classes=self.num_classes).permute(0, 3, 1, 2)
        
        c = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        total_loss = 0
 
        for i in range(c):
            dice_loss = dice(input[:,i], target[:,i])
            if weights is not None:
                dice_loss *= weights[i]
            total_loss += dice_loss
 
        return total_loss


if __name__ == '__main__':
    loss = MulticlassDiceLoss()
    x = torch.randn(10, 2, 224, 224)
    y = torch.ones(10, 1, 224, 224)
    print(loss(x, y))