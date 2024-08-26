import torch
import torch.nn as nn
import torch.nn.functional as Func


class ContrastiveLoss(nn.Module):
    """Contrastive learing loss."""

    def __init__(self, margin: float | None = 1.0):
        """
        Initialize the class.
        
        Args:
            margin (float): margin for contrastive loss.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Calculate the positive pair distance
        pos_dist = Func.pairwise_distance(anchor, positive)
        # Calculate the negative pair distance
        neg_dist = Func.pairwise_distance(anchor, negative)

        # Calculate the contrastive loss
        loss = torch.mean((pos_dist ** 2) + torch.clamp(self.margin - neg_dist, min=0.0) ** 2)

        return loss


class DiceLoss(nn.Module):
    """The Dice Loss class"""
    def __init__(self, smooth: float | None = 1.0):
        """
        Initialize the class.

        Args:
            smooth (float): smooth parameter for dice loss. Default to `1.0`.
        """
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        _, num_classes, _, _ = pred.shape
        target = target - 1  # convert values from 1-3 to 0-2

        assert target.min() >= 0 and target.max() < num_classes, 'target contains invalid class indices'

        # Convert logits to probabilities
        pred = torch.nn.functional.softmax(pred, dim=1)

        # convert targets, trimap, to one shot
        # (batch_size, width, height) -> (batch_size, 3, width, height)
        targets_one_hot = torch.nn.functional.one_hot(target, num_classes).permute(0, 3, 1, 2)
        targets_one_hot = targets_one_hot.type_as(pred)

        # calculate dice_score for each (batch,class)
        # sum over width and height
        intersection = torch.sum(pred * targets_one_hot, dim=(2, 3))
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)

        dice_loss = 1. - dice_score.mean()  # aveerage over batch and num_classes

        return dice_loss
