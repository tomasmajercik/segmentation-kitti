import torch

class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes, ignore_index=255, eps=1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)        # (B, C, H, W)
        masks = targets != self.ignore_index       # (B, H, W)

        preds = preds * masks.unsqueeze(1)

        one_hot = torch.zeros_like(preds)
        one_hot.scatter_(1, targets.unsqueeze(1).clamp(max=self.n_classes-1), 1)
        one_hot = one_hot * masks.unsqueeze(1)

        dims = (0, 2, 3)
        intersection = torch.sum(preds * one_hot, dims)
        cardinality = torch.sum(preds + one_hot, dims)

        dice_per_class = (2 * intersection + self.eps) / (cardinality + self.eps)
        dice = dice_per_class.mean()
        return 1 - dice