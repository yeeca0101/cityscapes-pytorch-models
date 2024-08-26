import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=None, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        
        # One-hot encode target
        target = target.long()
        target_one_hot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()
        
        # Create mask for ignored index
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float().unsqueeze(1)
        else:
            mask = torch.ones_like(pred)
        
        # Apply softmax to predictions
        pred = F.softmax(pred, dim=1)
        
        # Multiply pred and target by mask
        pred = pred * mask
        target_one_hot = target_one_hot * mask
        
        # Flatten pred and target
        pred = pred.view(B, C, -1)
        target_one_hot = target_one_hot.view(B, C, -1)
        
        # Compute Dice coefficient
        intersection = (pred * target_one_hot).sum(dim=2)
        union = pred.sum(dim=2) + target_one_hot.sum(dim=2)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1. - dice.mean()

class IOULoss(nn.Module):
    def __init__(self, ignore_index=None, smooth=1e-5):
        super(IOULoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        
        # One-hot encode target
        target = target.long()
        target_one_hot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()
        
        # Create mask for ignored index
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float().unsqueeze(1)
        else:
            mask = torch.ones_like(pred)
        
        # Apply softmax to predictions
        pred = F.softmax(pred, dim=1)
        
        # Multiply pred and target by mask
        pred = pred * mask
        target_one_hot = target_one_hot * mask
        
        # Flatten pred and target
        pred = pred.view(B, C, -1)
        target_one_hot = target_one_hot.view(B, C, -1)
        
        # Compute IOU
        intersection = (pred * target_one_hot).sum(dim=2)
        union = pred.sum(dim=2) + target_one_hot.sum(dim=2) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1. - iou.mean()

class SegmentationLoss(nn.Module):
    def __init__(self,ignore_index=None,scale=0.8) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.scale = scale
        self.dice_loss = DiceLoss(ignore_index=self.ignore_index)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index if ignore_index else -1)

    def forward(self,preds,targets):
        if not isinstance(targets,torch.LongTensor):
            targets = targets.long()
        losses = self.dice_loss(preds,targets) + self.ce_loss(preds,targets)
        return losses * self.scale

if __name__ == "__main__":
    # 예시 입력 생성
    B, C, H, W = 2, 3, 4, 4  # Batch size, Classes, Height, Width
    pred = torch.randn(B, C, H, W)  # 랜덤한 예측값 생성
    target = torch.randint(0, C, (B, H, W))  # 랜덤한 타겟 생성

    # ignore_index가 없는 경우
    dice_loss = SegmentationLoss()
    iou_loss = IOULoss()

    # ignore_index가 있는 경우
    dice_loss_ignore = SegmentationLoss(ignore_index=2)
    iou_loss_ignore = IOULoss(ignore_index=2)

    # 손실 계산
    dice_loss_value = dice_loss(pred, target)
    iou_loss_value = iou_loss(pred, target)
    dice_loss_ignore_value = dice_loss_ignore(pred, target)
    iou_loss_ignore_value = iou_loss_ignore(pred, target)

    print("입력 형태:")
    print(f"pred shape: {pred.shape}")
    print(f"target shape: {target.shape}")
    print("\n계산된 손실값:")
    print(f"Dice Loss (without ignore): {dice_loss_value.item():.4f}")
    print(f"IOU Loss (without ignore): {iou_loss_value.item():.4f}")
    print(f"Dice Loss (with ignore_index=2): {dice_loss_ignore_value.item():.4f}")
    print(f"IOU Loss (with ignore_index=2): {iou_loss_ignore_value.item():.4f}")

    # ignore_index 동작 확인
    print("\nignore_index 동작 확인:")
    target_with_ignore = target.clone()
    target_with_ignore[target_with_ignore == 2] = dice_loss_ignore.ignore_index
    dice_loss_ignore_value_2 = dice_loss_ignore(pred, target_with_ignore)
    iou_loss_ignore_value_2 = iou_loss_ignore(pred, target_with_ignore)
    
    print(f"Dice Loss (with actual ignored values): {dice_loss_ignore_value_2.item():.4f}")
    print(f"IOU Loss (with actual ignored values): {iou_loss_ignore_value_2.item():.4f}")

    # 결과 비교
    print("\n무시된 값의 비율:")
    ignored_ratio = (target_with_ignore == dice_loss_ignore.ignore_index).float().mean()
    print(f"Ignored values ratio: {ignored_ratio.item():.2%}")