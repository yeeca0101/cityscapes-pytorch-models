import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Cityscapes
from tensorboardX import SummaryWriter  # TensorBoard를 위한 SummaryWriter
from unet.unet_model import Unet
import argparse
import numpy as np
from tqdm import tqdm

# Argument Parsing
parser = argparse.ArgumentParser(description='U-Net Cityscapes Training with PyTorch')
parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--epochs', default=250, type=int, help='number of epochs')
parser.add_argument('--repeat', default=1, type=int, help='number of repetitive training')
parser.add_argument('--data_dir', default='/data/1003_Cityscapes', type=str, help='directory to save checkpoints')
parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str, help='directory to save checkpoints')
parser.add_argument('--log_dir', default='./logs', type=str, help='directory to save logs')
args = parser.parse_args()

# Checkpoint and Log Directories
os.makedirs(args.checkpoint_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Data Preparation
transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

id_to_train_id = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255,
    10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6,
    20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255,
    30: 255, 31: 16, 32: 17, 33: 18
}

class_names = [
    'unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground',
    'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge',
    'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle'
]

def label_map(label):
    label = np.array(label)
    mapped_label = np.zeros_like(label, dtype=np.uint8)
    for k, v in id_to_train_id.items():
        mapped_label[label == k] = v
    return mapped_label

target_transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.Lambda(label_map)
])

train_dataset = Cityscapes(root=args.data_dir, split='train', mode='fine',
                           target_type='semantic', transform=transform, target_transform=target_transform)
val_dataset = Cityscapes(root=args.data_dir, split='val', mode='fine',
                         target_type='semantic', transform=transform, target_transform=target_transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# U-Net Model Initialization
act = nn.ReLU()
model = Unet(in_channels=3, n_classes=30, act=act).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=255)

def calculate_iou(pred, label, num_classes):
    ious = []
    pred = pred.view(-1)
    label = label.view(-1)
    iou_per_class = {}

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = label == cls

        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection

        if union > 0:
            iou = float(intersection) / float(union)
            ious.append(iou)
            iou_per_class[cls] = iou
        else:
            ious.append(float('nan'))
            iou_per_class[cls] = float('nan')

    return np.nanmean(ious), iou_per_class

def evaluate(model, data_loader):
    model.eval()
    pixel_accuracy = 0.0
    total_miou = 0.0
    num_classes = 30
    iou_per_class_accum = {cls: 0.0 for cls in range(num_classes)}

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            pixel_accuracy += (predicted == labels).sum().item() / (labels.size(0) * labels.size(1) * labels.size(2))
            miou, iou_per_class = calculate_iou(predicted, labels, num_classes)
            total_miou += miou

            for cls, iou in iou_per_class.items():
                iou_per_class_accum[cls] += iou

    pixel_accuracy /= len(data_loader)
    total_miou /= len(data_loader)

    for cls, iou in iou_per_class_accum.items():
        print(f'{class_names[cls]}({cls}): IoU = {iou / len(data_loader):.4f}')

    return pixel_accuracy, total_miou

def train(epoch, model, optimizer, train_loader, criterion):
    model.train()
    running_loss = 0.0
    pixel_accuracy = 0.0

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch}')):
        images = images.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        pixel_accuracy += (predicted == labels).sum().item() / (labels.size(0) * labels.size(1) * labels.size(2))

    return running_loss / len(train_loader), pixel_accuracy / len(train_loader)

def main():
    best_acc = 0  # Initialize best accuracy
    best_model_file_name = ""

    for repeat_idx in range(1, args.repeat + 1):
        model = Unet(in_channels=3, n_classes=30, act=nn.ReLU()).to(device)  # Reset the model for each repeat
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        log_dir = os.path.join(args.log_dir, f'repeat_{repeat_idx}')
        writer = SummaryWriter(logdir=log_dir)

        for epoch in range(args.epochs):
            train_loss, train_pixel_acc = train(epoch, model, optimizer, train_loader, criterion)
            val_pixel_acc, val_miou = evaluate(model, val_loader)

            # Log results
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Pixel Accuracy/train', train_pixel_acc, epoch)
            writer.add_scalar('Pixel Accuracy/val', val_pixel_acc, epoch)
            writer.add_scalar('mIOU/val', val_miou, epoch)

            # Save the best model and remove previous best
            if val_pixel_acc > best_acc:
                print('Saving best model...')
                state = {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'val_pixel_acc': val_pixel_acc,
                    'val_miou': val_miou,
                }
                if best_model_file_name:
                    try:
                        os.remove(os.path.join(args.checkpoint_dir, best_model_file_name))
                    except FileNotFoundError:
                        pass

                best_model_file_name = f'best_model_repeat_{repeat_idx}_epoch_{epoch}.pth'
                torch.save(state, os.path.join(args.checkpoint_dir, best_model_file_name))
                best_acc = val_pixel_acc

        writer.close()

if __name__ == '__main__':
    main()
