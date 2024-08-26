import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Cityscapes
from tensorboardX import SummaryWriter  # TensorBoard를 위한 SummaryWriter
import argparse
import numpy as np
from tqdm import tqdm

from unet.unet_model import Unet,ResNeXtUnet
from unet.model import build_model
from utils import *


# Argument Parsing
parser = argparse.ArgumentParser(description='Cityscapes Training with PyTorch')
parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--epochs', default=250, type=int, help='number of epochs')
parser.add_argument('--repeat', default=1, type=int, help='number of repetitive training')
parser.add_argument('--data_dir', default='/data/1003_Cityscapes', type=str, help='cityscapes dir')
parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str, help='directory to save checkpoints')
parser.add_argument('--log_dir', default='./logs', type=str, help='directory to save logs')
parser.add_argument('--arch', default='resnext', type=str, help='architecture')
parser.add_argument('--n_classes', default=21, type=int, help='number of classes')
parser.add_argument('--pretrained', default=False, type=bool, help='coco pretrained')

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



def calculate_iou(pred, label, num_classes):
    ious = []
    pred = pred.view(-1)
    label = label.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = label == cls

        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection

        if union > 0:
            ious.append(float(intersection) / float(union))
        else:
            ious.append(float('nan'))

    return ious

def evaluate(model, data_loader):
    model.eval()
    pixel_accuracy = 0.0
    total_miou = 0.0
    num_classes = 30
    class_ious = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Validating', leave=False):
            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            pixel_accuracy += (predicted == labels).sum().item() / (labels.size(0) * labels.size(1) * labels.size(2))
            ious = calculate_iou(predicted, labels, num_classes)
            for cls in range(num_classes):
                if not np.isnan(ious[cls]):
                    class_ious[cls].append(ious[cls])
            total_miou += np.nanmean(ious)

    pixel_accuracy /= len(data_loader)
    total_miou /= len(data_loader)

    class_miou = [np.mean(cls_ious) if cls_ious else float('nan') for cls_ious in class_ious]

    return pixel_accuracy, total_miou, class_miou

def train(epoch, model, optimizer, train_loader, criterion):
    model.train()
    running_loss = 0.0
    total_pixel_acc = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        pixel_acc = (predicted == labels).sum().item() / (labels.size(0) * labels.size(1) * labels.size(2))
        total_pixel_acc += pixel_acc

        pbar.set_postfix({'Loss': running_loss / (batch_idx + 1), 'Pixel Acc': total_pixel_acc / (batch_idx + 1)})

    return running_loss / len(train_loader), total_pixel_acc / len(train_loader)

def main(acts_dict,repeat_idx):
    best_miou = 0
    best_model_file_name = ""

    class_names = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
        'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
        'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void'
    ]
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    
    for act_name, act_fn in acts_dict.items():
        if args.arch in ['unet','resnextunet']:
            model_class = Unet if args.arch !='resnext' else ResNeXtUnet
            model = model_class(in_channels=3, n_classes=args.n_classes, act=act_fn)
        else:
            build_model(args.arch,n_classes=args.n_classes,replace_type='instance',pretrained=args.pretrained)
        model.to(device)
        print(model_class.__name__)
        print(f'repeat id :{repeat_idx}, act : {act_name}')
        print_count_params(model)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        log_dir = os.path.join(args.log_dir, f'repeat_{repeat_idx}',act_name)
        print('log dir : ',log_dir)
        writer = SummaryWriter(logdir=log_dir)

        for epoch in range(args.epochs):
            train_loss, train_pixel_acc = train(epoch, model, optimizer, train_loader, criterion)
            val_pixel_acc, val_miou, class_miou = evaluate(model, val_loader)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Pixel Accuracy/train', train_pixel_acc, epoch)
            writer.add_scalar('Pixel Accuracy/val', val_pixel_acc, epoch)
            writer.add_scalar('mIOU/val', val_miou, epoch)

            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Pixel Acc: {train_pixel_acc:.4f}")
            print(f"Val Pixel Acc: {val_pixel_acc:.4f}, Val mIOU: {val_miou:.4f}")
            print("\nClass-wise IOU:")
            for idx, (name, iou) in enumerate(zip(class_names, class_miou)):
                print(f"{name}({idx}): {iou:.4f}")
            print()

            if val_miou > best_miou:
                print('Saving best model...')
                state = {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'val_pixel_acc': val_pixel_acc,
                    'val_miou': val_miou,
                }
                if best_model_file_name:
                    try:
                        if os.path.isfile(os.path.join(args.checkpoint_dir, best_model_file_name)):
                            os.remove(os.path.join(args.checkpoint_dir, best_model_file_name))
                    except FileNotFoundError:
                        pass

                best_model_file_name = f'{args.arch}_{act_name}_{repeat_idx}_{epoch}_{val_miou:.2f}.pth'
                torch.save(state, os.path.join(args.checkpoint_dir, best_model_file_name))
                best_miou = val_miou

        writer.close()

if __name__ == '__main__':
    acts = {
        'ReLU':nn.ReLU()
        }
    for repeat_idx in range(1, args.repeat + 1):
        main(acts,repeat_idx=repeat_idx)
        torch.cuda.empty_cache()
