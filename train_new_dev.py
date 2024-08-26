import os
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter  
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial

from unet.unet_model import Unet,ResNeXtUnet
from unet.model import build_model
from cityscapes import build_datasets
from losses import SegmentationLoss
from utils import *
from activations import *

# Argument Parsing
parser = argparse.ArgumentParser(description='U-Net Cityscapes Training with PyTorch')
parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--epochs', default=250, type=int, help='number of epochs')
parser.add_argument('--repeat', default=1, type=int, help='number of repetitive training')
parser.add_argument('--data_dir', default='/data/1003_Cityscapes', type=str, help='cityscapes dir')
parser.add_argument('--checkpoint_dir', default='', type=str, help='directory to save checkpoints')
parser.add_argument('--log_dir', default='./logs', type=str, help='directory to save logs')
parser.add_argument('--n_classes', default=20, type=int, help='cityscapes classes')
parser.add_argument('--device_id', default=3, type=int, help='single gpu id')
parser.add_argument('--img_width', default=512, type=int, help='train img size')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision training')
parser.add_argument('--arch', default='resnext', type=str, help='architecture')
parser.add_argument('--pretrained', default=False, type=bool, help='coco pretrained')

args = parser.parse_args()

# Checkpoint and Log Directories
checkpoint_dir = os.path.join('./checkpoints',args.checkpoint_dir)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')


# only used validation. use argmax fn
def calculate_iou(pred, target, num_classes, ignore_index=None):
    # pred : model(inp).argmax(dim=c) 
    pred = pred.long()
    target = target.long()
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Create a mask for valid pixels if ignore_index is provided
    if ignore_index is not None:
        valid_mask = target != ignore_index
        pred = pred[valid_mask]
        target = target[valid_mask]
    
    # Ensure all values are within the valid range
    pred = torch.clamp(pred, 0, num_classes - 1)
    target = torch.clamp(target, 0, num_classes - 1)
    
    # One-hot encoding
    pred_one_hot = torch.nn.functional.one_hot(pred, num_classes=num_classes)
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes)

    # Intersection and Union
    intersection = (pred_one_hot & target_one_hot).sum(dim=0)
    union = (pred_one_hot | target_one_hot).sum(dim=0)

    # IoU calculation
    iou = intersection.float() / union.float()
    iou[union == 0] = float('nan')  # Set IoU to NaN where union is zero

    return iou.tolist()


def evaluate(model, data_loader,ignore_index=None):
    model.eval()
    pixel_accuracy = 0.0
    total_miou = 0.0
    num_classes = args.n_classes if ignore_index is None else args.n_classes -1 # ignore index = 19
    class_ious = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Validating', leave=False):
            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)['out']
            _, predicted = torch.max(outputs.data, 1)

            pixel_accuracy += (predicted == labels).sum().item() / (labels.size(0) * labels.size(1) * labels.size(2))
            ious = calculate_iou(predicted, labels, num_classes,ignore_index=ignore_index)
            for cls in range(num_classes):
                if not np.isnan(ious[cls]):
                    class_ious[cls].append(ious[cls])
            total_miou += np.nanmean(ious)

    pixel_accuracy /= len(data_loader)
    total_miou /= len(data_loader)

    class_miou = [np.mean(cls_ious) if cls_ious else float('nan') for cls_ious in class_ious]

    return pixel_accuracy, total_miou, class_miou

def train(epoch, model, optimizer, train_loader, criterion, scaler=None):
    model.train()
    running_loss = 0.0
    total_pixel_acc = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)['out']
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)['out']
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

    # build dataset
    hw = (256,512) if args.img_width == 512 else (512,1024) 
    train_loader, val_loader = build_datasets(batch_size=args.batch_size,hw=hw)

    criterion = SegmentationLoss() 
    if args.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for act_name, act_fn in acts_dict.items():
        if args.arch in ['unet','resnextunet']:
            model_class = Unet if args.arch !='resnext' else ResNeXtUnet
            model = model_class(in_channels=3, n_classes=args.n_classes, act=act_fn)
        else:
            model = build_model(args.arch,n_classes=args.n_classes,replace_type='instance',pretrained=args.pretrained)
        
        model.to(device)
        print(model.__class__.__name__)
        print(f'repeat id :{repeat_idx}, act : {act_name}')
        print_count_params(model)
        log_dir = os.path.join(args.log_dir,args.checkpoint_dir,f'repeat_{repeat_idx}',act_name)
        print('log dir : ',log_dir)
        writer = SummaryWriter(logdir=log_dir)
        
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * args.lr},
            {'params': model.classifier.parameters(), 'lr': args.lr},
        ], lr=args.lr, momentum=0.9, weight_decay=1e-4)

        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1,verbose=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//5, gamma=0.1)

        for epoch in range(args.epochs):
            train_loss, train_pixel_acc = train(epoch, model, optimizer, train_loader, criterion, scaler=scaler)

            scheduler.step()
            val_pixel_acc, val_miou, class_miou = evaluate(model, val_loader)
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Pixel_Accuracy/train', train_pixel_acc, epoch)
            writer.add_scalar('Pixel_Accuracy/val', val_pixel_acc, epoch)
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
                        if os.path.isfile(os.path.join(checkpoint_dir, best_model_file_name)):
                            os.remove(os.path.join(checkpoint_dir, best_model_file_name))
                    except FileNotFoundError:
                        pass

                best_model_file_name = f'{args.arch}_{act_name}_{repeat_idx}_{epoch}_{val_miou*100:.2f}.pth'
                torch.save(state, os.path.join(checkpoint_dir, best_model_file_name))
                best_miou = val_miou

        writer.close()

if __name__ == '__main__':
    acts = {
        'SwishT_C':SwishT_C(beta_init=1.0003)
        # 'SMU':SMU(alpha=0.0),
        # 'SwishT_B':SwishT_B(beta_init=1.0004)
        # 'ReLU':nn.ReLU(inplace=True)
        }
    for repeat_idx in range(1, args.repeat + 1):
        main(acts,repeat_idx=repeat_idx)
        torch.cuda.empty_cache()
