import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
import argparse
import numpy as np
from functools import partial
from unet.unet_model import Unet, ResNeXtUnet
from unet.model import build_model
from cityscapes import build_datasets
from losses import SegmentationLoss
from utils import *
from activations import *


parser = argparse.ArgumentParser(description='U-Net Cityscapes Training with PyTorch Lightning')
parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--epochs', default=250, type=int, help='number of epochs')
parser.add_argument('--repeat', default=1, type=int, help='number of repetitive training')
parser.add_argument('--data_dir', default='/data/1003_Cityscapes', type=str, help='cityscapes dir')
parser.add_argument('--checkpoint_dir', default='', type=str, help='directory to save checkpoints')
parser.add_argument('--log_dir', default='./logs', type=str, help='directory to save logs')
parser.add_argument('--n_classes', default=20, type=int, help='cityscapes classes')
parser.add_argument('--device_ids', default='0', type=str, help='comma-separated list of GPU ids to use for training')    
parser.add_argument('--img_width', default=512, type=int, help='train img size')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision training')
parser.add_argument('--arch', default='resnext', type=str, help='architecture')
parser.add_argument('--pretrained', default=False, type=bool, help='coco pretrained')
parser.add_argument('--use_ddp',default=False, type=bool, help='use Distributed Data Parallel')
args = parser.parse_args()



class SegmentationModel(pl.LightningModule):
    def __init__(self, model, n_classes, lr, act_name):
        super().__init__()
        self.model = model
        self.n_classes = n_classes
        self.lr = lr
        self.act_name = act_name
        self.criterion = SegmentationLoss()
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)['out']

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        pixel_acc = (predicted == labels).sum().item() / (labels.numel())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log('train_pixel_acc', pixel_acc, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        pixel_acc = (predicted == labels).sum().item() / (labels.numel())
        ious = calculate_iou(predicted, labels, self.n_classes,ignore_index=19)
        self.validation_step_outputs.append({'val_loss': loss, 'val_pixel_acc': pixel_acc, 'val_ious': ious})
        return {'val_loss': loss, 'val_pixel_acc': pixel_acc, 'val_ious': ious}

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_pixel_acc = torch.tensor([x['val_pixel_acc'] for x in outputs]).mean()
        all_ious = torch.stack([torch.tensor(x['val_ious']) for x in outputs])
        avg_miou = all_ious.mean(dim=0).nanmean()
        
        self.log('val_loss', avg_loss, prog_bar=True,sync_dist=True)
        self.log('val_pixel_acc', avg_pixel_acc.item(), prog_bar=True,sync_dist=True)
        self.log('val_miou', avg_miou.item(), prog_bar=True,sync_dist=True)

        class_names = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
            'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void'
        ]
        
        class_ious = all_ious.mean(dim=0)
        for idx, (name, iou) in enumerate(zip(class_names, class_ious)):
            self.log(f'val_iou_{name}', iou.item(),sync_dist=True)
            print(f"{name}({idx}): {iou.item():.4f}")
        
        self.validation_step_outputs.clear()  # 메모리를 위해 출력 목록을 비웁니다


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=self.lr * 0.1)
        return [optimizer], [scheduler]

class CityscapesDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, img_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size

    def setup(self, stage=None):
        self.train_loader, self.val_loader = build_datasets(batch_size=self.batch_size, hw=self.img_size)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

class CustomCheckpoint(Callback):
    def __init__(self, checkpoint_dir,repeat_idx):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.best_miou = 0
        self.best_model_file_name = ""
        self.repeat_idx = repeat_idx

    def on_validation_end(self, trainer, pl_module):
        if trainer.is_global_zero:  # DDP에서 마스터 프로세스만 체크포인트 저장
            val_miou = trainer.callback_metrics.get('val_miou', 0)
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                if self.best_model_file_name:
                    old_path = os.path.join(self.checkpoint_dir, self.best_model_file_name)
                    if os.path.exists(old_path):
                        os.remove(old_path)
                
                self.best_model_file_name = f'{pl_module.model.__class__.__name__}_{pl_module.act_name}_{self.repeat_idx}_{trainer.current_epoch}_{val_miou*100:.2f}.pth'
                new_path = os.path.join(self.checkpoint_dir, self.best_model_file_name) # {act_name}_{repeat_idx}_{epoch}_{val_miou:.2f}.pth
                trainer.save_checkpoint(new_path)

def calculate_iou(pred, target, num_classes, ignore_index=None):
    pred = pred.long().view(-1)
    target = target.long().view(-1)  # target을 long 타입으로 변환하여 인덱스로 사용
    
    if ignore_index is not None:
        valid_mask = target != ignore_index
        pred = pred[valid_mask]
        target = target[valid_mask]
    
    pred = torch.clamp(pred, 0, num_classes - 1)
    target = torch.clamp(target, 0, num_classes - 1)
    
    pred_one_hot = torch.nn.functional.one_hot(pred, num_classes=num_classes)
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes)
    
    intersection = (pred_one_hot & target_one_hot).sum(dim=0).float()
    union = (pred_one_hot | target_one_hot).sum(dim=0).float()
    
    iou = intersection / union
    iou[union == 0] = float('nan')
    
    return iou

def main(acts_dict, repeat_idx):
    if args.use_ddp:
        strategy = 'ddp'
        devices = [int(id) for id in args.device_ids.split(',')] 
    else:
        strategy = None
        devices = [int(args.device_ids.split(',')[0])]  # 첫 번째 GPU만 사용
        
    for act_name, act_fn in acts_dict.items():
        if args.arch in ['unet', 'resnextunet']:
            model_class = Unet if args.arch != 'resnext' else ResNeXtUnet
            model = model_class(in_channels=3, n_classes=args.n_classes, act=act_fn)
        else:
            model = build_model(args.arch, n_classes=args.n_classes, replace_type='instance', pretrained=args.pretrained)
        
        print(model.__class__.__name__)
        print(f'repeat id: {repeat_idx}, act: {act_name}')
        print_count_params(model)
        
        lightning_model = SegmentationModel(model, args.n_classes, args.lr, act_name)
        data_module = CityscapesDataModule(args.data_dir, args.batch_size, (args.img_width, args.img_width*2))
        
        logger = TensorBoardLogger(save_dir=args.log_dir, name=f'{args.checkpoint_dir}/repeat_{repeat_idx}/{act_name}')
        
        checkpoint_callback = CustomCheckpoint(checkpoint_dir,repeat_idx)
        
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            devices=2 ,
            logger=logger,
            precision=16 if args.mixed_precision else 32,
            callbacks=[checkpoint_callback],
            strategy=strategy
        )
        
        trainer.fit(lightning_model, data_module)


if __name__ == '__main__':

    checkpoint_dir = os.path.join('./checkpoints', args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    acts = {
        'SwishT_C': SwishT_C(beta_init=1.0003)
    }
    
    for repeat_idx in range(1, args.repeat + 1):
        main(acts, repeat_idx=repeat_idx)
        torch.cuda.empty_cache()