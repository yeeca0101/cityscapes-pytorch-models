import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import Cityscapes
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Mapping from original Cityscapes label IDs to training IDs
id_to_train_id = {
    0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 19, 6: 19, 7: 0, 8: 1, 9: 19,
    10: 19, 11: 2, 12: 3, 13: 4, 14: 19, 15: 19, 16: 19, 17: 5, 18: 19, 19: 6,
    20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 19,
    30: 19, 31: 16, 32: 17, 33: 18
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

class CustomCityscapes(Cityscapes):
    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None, target_transform=None):
        super(CustomCityscapes, self).__init__(root, split=split, mode=mode, target_type=target_type, transform=None, target_transform=None)
        self.transform = transform

    def __getitem__(self, index):
        image, target = super(CustomCityscapes, self).__getitem__(index)

        # Apply the label mapping ndarray for albumentation 
        target = label_map(target)
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image, mask=target)
            image = augmented['image']
            target = augmented['mask']

        return image, target

def build_datasets(data_dir='/data/1003_Cityscapes', batch_size=8,hw=(512,1024)):
    # Define augmentations using Albumentations for the training set
    train_transform = A.Compose([
        # A.Resize(height=1024, width=2048,interpolation=cv2.INTER_NEAREST),
        # A.RandomScale(scale_limit=(-0.4, 1.2), p=0.5), #
        A.RandomCrop(height=768, width=768),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.3), #
        A.Rotate(limit=30, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Define a transform without augmentations for the validation set using Albumentations
    val_transform = A.Compose([
        # A.Resize(1024, width=2048,interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


    # Create datasets using the custom Cityscapes class
    train_dataset = CustomCityscapes(root=data_dir, split='train', mode='fine',
                                     target_type='semantic', transform=train_transform)

    val_dataset = CustomCityscapes(root=data_dir, split='val', mode='fine',
                                   target_type='semantic', transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

if __name__ == '__main__':
    train_loader, val_loader = build_datasets('/data/1003_Cityscapes',8)
    print(f'train set : \n',train_loader.dataset)
    print(f'train set : \n',val_loader.dataset)

    for images,targets in train_loader:
        print(images.shape)
        print(targets.shape)
        print(targets.unique())
        break