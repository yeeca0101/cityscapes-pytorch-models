from torch import nn
import torch
import torchvision.models.segmentation as segmentation_models

# torchvision version : 0.15.2+cu118
sup_model = {
    'deeplabv3_resnet50': segmentation_models.deeplabv3_resnet50,
    'deeplabv3_resnet101': segmentation_models.deeplabv3_resnet101,
    'fcn_resnet50': segmentation_models.fcn_resnet50,
    'fcn_resnet101': segmentation_models.fcn_resnet101,
    # Add other models here if needed
}

def replace_activation(module, activation_old, activation_new,by=None):
    assert by in ('class','instance'), NameError(f'by arguments not support {by}')

    for name, child in module.named_children():
        if isinstance(child, activation_old): # nn.ReLU
            if by == 'class': # nn.ReLU
                setattr(module, name, activation_new())
            elif by == 'instance': # nn.ReLU()
                setattr(module, name, activation_new)
        else:
            replace_activation(child, activation_old, activation_new,by=by)


def build_model(arch, n_classes, act=None, replace_type='instance', pretrained=False):
    
    if arch not in sup_model:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    model = sup_model[arch](pretrained=pretrained)
    
    if isinstance(model.classifier, nn.Sequential):
        model.classifier[4] = nn.Conv2d(in_channels=model.classifier[4].in_channels, out_channels=n_classes, kernel_size=(1, 1))
    elif hasattr(model.classifier, 'project'):
        model.classifier[-1] = nn.Conv2d(in_channels=model.classifier[-1].in_channels, out_channels=n_classes, kernel_size=(1, 1))
    else:
        raise ValueError(f"Unsupported model classifier type: {type(model.classifier)}")
    
    if act:
        replace_activation(model, (nn.ReLU, nn.GELU), act, by=replace_type)
    
    return model

if __name__ == '__main__':
    inp = torch.randn((2,3,512,1024))
    model = build_model('fcn_resnet50', n_classes=20, act=nn.LeakyReLU(), pretrained=False)
    print(model)
    print(model(inp)['out'].shape)
    model = build_model('deeplabv3_resnet50', n_classes=20, act=nn.LeakyReLU(), pretrained=False)
    print(model(inp)['out'].shape)




