import torch
from torchvision.models.segmentation import deeplabv3_resnet50

def create_deeplabv3(num_classes):
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model
