import torch

from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.models import alexnet
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)
print(f'Using device: {device_type}')

# Pre-process the images to match the input size of torchvision's AlexNet, using the same pre-processing as in the original paper
data_transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print('Loading ImageNet Dataset (this may take a while)...')
validation_set = ImageNet(root='./dataset', split='val', transform=data_transform)
print('Finished loading ImageNet Dataset')

print('Loading AlexNet model with pre-trained weights...')
model = alexnet(weights='IMAGENET1K_V1').eval().to(device)
print('Finished loading AlexNet model')

val_loader = DataLoader(validation_set, batch_size=100, shuffle=True)

print('Performing classification on ImageNet validation set...')
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        # TODO: Probably should report this using a both top-1 and top-5 accuracy
        _, predictions = torch.max(outputs, dim=1)
        print('Accuracy: {:.2f}%'.format(torch.sum(predictions == labels).item() / labels.size(0) * 100))
        break # Only classify one batch of 100 images

print('Finished performing classification')
