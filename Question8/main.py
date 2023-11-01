import torch

from torch.utils.data import DataLoader, RandomSampler
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

# I'm going to assume that most GPUs that will run this should have enough memory to load all 100 images in a batch for inference
# So that I can simplify the code and not have to worry about computing metrics across multiple batches
val_loader = DataLoader(validation_set, batch_size=100, sampler=RandomSampler(validation_set, num_samples=100))

print('Performing classification on ImageNet validation set...')
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, top_5_predictions = torch.topk(outputs, k=5, dim=1)
        top_1_predictions = top_5_predictions[:, 0]

        top_1_accuracy = torch.sum(top_1_predictions == labels).item() / labels.size(0) * 100
        top_5_accuracy = torch.sum(torch.any(top_5_predictions == labels.unsqueeze(dim=1), dim=1)).item() / labels.size(0) * 100

        print(f'Top-1 Accuracy: {top_1_accuracy:.2f}%')
        print(f'Top-5 Accuracy: {top_5_accuracy:.2f}%')
        break # Only classify one batch of 100 images

print('Finished performing classification')
