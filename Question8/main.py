import torch

from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import ImageNet
from torchvision.models import alexnet
from torchvision.transforms import transforms

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)
print(f'Using device: {device_type}\n')

NUM_IMAGES = 100

# Pre-process the images to match the input size of torchvision's AlexNet, using the same pre-processing as in the original paper
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print('Loading ImageNet Dataset (this may take a while)...')
validation_set = ImageNet(root='./dataset', split='val', transform=data_transform)
with open('./dataset/imagenet_classes.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
print('Finished loading ImageNet Dataset\n')

print('Loading AlexNet model with pre-trained weights...')
model = alexnet(weights='IMAGENET1K_V1').eval().to(device)
print('Finished loading AlexNet model\n')

val_loader = DataLoader(validation_set, batch_size=64, sampler=RandomSampler(validation_set, num_samples=NUM_IMAGES))

print('Performing classification on ImageNet validation set...')
top_1_correct_count = 0
top_5_correct_count = 0
batch_num = 0
with torch.no_grad():
    for images, labels in val_loader:
        batch_num += 1
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, top_5_predictions = torch.topk(outputs, k=5, dim=1)
        top_1_predictions = top_5_predictions[:, 0]

        top_1_correct_count += torch.sum(top_1_predictions == labels).item()
        top_5_correct_count += torch.sum(torch.any(top_5_predictions == labels.unsqueeze(dim=1), dim=1)).item()
        
        top_1_batch_accuracy = torch.sum(top_1_predictions == labels).item() / labels.size(0)
        top_5_batch_accuracy = torch.sum(torch.any(top_5_predictions == labels.unsqueeze(dim=1), dim=1)).item() / labels.size(0)
        print(f'Batch {batch_num}: Top 1 Accuracy: {top_1_batch_accuracy * 100:.2f}%, Top 5 Accuracy: {top_5_batch_accuracy * 100:.2f}%')

print('Finished performing classification\n')

print('Results:')
top_1_accuracy = top_1_correct_count / val_loader.sampler.num_samples
top_5_accuracy = top_5_correct_count / val_loader.sampler.num_samples
print(f'Top-1 Accuracy: {top_1_accuracy * 100:.2f}%')
print(f'Top-5 Accuracy: {top_5_accuracy * 100:.2f}%')
