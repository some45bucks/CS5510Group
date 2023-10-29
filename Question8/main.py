from torchvision.datasets import ImageNet

print('Loading ImageNet Dataset (this may take a while)...')
validation_set = ImageNet(root='./dataset', split='val')
print('Finished loading ImageNet Dataset')
