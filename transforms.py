from config import *
from torchvision import transforms

# Data loading and preprocessing.
transform = transforms.Compose([
    transforms.Resize([sample_size, sample_size]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# crop (fill with white), rotate, flip, normalize, add noise, to tensor
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomCrop([sample_size, sample_size], padding=4, padding_mode='reflect'),
    transforms.Resize([sample_size, sample_size]),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([sample_size, sample_size]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

_transforms = {
    'train': transform_train,
    'test': transform_test,
    'raw': transform
}


def get_transform(name):
    return _transforms[name]
