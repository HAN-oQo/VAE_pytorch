import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

def mnist(path_to_data, batch_size=16, size=28, train=True, download=False):
    """MNIST dataloader.
    Args:
        path_to_data (string): Path to MNIST data files.
        batch_size (int):
        size (int): Size (height and width) of each image. Default is 28 for no resizing. 
    """
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5), std = (0.5))
    ])

    dataset = datasets.MNIST(path_to_data, train=train, download=download,
                             transform=all_transforms)
                        

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, len(dataset)

def CelebA(path_to_data, batch_size = 64, size = 128, train = True):

    all_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])

    if train:
        data_path = os.path.join(path_to_data, 'trainset')
    else:
        data_path = os.path.join(path_to_data, 'testset')
    
    dataset = datasets.ImageFolder(data_path, transform = all_transforms)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = train)

    return dataloader, len(dataset)


# def celebahq(path_to_data, batch_size=16, size=256):
#     all_transforms = transforms.Compose([
#         transforms.Resize(size),
#         transforms.ToTensor()
#     ])

#     dataset = datasets.ImageFolder(path_to_data, transform=all_transforms)

#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     return dataloader