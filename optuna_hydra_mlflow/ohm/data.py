import os
import torch
import torchvision
from torchvision import transforms


def get_dataloader(batch_size, validation_ratio):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))])
    print(os.path.abspath(__file__))
    dataset = torchvision.datasets.MNIST(root="/home/kei/notebooks/hydra/data",
                                         train=True,
                                         download=True,
                                         transform=transform)
    nb_val = int(len(dataset) * validation_ratio)

    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - nb_val, nb_val])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=2)

    return train_loader, valid_loader
