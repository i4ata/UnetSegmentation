"""Script to train a Unet"""

from unet import Unet
from dataset import SegmentationDataset

import torch

if __name__ == '__main__':

    epochs = 25
    random_state = 0

    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)

    dataset = SegmentationDataset()
    dataset.split(random_state=random_state)
    dataset.get_dataloaders()

    model = Unet('unet')
    model.train_model(dataset.train_dataloader, dataset.test_dataloader, epochs=epochs)
