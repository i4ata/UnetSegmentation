"""Script to train a Unet"""

from unet import Unet
from dataset import SegmentationDataset

import torch
import albumentations as A

import argparse
from typing import Tuple
import logging

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=Tuple[int, int], nargs=2, default=(320,320), help='Model input size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Optimizer learning rate')
    parser.add_argument('--encoder_name', type=str, default='timm-efficientnet-b0', help='Unet encoder')
    parser.add_argument('--augmentation', type=bool, default=True, help='Include data augmentation for training')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=2, help='Early stopping patience')
    parser.add_argument('--train_size', type=float, default=.8, help='Proportion of data used for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Model batch size')
    parser.add_argument('--model_name', type=str, default='unet', help='Name of the model')
    parser.add_argument('--log_dir', type=str, default='runs', help='Where to load the model training')
    parser.add_argument('--device', type=str, default='cuda', choices=('cpu', 'cuda'), help='Device on which to train the model')

    return parser.parse_args()

if __name__ == '__main__':

    random_state = 0
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)

    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('device set to cuda but no cuda found. Defaulting to cpu')
        args.device = 'cpu'

    # Training data augmentation
    TRAIN_TRANSFORM = A.Compose(
        transforms=[
            A.Resize(*args.image_size),
            A.HorizontalFlip(p=.5),
            A.VerticalFlip(p=.5)
        ], 
        is_check_shapes=False
    )

    # Testing data augmentation
    TEST_TRANSFORM = A.Compose(
        transforms=[
            A.Resize(*args.image_size) 
        ], 
        is_check_shapes=False
    )

    dataset = SegmentationDataset()
    dataset.split(train_size=args.train_size, 
                  train_transform=TRAIN_TRANSFORM if args.augmentation else TEST_TRANSFORM,
                  test_transform=TEST_TRANSFORM,
                  random_state=random_state)
    dataset.get_dataloaders(batch_size=args.batch_size)
    model = Unet(name=args.model_name, image_size=args.image_size, device=args.device)
    model.configure_optimizers(lr=args.lr, patience=args.patience)
    model.train_model(dataset.train_dataloader, 
                      dataset.test_dataloader, 
                      epochs=args.epochs, 
                      log_dir=args.log_dir)