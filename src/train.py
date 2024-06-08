import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms.functional import resize
import torchvision.transforms as transforms
from segmentation_models_pytorch.losses import DiceLoss
from torchmetrics.functional.segmentation import mean_iou
from typing import Tuple
from torchinfo import summary
from PIL import Image
import numpy as np
import argparse

from src.utils import val_transform, get_pretrained_unet
from src.dataset import SegmentationDataset
from src.early_stopper import EarlyStopper
from src.custom_unet import CustomUnet

class UnetLoss(nn.Module):
    def __init__(self) -> None:
        super(UnetLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(mode='binary')

    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return self.bce_loss(logits, masks) + self.dice_loss(logits, masks)

class Trainer:
    def __init__(self, model: nn.Module, name: str = 'default_name', device: str = 'cpu') -> None:
        self.model = model.to(device)
        self.name = name
        self.device = device

    def _train_step(self, train_dataloader: DataLoader) -> Tuple[float, float]:
        train_loss, train_iou = 0, 0
        self.model.train()
        for images, masks in train_dataloader:
            images, masks = images.to(self.device), masks.to(self.device)
            predictions = self.model(images)
            loss = self.loss_fn(predictions, masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss
            train_iou += mean_iou(predictions.sigmoid().round().long(), masks.long(), num_classes=1).mean()
        return train_loss / len(train_dataloader), train_iou / len(train_dataloader)

    def _val_step(self, val_dataloader: DataLoader) -> Tuple[float, float]:
        val_loss, val_iou = 0, 0
        self.model.eval()
        for images, masks in val_dataloader:
            images, masks = images.to(self.device), masks.to(self.device)
            with torch.inference_mode():
                predictions = self.model(images)
                loss = self.loss_fn(predictions, masks)
            val_loss += loss
            val_iou += mean_iou(predictions.sigmoid().round().long(), masks.long(), num_classes=1).mean()
        return val_loss / len(val_dataloader), val_iou / len(val_dataloader)

    def fit(self, train_size: float = .8, batch_size: int = 10, epochs: int = 10, learning_rate: float = .001) -> None:
        writer = SummaryWriter(log_dir='runs/' + self.name)
        dataset = SegmentationDataset()
        dataset.split(train_size=train_size)
        train_dataloader, val_dataloader = dataset.get_dataloaders(batch_size=batch_size)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)
        early_stopper = EarlyStopper()
        self.loss_fn = UnetLoss()

        for epoch in range(epochs):
            train_loss, train_iou = self._train_step(train_dataloader)
            val_loss, val_iou = self._val_step(val_dataloader)
            print(f'{epoch} | Train loss: {train_loss} | Train IoU: {train_iou} | Val loss: {val_loss} | Val IoU: {val_iou}', flush=True)
            if early_stopper.check(val_loss):
                print('Training stops early due to overfitting suspicion')
                break
            if early_stopper.save_model: torch.save(self.model.state_dict(), 'models/' + self.name + '.pt')
            writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalars('IoU', {'train': train_iou, 'val': val_iou}, epoch)
        else:
            print('Model might not have converged')
        writer.close()

    def print_summary(self):
        print(summary(self.model, input_size=(16, 3, 320, 320)))

    def predict(self, image_path: str) -> np.ndarray:
        self.model.eval()
        
        image = np.asarray(Image.open(image_path))
        h, w = image.shape[:-1]

        image = torch.from_numpy(val_transform(image=image)['image']).float().permute(2,0,1) / 255.
        with torch.inference_mode():
            prediction = self.model(image.to(self.device).unsqueeze(0))[0].sigmoid().round().cpu()
        mask = resize(img=prediction, size=(h,w), interpolation=transforms.InterpolationMode.NEAREST)[0].numpy()
        return mask

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',          type=str,   default='custom',        help='Whether to use the custom implementation or the pretrained one')
    parser.add_argument('--epochs',         type=int,   default=20,              help='Number of epochs to train the model for')
    parser.add_argument('--learning_rate',  type=float, default=1e-3,            help='Model learning rate')
    parser.add_argument('--name',           type=str,   default='default_name',  help='Experiment name')
    parser.add_argument('--batch_size',     type=int,   default=16,              help='Dataloader batch size')
    parser.add_argument('--train_size',     type=float, default=.8,              help='Proportion of data to use for training')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    model = CustomUnet() if args.model == 'custom' else get_pretrained_unet()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(model=model, name=args.name, device=device)
    print('Training starts', flush=True)
    trainer.fit(
        train_size=args.train_size,
        batch_size=args.batch_size,
        epochs=args.epochs, 
        learning_rate=args.learning_rate
    )