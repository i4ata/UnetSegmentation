import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms.functional import resize
import torchvision.transforms as transforms
from segmentation_models_pytorch.losses import DiceLoss
from torchmetrics.functional.segmentation import mean_iou
from typing import Tuple, Optional
from PIL import Image
import numpy as np
import os
import yaml
from tqdm import tqdm

from src.utils import val_transform, get_pretrained_unet
from src.dataset import SegmentationDataset
from src.early_stopper import EarlyStopper
from src.custom_unet import CustomUnet

class UnetLoss(nn.Module):
    """The loss for the model: BCE + Dice"""

    def __init__(self) -> None:
        super(UnetLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(mode='binary')

    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return self.bce_loss(logits, masks) + self.dice_loss(logits, masks)

class Trainer:
    def __init__(self, model: nn.Module, name: str, random_state: Optional[int]) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.name = name

        if random_state:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)

    def _train_step(self) -> Tuple[float, float]:
        """Perform one sweep through the train dataloader
        Return the average loss and IOU
        """

        train_loss, train_iou = 0, 0
        self.model.train()
        for images, masks in self.train_dataloader:
            images, masks = images.to(self.device), masks.to(self.device)
            predictions: torch.Tensor = self.model(images)
            loss: torch.Tensor = self.loss_fn(predictions, masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss
            train_iou += mean_iou(predictions.sigmoid().round().long(), masks.long(), num_classes=1).mean()
        return train_loss / len(self.train_dataloader), train_iou / len(self.train_dataloader)

    def _val_step(self) -> Tuple[float, float]:
        """Perform one sweep through the validation dataloader
        Return the average loss and IOU
        """        

        val_loss, val_iou = 0, 0
        self.model.eval()
        for images, masks in self.val_dataloader:
            images, masks = images.to(self.device), masks.to(self.device)
            with torch.inference_mode():
                predictions: torch.Tensor = self.model(images)
                loss = self.loss_fn(predictions, masks)
            val_loss += loss
            val_iou += mean_iou(predictions.sigmoid().round().long(), masks.long(), num_classes=1).mean()
        return val_loss / len(self.val_dataloader), val_iou / len(self.val_dataloader)

    def fit(self, train_size: float, batch_size: int, epochs: int, learning_rate: float, patience: int) -> None:
        """Train the model. Log metrics to tensorboard"""

        writer = SummaryWriter(log_dir=os.path.join('runs', self.name))
        dataset = SegmentationDataset()
        self.train_dataloader, self.val_dataloader = dataset.get_dataloaders(batch_size, train_size)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)
        early_stopper = EarlyStopper(patience)
        self.loss_fn = UnetLoss()

        models_path = 'models'
        os.makedirs(models_path, exist_ok=True)
        save_path = os.path.join(models_path, self.name + '.pt')

        for epoch in tqdm(range(epochs)):
            train_loss, train_iou = self._train_step()
            val_loss, val_iou = self._val_step()
            print(f'{epoch} | Train loss: {train_loss} | Train IoU: {train_iou} | Val loss: {val_loss} | Val IoU: {val_iou}', flush=True)
            
            if early_stopper.check(val_loss):
                print('Training stops early due to overfitting suspicion')
                break
            if early_stopper.save_model: torch.save(self.model.state_dict(), save_path)
            writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalars('IoU', {'train': train_iou, 'val': val_iou}, epoch)
        else:
            print('Model might not have converged')
        writer.close()

    def predict(self, image_path: str) -> torch.Tensor:
        """
        Use the model to predict the mask for one image.
        Returns a binary numpy array
        """

        self.model.eval()
        image = np.asarray(Image.open(image_path))
        h, w = image.shape[:-1]
        image = torch.from_numpy(val_transform(image=image)['image']).float().permute(2,0,1) / 255.
        with torch.inference_mode():
            prediction = self.model(image.to(self.device).unsqueeze(0))[0].sigmoid().cpu() > .5
        mask = resize(img=prediction, size=(h,w), interpolation=transforms.InterpolationMode.NEAREST)[0]
        return mask

if __name__ == '__main__':

    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    # CUSTOM MODEL
    model_params = params['models']['custom']
    model = CustomUnet(in_channels=3, depth=model_params['depth'], start_channels=model_params['start_channels'])
    trainer = Trainer(model=model, name=model_params['name'], random_state=params['seed'])
    trainer.fit(**params['train'])

    # PRETRAINED MODEL
    model_params = params['models']['pretrained']
    model = get_pretrained_unet()
    trainer = Trainer(model=model, name=model_params['name'], random_state=params['seed'])
    trainer.fit(**params['train'])
