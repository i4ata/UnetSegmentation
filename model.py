"""This module contains the base class for segmentation models"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import resize
from torch.utils.tensorboard.writer import SummaryWriter

import numpy as np
import cv2 as cv
import albumentations as A

from typing import Optional, Union, Tuple, Literal

from early_stopper import EarlyStopper

class SegmentationModel(nn.Module):

    name: str = "base name"
    device: Literal['cpu', 'cuda'] = None

    optimizer: torch.optim.Optimizer = None
    early_stopper: EarlyStopper = None
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None
    save_path: str = None
    image_size: Tuple[int, int] = None

    def configure_optimizers(self, **kwargs) -> None:
        raise NotImplementedError()

    def forward(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        raise NotImplementedError()

    def _train_step(self, data_loader: DataLoader) -> float:

        self.train()
        total_loss = 0.
        for images, masks in data_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            logits, loss = self(images, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(data_loader)

    def _test_step(self, data_loader: DataLoader) -> float:

        self.eval()
        total_loss = 0.
        with torch.inference_mode():
            for images, masks in data_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                logits, loss = self(images, masks)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def train_model(self, train_loader: DataLoader, test_loader: DataLoader, epochs: int, log_dir: str) -> None:
        
        writer = SummaryWriter(log_dir=f'{log_dir}/{self.name}')
        
        for i in range(epochs):
            train_loss = self._train_step(train_loader)
            test_loss = self._test_step(test_loader)
            
            if self.early_stopper is not None:
                if self.early_stopper.check(test_loss):
                    print(f'Model stopped early due to risk of overfitting')
                    break

                if self.early_stopper.save_model:
                    self.save()
                    print('saved model')

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                
            print(f'{i}: Train loss: {train_loss :.2} | Test loss: {test_loss :.2}')
        
            writer.add_scalars(main_tag='Loss over time',
                               tag_scalar_dict={'train loss': train_loss, 'test loss': test_loss},
                               global_step=i)
                                
        else:
            if self.early_stopper is not None:
                print('Model did not converge. Possibility of underfitting')
            self.save()
        writer.close()

    def save(self) -> None:
        raise NotImplementedError()

    def predict(self, 
                test_image_path: str, 
                option: Literal['mask', 'image_with_mask', 'mask_and_image_with_mask'] = 'image_with_mask'
        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        
        self.eval()
        input_resizer = A.Resize(*self.image_size)
        
        original_image = cv.cvtColor(cv.imread(test_image_path), cv.COLOR_BGR2RGB)
        original_image_tensor = torch.from_numpy(original_image).permute(2,0,1).type(torch.uint8)
        resized_image_tensor = (torch.from_numpy(input_resizer(image=original_image)['image']).float() / 255.).permute(2,0,1)

        with torch.inference_mode():
            logits = self(resized_image_tensor.unsqueeze(0).to(self.device)).squeeze(0).cpu().detach()
        probs = torch.sigmoid(logits)
        resized_mask_tensor = probs > .5
        
        original_mask_tensor = resize(resized_mask_tensor, size=original_image.shape[:-1], antialias=True)
        
        image_with_mask = draw_segmentation_masks(image=original_image_tensor,
                                                  masks=original_mask_tensor,
                                                  alpha=.5,
                                                  colors='white')

        if option == 'mask':
            return resized_mask_tensor.numpy()
        if option == 'image_with_mask':
            return image_with_mask.permute(1,2,0).numpy()
        if option == 'mask_and_image_with_mask':
            return resized_mask_tensor.numpy(), image_with_mask.permute(1,2,0).numpy()
    
    def print_summary(self) -> None:
        raise NotImplementedError()