"""This module contains the base class for segmentation models"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import draw_segmentation_masks
from torch.utils.tensorboard.writer import SummaryWriter

import numpy as np

from typing import Optional, Union, Tuple, Literal

from dataset import get_image
from early_stopper import EarlyStopper

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class SegmentationModel(nn.Module):
    """Base class for segmentation model
    
    Attributes:
        name: str
        device: 'cpu' or 'cuda'
        optimizer: PyTorch optimizer
        early_stopper: Optionally, enable early stopping
        lr_scheduler: Optionally, enable lr scheduling
        save_path: str - where to save the model

    Methods:
        forward: defines the forward pass
        _train_step: how a single pass through the training dataloader goes
        _test_step: how a single pass through the testing dataloader goes
        train_model: train the model
        save: save a checkpoint of the model's state dict to memory
        load: load a model's state dict from memory
        predict: perform inference on a single image
        print_summary: print the structure of the model (to be used with torchsummary)
    """

    def __init__(self) -> None:
        """Define the attributes. To be overwritten in subclasses"""
        super().__init__()
        self.name: str = "base name"
        self.device: Literal['cpu', 'cuda'] = DEVICE

        self.optimizer: torch.optim.Optimizer = None
        self.early_stopper: EarlyStopper = None
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None
        self.save_path: str = None

    def forward(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass. To be overwritten in subclasses
        
        Args:
            images: tensor (BCHW)
            masks: tensor (B1HW)

        Returns:
            logits of the model
            or
            logits with loss function
        """
        raise NotImplementedError()

    def _train_step(self, data_loader: DataLoader) -> float:
        """Standard training step. Optimize the model parameters.
        
        Args:
            data_loader: data into batches
        Returns:
            mean loss for the dataloader
        """
        self.train()
        total_loss = 0.
        for images, masks in data_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            self.optimizer.zero_grad()
            logits, loss = self(images, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(data_loader)

    def _test_step(self, data_loader: DataLoader) -> float:
        """Standard testing step. Perform inference on the test data
        
        Args:
            data_loader: data into batches
        Returns:
            mean loss for the dataloader
        """
        self.eval()
        total_loss = 0.
        with torch.inference_mode():
            for images, masks in data_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                logits, loss = self(images, masks)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def train_model(self, train_loader: DataLoader, test_loader: DataLoader, epochs: int = 10) -> None:
        """Standard training procedure. Log results to tensorboard.
        Use early stopping and learning rate scheduling if implemented
        
        Args:
            train_loader, test_loader: the PyTorch Dataloaders used for training and validation
            epochs: the number of training epochs
        """
        writer = SummaryWriter(log_dir='runs')
        
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
            print('Model did not converge. Possibility of underfitting')
            self.save()
        writer.close()

    def save(self) -> None:
        """Save the model's state dict to memory"""
        torch.save(self.state_dict(), self.save_path)

    def load(self) -> None:
        """Load the model from memory"""
        self.load_state_dict(torch.load(self.save_path, map_location=self.device))
        self.eval()

    def predict(self, 
                test_image_path: str, 
                option: Literal['mask', 'image_with_mask', 'mask_and_image_with_mask'] = 'image_with_mask'
        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict with a trained model on an image.
        
        Args:
            test_image_path: path to the image to predict
            option: whether to return the image with the overlayed mask, only the mask, or both
        Return:
            depending on the value of `option`
        """
        self.eval()
        data = get_image(img_path=test_image_path, transform='test')
        with torch.inference_mode():
            logits = self(data['transformed']['image'].unsqueeze(0).to(self.device)).squeeze().cpu().detach()
        probs = torch.sigmoid(logits)
        mask = probs > .5

        original_image: np.ndarray = data['original']['image']
        
        original_image_tensor = torch.from_numpy(original_image).permute(2,0,1).type(torch.uint8)
        resized_mask_tensor = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), 
                                            size=original_image.shape[:-1], 
                                            mode='nearest').squeeze().bool()
        
        image_with_mask = draw_segmentation_masks(image=original_image_tensor,
                                                  masks=resized_mask_tensor,
                                                  alpha=.5,
                                                  colors='white')

        if option == 'mask':
            return resized_mask_tensor.numpy()
        if option == 'image_with_mask':
            return image_with_mask.permute(1,2,0).numpy()
        if option == 'mask_and_image_with_mask':
            return resized_mask_tensor.numpy(), image_with_mask.permute(1,2,0).numpy()
    
    def print_summary(self) -> None:
        """Print the model architecture"""
        pass
