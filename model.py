import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.utils import draw_segmentation_masks
from torch.utils.tensorboard.writer import SummaryWriter

import numpy as np

from typing import Optional, Union, Tuple

from dataset import get_image
from early_stopper import EarlyStopper

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SegmentationModel(nn.Module):
    def __init__(self) -> None:

        super().__init__()
        self.name: str = "base name"
        self.device: str = None
        self.optimizer: torch.optim.Optimizer = None
        self.early_stopper: EarlyStopper = None
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None
        self.save_path: str = None

    def forward(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pass

    def _train_step(self, data_loader: DataLoader) -> float:
        self.train()
        total_loss = 0.
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            
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
                images, masks = images.to(device), masks.to(device)
                logits, loss = self(images, masks)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def train_model(self, train_loader: DataLoader, test_loader: DataLoader, epochs: int) -> None:
        
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

    def save(self):
        torch.save(self.state_dict(), self.save_path)

    def load(self) -> None:
        self.load_state_dict(torch.load(self.save_path, map_location=device))
        self.eval()

    def predict(self, test_image_path: str) -> np.ndarray:
        self.eval()
        data = get_image(img_path=test_image_path, transform='test')
        with torch.inference_mode():
            logits = self(torch.unsqueeze(data['transformed']['image'], 0).to(device)).squeeze().cpu().detach()
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

        return image_with_mask.permute(1,2,0).numpy()
    
    def print_summary(self) -> None:
        pass
