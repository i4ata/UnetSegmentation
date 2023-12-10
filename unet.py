import torch.nn as nn
import torch
from torchinfo import summary
import segmentation_models_pytorch as smp
from early_stopper import EarlyStopper

from model import SegmentationModel

from typing import Optional, Union, Tuple

class Unet(SegmentationModel):
    
    def __init__(self, name: str = "default_name") -> None:
        super().__init__()
        
        self.name = name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.unet = smp.Unet(
            encoder_name='timm-efficientnet-b0',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(params=self.unet.parameters(), lr=1e-3)
        self.early_stopper = EarlyStopper()
        self.save_path = f'models/{self.name}.pth'
        
        bce_loss_fn = nn.BCEWithLogitsLoss()
        dice_loss_fn = smp.losses.DiceLoss(mode='binary')
        self.loss_fn = lambda logits, masks: bce_loss_fn(logits, masks) + dice_loss_fn(logits, masks)
    
    def forward(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        logits = self.unet(images)
        if masks is None:
            return logits
        return logits, self.loss_fn(logits, masks)
    
    def print_summary(self) -> None:
        print(summary(self.unet, input_size=(16, 3, 320, 320),
                      col_names=['input_size', 'output_size', 'num_params'],
                      row_settings=['var_names']))