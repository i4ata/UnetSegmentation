"""Train the models"""

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
    """
    The loss for the models: BCE + Dice
    BCE is used since we are doing binary classification
    Dice is 2x intersection / union for the predicted and ground truth segmentations
    """

    def __init__(self) -> None:
        """Initialize the 2 loss functions"""

        super(UnetLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(mode='binary')

    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss based on raw model outputs and ground truths
        
        :param logits (torch.Tensor): 4d (B1HW) f32 raw model outputs. Larger values indicate higher confidence of class 1 (i.e. that this pixel corresponds to a person)
        :param masks (torch.Tensor): 4d (B1HW) long ground truth masks (1 if person, else, 0)

        :return loss (torch.Tensor): A single scalar representing the total loss
        """
        
        return self.bce_loss(logits, masks) + self.dice_loss(logits, masks)

class Trainer:
    """Main functionality for training the models"""

    def __init__(self, model: nn.Module, name: str, random_state: Optional[int]) -> None:
        """
        Set the model and send it to the available device
        
        :param model (torch.nn.Module): The model to train
        :param name (str): The model name. Weights are saved to models/<name>.pt
        :param random_state (int|None): Seed for reproducibility
        """

        # Use GPU if it's available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = model.to(self.device)
        self.name = name

        if random_state:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)

    def _train_step(self) -> Tuple[float, float]:
        """
        Perform one sweep through the train dataset and train the model

        :return loss (float): Mean loss for the entire training dataset
        :return IoU (float): Mean IoU between the predicted and ground truth masks
        """

        # Set accumulators
        train_loss, train_iou = 0, 0

        # Set to train mode
        self.model.train()

        # Loop over the entire training dataset in batches
        for images, masks in self.train_dataloader:
            
            # Send the data to the current device
            images, masks = images.to(self.device), masks.to(self.device)
            
            # Forward pass
            predictions: torch.Tensor = self.model(images)
            
            # Calculate the loss
            loss: torch.Tensor = self.loss_fn(predictions, masks)
            
            # Reset the state of the optimizer
            self.optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Gradient descent
            self.optimizer.step()

            # Record metrics. For IoU simply take the sigmoid and then threshold at 0.5
            train_loss += loss
            train_iou += mean_iou(predictions.sigmoid().round().long(), masks.long(), num_classes=1).mean()
        
        return train_loss / len(self.train_dataloader), train_iou / len(self.train_dataloader)

    def _val_step(self) -> Tuple[float, float]:
        """
        Perform one sweep through the train dataset and only record the metrics
        
        :return loss (float): Mean loss for the entire validation dataset
        :return IoU (float): Mean IoU between the predicted and ground truth masks
        """        

        # Set accumulators
        val_loss, val_iou = 0, 0
        
        # Set to evaluation mode
        self.model.eval()

        # Loop over the entire validation dataset in batches
        for images, masks in self.val_dataloader:

            # Set the data to the current device
            images, masks = images.to(self.device), masks.to(self.device)
            
            # Forward pass without keeping track of the gradients
            with torch.inference_mode():
                predictions: torch.Tensor = self.model(images)
            
            # Calculate the loss
            loss = self.loss_fn(predictions, masks)
            
            # Record metrics. For IoU simply take the sigmoid and then threshold at 0.5
            val_loss += loss
            val_iou += mean_iou(predictions.sigmoid().round().long(), masks.long(), num_classes=1).mean()

        return val_loss / len(self.val_dataloader), val_iou / len(self.val_dataloader)

    def fit(self, train_size: float, batch_size: int, epochs: int, learning_rate: float, patience: int) -> None:
        """
        Train the model. Log metrics to tensorboard
        
        :param train_size (float): Proportion of the dataset to keep for training
        :param batch_size (int): The number of samples in one batch
        :param epochs (int): Maximum number of training epochs
        :param learning_rate (float): Optimizer learning rate
        :param patience (int): For early stopping. If the validation loss hasn't improved in that many consecutive epochs, stop the training early
        """

        # Log to `runs/<name>`
        writer = SummaryWriter(log_dir=os.path.join('runs', self.name))
        
        # Get the dataloaders
        dataset = SegmentationDataset()
        self.train_dataloader, self.val_dataloader = dataset.get_dataloaders(batch_size, train_size)
        
        # Use Adam optimizer
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)
        
        early_stopper = EarlyStopper(patience)
        self.loss_fn = UnetLoss()

        # Where to save the model weights
        models_path = 'models'
        os.makedirs(models_path, exist_ok=True)
        save_path = os.path.join(models_path, self.name + '.pt')

        # Train!
        for epoch in tqdm(range(epochs)):
            
            # Training loop
            train_loss, train_iou = self._train_step()
            
            # Validation loop
            val_loss, val_iou = self._val_step()
            
            # Check for early stopping
            if early_stopper.check(val_loss):
                print('Training stops early due to overfitting suspicion')
                break
            # Save the model if it is the best one yet according to its performance on the validation data
            if early_stopper.save_model: torch.save(self.model.state_dict(), save_path)
            
            # Log metrics to tensorboard
            writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalars('IoU', {'train': train_iou, 'val': val_iou}, epoch)
        
        else:
            print('Model might not have converged')
        
        writer.close()

    def predict(self, image_path: str) -> torch.Tensor:
        """
        Use the model to predict the mask for one image
        
        :param image_path (str): The local path to an image file
        
        :return mask (torch.Tensor): 2D (HW) bool mask with the same shape as the image, with True representing predicted pixels with a person 
        """

        # Set to evaluation
        self.model.eval()

        # Read the image from memory
        image = np.asarray(Image.open(image_path))
        
        # Store the original dimensions to resize the mask later
        h, w = image.shape[:-1]

        # Resize to the expected model size and convert to a torch tensor
        image = torch.from_numpy(val_transform(image=image)['image']).float().permute(2,0,1) / 255.
        
        # Make a prediction, take the sigmoid, threshold at 0.5. Don't keep track of the gradients
        with torch.inference_mode():
            prediction = self.model(image.to(self.device).unsqueeze(0))[0].sigmoid().cpu() > .5

        # Resize the mask to the shape of the original image
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
