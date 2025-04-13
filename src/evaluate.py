"""Simple scipt to visually compare the performance of the 2 models on some examples in the datasets"""

import torch
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
from glob import glob
import random as rd
from torchvision.io import read_image
from torchvision.transforms.functional import resize
import torchvision.transforms as transforms
import yaml
import os

import matplotlib
matplotlib.use('TkAgg')

from src.train import Trainer
from src.custom_unet import CustomUnet
from src.utils import get_pretrained_unet

if __name__ == '__main__':
    
    with open('params.yaml') as f:
        params = yaml.safe_load(f)
    
    # Choose 2 random images
    rd.seed(params['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    example_images = rd.sample(glob(os.path.join('Human-Segmentation-Dataset-master', 'Training_Images', '*.jpg')), k=2)
    example_masks = [image_path.replace('Training_Images', 'Ground_Truth').replace('jpg', 'png') for image_path in example_images]

    # Load the custom model and do predictions
    model_params = params['models']['custom']
    custom_unet = CustomUnet(in_channels=3, depth=model_params['depth'], start_channels=model_params['start_channels'])
    custom_unet.load_state_dict(torch.load(os.path.join('models', model_params['name'] + '.pt'), map_location=device, weights_only=True))
    custom_trainer = Trainer(model=custom_unet, name=model_params['name'], random_state=params['seed'])
    custom_unet_preds = [custom_trainer.predict(example) for example in example_images]

    # Load the pretrained model and do predictions
    model_params = params['models']['pretrained']
    pretrained_unet = get_pretrained_unet()
    pretrained_unet.load_state_dict(torch.load(os.path.join('models', model_params['name'] + '.pt'), map_location=device, weights_only=True))
    pretrained_trainer = Trainer(model=pretrained_unet, name=model_params['name'], random_state=params['seed'])
    pretrained_unet_preds = [pretrained_trainer.predict(example) for example in example_images]

    # Read the images and masks
    images = list(map(read_image, example_images))
    masks_ground_truth = [
        resize(read_image(mask).bool(), image.shape[-2:], interpolation=transforms.InterpolationMode.NEAREST) 
        for image, mask in zip(images, example_masks)
    ]

    # Draw the masks on the images
    overlay = lambda image, mask: draw_segmentation_masks(image=image, masks=mask, colors='red', alpha=.8).permute(1,2,0)
    overlayed_masks_ground_truth = list(map(overlay, images, masks_ground_truth))
    overlayed_masks_custom = list(map(overlay, images, custom_unet_preds))
    overlayed_masks_pretrained = list(map(overlay, images, pretrained_unet_preds))

    # Plot the predictions
    fig, axs = plt.subplots(nrows=2, ncols=4)
    for a in axs.flatten(): a.axis('off')
    axs[0,0].imshow(images[0].permute(1,2,0))
    axs[1,0].imshow(images[1].permute(1,2,0))
    axs[0,1].imshow(overlayed_masks_ground_truth[0])
    axs[1,1].imshow(overlayed_masks_ground_truth[1])
    axs[0,2].imshow(overlayed_masks_custom[0])
    axs[1,2].imshow(overlayed_masks_custom[1])
    axs[0,3].imshow(overlayed_masks_pretrained[0])
    axs[1,3].imshow(overlayed_masks_pretrained[1])

    axs[0,0].set_title('Original Image')
    axs[0,1].set_title('Ground Truth')
    axs[0,2].set_title('Custom Unet')
    axs[0,3].set_title('Pretrained Unet')
    
    plt.tight_layout()
    plt.savefig('example_predictions.png', dpi=400)
