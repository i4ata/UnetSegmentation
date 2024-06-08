"""Simple scipt to visually compare the performance of the 2 models on some examples in the datasets.1"""

import torch
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
from glob import glob
import random as rd
from torchvision.io import read_image

from src.train import Trainer
from src.custom_unet import CustomUnet
from src.utils import get_pretrained_unet

if __name__ == '__main__':
    
    rd.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    example_list = rd.sample(glob('Human-Segmentation-Dataset-master/Training_Images/*.jpg'), k=2)

    custom_unet = CustomUnet()
    custom_unet.load_state_dict(torch.load('models/custom_unet.pt', map_location=device))
    custom_trainer = Trainer(model=custom_unet, device=device)
    custom_unet_preds = [custom_trainer.predict(example) for example in example_list]

    pretrained_unet = get_pretrained_unet()
    pretrained_unet.load_state_dict(torch.load('models/pretrained_unet.pt', map_location=device))
    pretrained_trainer = Trainer(model=pretrained_unet, device=device)
    pretrained_unet_preds = [pretrained_trainer.predict(example) for example in example_list]

    images = list(map(read_image, example_list))

    overlayed_masks_custom = [
        draw_segmentation_masks(
            image=image,
            masks=torch.from_numpy(preds).unsqueeze(0).bool(),
            colors='red',
            alpha=.8
        ).permute(1,2,0)
        for image, preds in zip(images, custom_unet_preds)
    ]
    overlayed_masks_pretrained = [
        draw_segmentation_masks(
            image=image,
            masks=torch.from_numpy(preds).unsqueeze(0).bool(),
            colors='red',
            alpha=.8
        ).permute(1,2,0)
        for image, preds in zip(images, pretrained_unet_preds)
    ]

    fig, axs = plt.subplots(nrows=2, ncols=3)
    for a in axs.flatten(): a.axis('off')
    axs[0,0].imshow(images[0].permute(1,2,0))
    axs[1,0].imshow(images[1].permute(1,2,0))
    axs[0,1].imshow(overlayed_masks_custom[0])
    axs[0,2].imshow(overlayed_masks_pretrained[0])
    axs[1,1].imshow(overlayed_masks_custom[1])
    axs[1,2].imshow(overlayed_masks_pretrained[1])

    axs[0,0].set_title('Original Image')
    axs[0,1].set_title('Custom Unet')
    axs[0,2].set_title('Pretrained Unet')
    
    plt.tight_layout()
    plt.savefig('example_predictions.png', dpi=400)
