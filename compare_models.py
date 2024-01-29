"""Simple scipt to visually compare the performance of the 2 models on some examples in the datasets.1"""

import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
from glob import glob
import random as rd

from custom_unet import CustomUnet
from unet import Unet

if __name__ == '__main__':
    
    k = 2
    image_size = 320, 320
    
    example_list = rd.sample(glob('Human-Segmentation-Dataset-master/Training_Images/*.jpg'), k=k)
    unet, custom_unet = Unet('unet', from_file=True), CustomUnet('custom_unet', from_file=True)
    print('models loaded')
    
    original_images = np.concatenate([cv.resize(cv.cvtColor(cv.imread(example), cv.COLOR_BGR2RGB), image_size) for example in example_list], 0)
    
    get_preds = lambda model: np.concatenate([cv.resize(model.predict(image), image_size) for image in example_list], 0)
    unet_preds, custom_unet_preds = get_preds(unet), get_preds(custom_unet)

    preds = np.concatenate((original_images, unet_preds, custom_unet_preds), 1)
    
    x_ticks_locations = [image_size[1] // 2, image_size[1] * 1.5, image_size[1] * 2.5]
    x_labels = ['original', unet.name, custom_unet.name]
    plt.xticks(ticks=x_ticks_locations, labels=x_labels)

    plt.imshow(preds)
    plt.title(f'{k} examples from the dataset for {unet.name} and {custom_unet.name}')
    plt.yticks([])
    plt.savefig('models_predictions_visualized.pdf')
    plt.show()

