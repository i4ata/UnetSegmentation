"""Launch a gradio application"""

import gradio as gr
import random as rd
from glob import glob
import argparse

from custom_unet import CustomUnet
from unet import Unet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_custom', type=bool, default=False, help='Whether to use the custom or pretrained implementation')
    parser.add_argument('--model_name', type=str, default='unet', help='Model instance to load')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    model_class = CustomUnet if args.use_custom else Unet

    model = model_class(args.model_name, from_file=True)

    # Get 3 random examples from the dataset
    example_list = rd.sample(glob('Human-Segmentation-Dataset-master/Training_Images/*.jpg'), k=3)

    demo = gr.Interface(
        fn=model.predict,
        inputs=gr.Image(type='filepath'),
        outputs=gr.Image(type='numpy'),
        title='Penis title',
        description='Penis description',
        article='Penis article',
        examples=example_list
    )

    demo.launch(share=True)