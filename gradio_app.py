"""Launch a gradio application"""

import gradio as gr
import random as rd
from glob import glob

from unet import Unet

if __name__ == '__main__':

    model = Unet(name='unet')
    model.load()

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