# UnetSegmentation

My own implementation of the U-net architecture compared to a pretrained model from PyTorch Segmentation Models. The dataset can be found [here](https://github.com/VikramShenoy97/Human-Segmentation-Dataset)

To train:

```python train.py```

To visualize training:

```tesnorboard --logdir runs```

To visually compare models on some examples:

```python compare_models.py```

To launch a Gradio application:

```python3 gradio_app.py```