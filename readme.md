# CISC 442 PR 3

Ran out of compute time on Colab, so I had to do it locally.

![AHHHHHHH](https://miro.medium.com/v2/resize:fit:425/0*eASvpnSYKRE7UOXf.jpg)

<small>I hate CUDA</small>

## Setup

Tested on Python 3.12.3 and [PyTorch](https://pytorch.org/get-started/locally/) 2.7.0

```bash
pip install numpy pillow matplotlib torchsummary
```

Follow the instructions in the [PyTorch documentation](https://pytorch.org/get-started/locally/) to install PyTorch for your platform.

## Architecture

3 Jupyter notebooks are provided, 1 for each part of the assignment.

Each notebook is also available as a `.py` file.

They can each be run using "Run All" to produce required deliverables.

### Part 1

The input data will be downloaded to the `sample_data` directory.

Pre-trained weights are provided in [output/part-1](output/part-1).

```
Train Accuracy: 63.27%
Test Accuracy : 65.07%
Best Iteration: 29/30
```

### Part 2

My modified VGG16 model is loosely based on a model I made for [CISC484](https://github.com/mcmerdith/cisc484/tree/hw4)

The input data will be downloaded to the `sample_data` directory.

Pre-trained weights are provided in [output/part-2](output/part-2).

```
Train Accuracy: 97.35%
Test Accuracy : 55.78%
Best Iteration: 25/30
```

This model showed signs of overfitting

### Part 3

The input images are provided in the `images` directory.

The segmentation images and feature maps are provided in [output/part-3](output/part-3).
