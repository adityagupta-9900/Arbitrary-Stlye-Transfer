# Arbitrary Style Transfer
Arbitrary-Style-Per-Model Fast Neural Style Transfer Method

## Description
A Pytorch implementation of the 2017 Huang et. al. paper "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization" https://arxiv.org/abs/1703.06868 

This Encoder-AdaIN-Decoder architecture - Deep Convolutional Neural Network as a Style Transfer Network (STN) which can receive two arbitrary images as inputs (one as content, the other one as style) and output a generated image that recombines the content and spatial structure from the former and the style (color, texture) from the latter without re-training the network.

![Architecture](images/tmp/Architecture.png)


## How to run
- Download the Vgg model from [here](https://drive.google.com/file/d/1yOy1mWOa3dY-lpj8IZUIDayUnBuHKNx0/view?usp=sharing) and place it into the models folder.
- Download the pretrained model from [here](https://drive.google.com/file/d/18AtLdqyAjLD54RRIfwhcq9g80CYzrWqA/view?usp=sharing) and place it into the models folder.
- Detailed help about running can be found by `python3 main.py -h`
![Instruction](images/tmp/Instruction.png)

- For training, the dataset folder structure has to be: content/1/\*.jpg and style/1/\*.jpg