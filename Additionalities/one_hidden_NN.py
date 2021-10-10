import numpy as np
import math
from matplotlib import pyplot as plt

import torch
import torchvision 
import torchvision.datasets as datasets

from torchvision import datasets, transforms

img_of_number = 7
batch_size = 1000 # need to be > 4 coz we show 4 images below
PCs = 100

### Step 1: Load and normalize the MNIST training and test datasets using torchvision

# Those images are described by 28x28 dimensions => normalizing with mean = 0.5 and std = 0.5
# image = (image - mean) / std
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Traning dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# get only number img_of_number
idx = mnist_trainset.targets == img_of_number
mnist_trainset.targets = mnist_trainset.targets[idx]
mnist_trainset.data = mnist_trainset.data[idx]

trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size = batch_size) 

# Testing dataset
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# get only number img_of_number
idx = mnist_testset.targets == img_of_number
mnist_testset.targets = mnist_testset.targets[idx]
mnist_testset.data = mnist_testset.data[idx]

testloader = torch.utils.data.DataLoader(mnist_testset, batch_size = batch_size) # 16 images / 1 tensor

classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
            '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

## get 1 batch of data
dataiter = iter(trainloader)
images, labels = dataiter.next()

## check
plt.imshow(images[0].reshape(28,28))
plt.show()

## change images 28x28 to vectors 784 dimenstions
def img_to_vector(images):
    X = []
    for img in images:
        X.append(img.numpy().reshape(784,))

    X = np.asmatrix(X).T
    return X


def sigmoid(Z):
  return 1 / (1 + math.exp(-Z))

def layer_forward(X, W):

    Z = W.T*X 
    out = sigmoid(Z) # the output 

    cache = (X, W, Z, out) # Values we need to compute gradients

    return out, cache


def layer_backward(dout, cache):
    # """
    # Receive dout (derivative of loss with respect to outputs) and cache,
    # and compute derivative with respect to inputs.
    # """
    # Unpack cache values
    X, W, Z, out = cache

    # Use values in cache to compute derivatives
    dx = 1 # Derivative of loss with respect to x
    dw = 1 # Derivative of loss with respect to w

    return dx, dw