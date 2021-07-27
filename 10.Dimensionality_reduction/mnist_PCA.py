# Principal component analysis (PCA)
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision 
import torchvision.datasets as datasets

from torchvision import datasets, transforms

img_of_number = 8
batch_size = 1000 # need to be > 4 coz we show 4 images below
PCs = 100

### Step 1 + 2: Normalization
# Those images are described by 28x28 dimensions => normalizing with mean = 0.5 and std = 0.5
# image = (image - mean) / std
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

## Using MNIST dataset - the images of the hand-writing numbers from 0->9
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

## get 1 batch of data
dataiter = iter(trainloader)
images, labels = dataiter.next()

# axarr[k].imshow(images[k].numpy().squeeze())

## change images 28x28 to vectors 784 dimenstions
def img_to_vector(images):
    X = []
    for img in images:
        X.append(img.numpy().reshape(784,))

    X = np.asmatrix(X).T
    return X


### Finding eigenvectors of the covariance matrix (Using SVD)
# Eigenvectors of S is the U of X
def SVD(X):
    U, s, V = np.linalg.svd(X)

    λ = s*s.T/X.shape[1]
    
    ## get only the eigenvectors have eigenvalues > 0 
    B = U[:,0:λ.shape[0]]
    
    return λ, B


### Projection 
def projection(B, λ, X):
    # get only the eigenvectors have the highest eigenvalues
    B = B[:,0:PCs]

    Z = B.T*X

    X_hat = B*Z

    return X_hat


############ Run ############
X = img_to_vector(images)
λ, B = SVD(X)
X_hat = projection(B, λ, X)


f, axarr = plt.subplots(1,4) 

for k in range(4):
    axarr[k].imshow(X_hat[:,k].reshape(28,28))
    
    # remove x,y axises
    axarr[k].axes.get_yaxis().set_visible(False)
    axarr[k].axes.get_xaxis().set_visible(False)


plt.title("PCs = " + str(PCs), loc = 'left', color = "blue")
plt.show()