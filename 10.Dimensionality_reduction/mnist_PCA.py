# Principal component analysis (PCA)
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision 
import torchvision.datasets as datasets

from torchvision import datasets, transforms

img_of_number = 2

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
trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size = 3) 

# Testing dataset
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# get only number img_of_number
idx = mnist_testset.targets == img_of_number
mnist_testset.targets = mnist_testset.targets[idx]
mnist_testset.data = mnist_testset.data[idx]
testloader = torch.utils.data.DataLoader(mnist_testset, batch_size = 16) # 16 images / 1 tensor

# get 1 batch of data
dataiter = iter(trainloader)
images, labels = dataiter.next()

# plt.imshow(images[0].numpy().squeeze())
    
## change images 28x28 to vectors 784 dimenstions
X = []
for img in images:
    X.append(img.numpy().reshape(784,))

X = np.asmatrix(X).T
print("X:", X.shape)


### Finding eigenvectors of the covariance matrix (Using SVD)
# Eigenvectors of S is the U of X
def SVD(X):
    U, s, V = np.linalg.svd(X)

    λ = s*s.T/X.shape[1]
    
    ## get only the eigenvectors have eigenvalues > 0 
    B = U[:,0:λ.shape[0]]
    
    return λ, B

λ, B = SVD(X)

### Projection 
def projection(B, λ, X):
    
    Z = B.T*X

    X_hat = B*Z

    print("X_har: ", X_hat.shape)
    print(X_hat[:,0].reshape(28,28).shape)
    plt.imshow(X_hat[:,0].reshape(28,28))


projection(B, λ, X)
plt.show()