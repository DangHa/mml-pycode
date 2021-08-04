# Expectation maximization (EM) algorithm
# GMM - Gaussian Mixture Models
# other instruction
# https://angusturner.github.io/generative_models/2017/11/03/pytorch-gaussian-mixture-model.html

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.distributions import Normal 

number_of_point = 100
K = 3

### generating data inside a circle
def gererating_data(size):
    # First cluster
    X11 = torch.empty(size).normal_(mean=5,std=0.5).numpy()
    X12 = torch.empty(size).normal_(mean=2,std=0.5).numpy()

    # Second cluster
    X21 = torch.empty(size).normal_(mean=-1,std=0.5).numpy()
    X22 = torch.empty(size).normal_(mean=-1,std=0.5).numpy()

    # Third cluster
    X31 = torch.empty(size).normal_(mean=-1,std=1).numpy()
    X32 = torch.empty(size).normal_(mean=1.5,std=1).numpy()

    X1 = np.concatenate((X11, X21, X31)).reshape(1, size*3)
    X2 = np.concatenate((X12, X22, X32)).reshape(1, size*3)

    return X1,X2


X1,X2 = gererating_data(number_of_point)

plt.plot(X1, X2, color='dimgrey', marker='o')

### Initialize u_k , Σ_k , π_k .
u = np.random.rand(K, 2)
Σ = np.random.rand(K, 2)
π = np.random.rand(1, K)

## calculate responsbilities r
# Σ = I*σ^2 
X11 = np.matrix([X1[0,0], X2[0,0]])

log_r11 = 0
for i in range(K):
    u_i = np.asmatrix(u[i,:])
    Σ_i = np.diag(Σ[i,:])
    π_i = π[0,0]

    # 11.11 formulation in mml-book
    log_1 = -0.5 * np.log(2 * np.pi)
    log_2 = -0.5 * np.log(np.linalg.det(Σ_i))
    log_3 = -0.5 * (X11 - u_i).reshape(1,2) * np.linalg.inv(Σ_i) * (X11 - u_i).reshape(2,1)

    log_r11 = log_1 + log_2 + log_3[0,0]

    print("log_r11: ",log_r11)


############ Run ############
plt.grid(linestyle='--')
plt.show()