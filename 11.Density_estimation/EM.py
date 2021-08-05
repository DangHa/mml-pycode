# Expectation maximization (EM) algorithm
# GMM - Gaussian Mixture Models
# other instruction
# https://angusturner.github.io/generative_models/2017/11/03/pytorch-gaussian-mixture-model.html

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.distributions import Normal 

number_of_point = 150
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

X1,X2 = gererating_data(int(number_of_point/3))


### Initialize u_k , Σ_k , π_k .
u = np.random.rand(K, 2)
Σ = np.array([[0.5, 0], [0, 0.5], [0.5, 0], [0, 0.5], [0.5, 0], [0, 0.5]])
π = np.random.rand(1, K)

### EM starting
loop = 6

for l in range(loop):
    ### E-step: calculate responsbilities R nxk
    R = np.empty([number_of_point, K])

    for j in range(number_of_point):
        X_j = np.matrix([X1[0,j], X2[0,j]])

        log_r11 = 0
        for i in range(K):
            u_i = np.asmatrix(u[i,:])
            Σ_i = np.asmatrix(Σ[i*2:i*2+2,:])
            π_i = π[0,0]

            # 11.11 formulation in mml-book
            log_1 = np.power(np.pi,-0.5)
            log_2 = np.power(np.linalg.det(Σ_i),-0.5)
            log_3 = np.exp(-0.5 * (X_j - u_i).reshape(1,2) * np.linalg.inv(Σ_i) * (X_j - u_i).reshape(2,1))

            log_r11 = π_i * log_1 * log_2 * log_3[0,0]

            R[j,i] = log_r11

        sum = np.sum(R[0,:])
        for i in range(K):
            R[j,i] = R[j,i]/sum

    ### M-step: recalculate mean and variance

    # mean
    X = np.asmatrix(np.hstack((X1.T, X2.T)))

    for k in range(K):
        r_nk = R[:, k].reshape(number_of_point,1)
        N_k = np.sum(r_nk)

        u_k = r_nk.T*X/N_k

        u[k,:] = u_k
    print("u: ", u)

    # covariance
    for k in range(K):
        r_nk = R[:, k].reshape(number_of_point,1)
        N_k = np.sum(r_nk)

        u_k = u[k, :].reshape(2,1)
        
        Σ_k_new = 0
        for n in range(number_of_point):
            x_n = X[n,:].T
            Σ_k_new += r_nk[n,0]*(x_n - u_k)*(x_n - u_k).T
        
        Σ_k_new /= N_k

        Σ[k*2:k*2+2,:] = Σ_k_new
    print("Σ: ", Σ)

    # weight
    for k in range(K):
        r_nk = R[:, k].reshape(number_of_point,1)
        N_k = np.sum(r_nk)

        π_k_new = N_k/number_of_point

        π[0, k] = π_k_new
    print("π: ", π)

############ Visualization ############

plt.plot(X1, X2, color='blue', marker='+')
plt.plot(u[:,0], u[:,1], 'ro')
plt.grid(linestyle='--')
plt.show()