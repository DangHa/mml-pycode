# Principal component analysis (PCA)
import numpy as np
import matplotlib.pyplot as plt

## generating data
def gererating_data(number_point):
    x1 = np.random.rand(number_point)

    # y = a*x + b
    a = np.random.uniform(-2, 2)
    b = np.random.uniform(-2, 2)
    x2 = a*x1 + b - x1[::-1]/2

    plt.plot(x1, x2, 'ro')
    return x1,x2

x1,x2 = gererating_data(50)

## Step 1: Mean subtraction
## Step 2: Standardization
### S1 + S2 = Data Nomalization (Normalizing Inputs)
def nomalization(x1, x2):
    mean_x1 = np.mean(x1)
    std_x1 = np.std(x1)

    mean_x2 = np.mean(x2)
    std_x2 = np.std(x2)

    x1 = (x1 - mean_x1)/(std_x1)
    x2 = (x2 - mean_x2)/(std_x2)

    plt.plot(x1, x2, 'bo')
    return x1, x2

x1, x2 = nomalization(x1, x2)

## Step 3: Eigendecomposition of the covariance matrix
# compute the principal components (B) 

# S = (1/N)*X*Xt
X = np.matrix([x1, x2])
print(X.shape[1])
S = 1/(X.shape[1])*X*X.T

λ, B = np.linalg.eig(S)
print(B)
print(λ)
# get the highest λ with its respective eigenvector b 

b1λ1 = np.asarray(B)[:,0]*λ[0]
b2λ2 = np.asarray(B)[:,1]*λ[1]

origin = np.array([[0,0],[0,0]]) # origin point
# [b1λ1[0], b2λ2[0]] = x of b1 and b2; [b1λ1[1], b2λ2[1]] = y of b1 and b2
plt.quiver(*origin, [b1λ1[0], b2λ2[0]], [b1λ1[1], b2λ2[1]], color=['y','r'], scale=5)
plt.show()

