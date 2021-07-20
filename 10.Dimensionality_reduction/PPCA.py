# Probabilistic principal component analysis (PPCA)
import numpy as np
import matplotlib.pyplot as plt
# other instruction
# http://edwardlib.org/tutorials/probabilistic-pca

number_of_data = 150

### generating data
def gererating_data(number_point):
    x1 = np.random.uniform(low=-3.0, high=3.0, size=number_point)
    x2 = np.random.uniform(low=-3.0, high=3.0, size=number_point)

    plt.plot(x1, x2, 'r+', label='Original data')
    return x1,x2


### Finding eigenvectors of the covariance matrix (SVD)
def SVD(X):
    U, S, V = np.linalg.svd(X)

    λ = np.power(S, 2)/X.shape[1]

    return λ, U


### Regeneration 
def regeneration(B, λ, X):
    # try with 10 z
    X_regenerated = np.empty((0, 2))

    for k in range(30):
        # p(z) = N(0, I)
        # choose a z from its normal distribution
        z = np.random.multivariate_normal([0, 0], np.identity(2), 1)
    
        # p(ε) = N(0, σ^2*I)
        # x = Bz + u + ε
        # => p(x|z,B,u,ε) = N(x|Bz+u, σ^2*I)        (1)
        # => p(x|B,u,ε)   = N(x|u, B*B.T+σ^2*I)     (2)
        σ2 = λ.sum() / λ.shape[0]  # assume σ^2
        u = X.mean(axis = 1)

        # calculate mean and variance of (1)
        mean_x = B*z.T + u 
        variance_x = σ2 * np.identity(B.shape[0])

        # choose a new_x from its normal distribution (pick 5)
        new_x = np.random.multivariate_normal(np.asarray(mean_x).squeeze(), variance_x, 5)

        X_regenerated = np.append(X_regenerated, np.array(new_x), axis=0)

    plt.plot(X_regenerated[:,0], X_regenerated[:,1], 'g.',  label='Simulated data')
    return X_regenerated


# Preparing data
x1, x2 = gererating_data(number_of_data)
X = np.matrix([x1, x2])
N = X.shape[0]*X.shape[1]

# Finding eigenvectors of the covariance matrix (SVD)
λ, B = SVD(X) 

#  Regeneration
X_regenerated = regeneration(B, λ, X)

plt.axis('equal')
plt.grid(linestyle='--')
plt.title("The number of points = " + str(number_of_data), loc = 'left', color = "blue")

plt.legend(loc="lower left")
plt.show()