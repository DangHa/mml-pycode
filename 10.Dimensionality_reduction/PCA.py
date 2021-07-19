# Principal component analysis (PCA) with 2 dimensions
import numpy as np
import matplotlib.pyplot as plt
# other instruction
# http://www.math.union.edu/~jaureguj/PCA.pdf

import data_functions as df

number_of_data = 50

### Step 3: Finding eigenvectors of the covariance matrix (Eigendecomposition)
# compute the principal components (B) 
# S = (1/N)*X*Xt - covariance matrix
def eigendecomposition_of_S(X):
    S = 1/(X.shape[1])*X*X.T

    λ, B = np.linalg.eig(S)

    # draw 2 eigenvectors*eigenvalues
    b1λ1 = B[:,0]*λ[0]
    b2λ2 = B[:,1]*λ[1]
    
    origin = [0,0]    # origin point
    # [b1λ1[0], b2λ2[0]] = x of b1 and b2; [b1λ1[1], b2λ2[1]] = y of b1 and b2
    v = plt.quiver(origin, origin, [b1λ1[0,0], b2λ2[0,0]], [b1λ1[1,0], b2λ2[1,0]], color=['r'], width=0.005, scale=5)
    plt.quiverkey(v, .08, .23, .21, 'Eigenvectors', color='r', labelpos='E')
    
    return λ, B

# Using SVD
# Eigenvectors of S is the U of X
def SVD(X):
    U, S, V = np.linalg.svd(X)

    λ = np.power(S, 2)/X.shape[1]

    # draw 2 eigenvectors*eigenvalues
    b1λ1 = U[:,0]*λ[0]
    b2λ2 = U[:,1]*λ[1]

    origin = [0,0]    # origin point
    # [b1λ1[0], b2λ2[0]] = x of b1 and b2; [b1λ1[1], b2λ2[1]] = y of b1 and b2
    v = plt.quiver(origin, origin, [b1λ1[0,0], b2λ2[0,0]], [b1λ1[1,0], b2λ2[1,0]], color=['r'], width=0.005, scale=5)
    plt.quiverkey(v, .08, .23, .21, 'Eigenvectors', color='r', labelpos='E')

    return λ, U


### Step 4: Projection 
def projection(B, λ, X):
    # get the highest λ with its respective eigenvector b
    b_m = B[:,0]
    if λ[0] < λ[1]:
        b_m = B[:,1]

    # draw the line for projection
    plt.axline((0, 0), (b_m[0,0], b_m[1,0]), color='pink', linestyle='--')

    # Z = Bt * X - Coordinate on the basis of b_m
    Z = b_m.T*X

    # projection of X with coordinates Z 
    X_projection = b_m*Z

    plt.plot(np.asarray(X_projection)[0,:], np.asarray(X_projection)[1,:], 'g.',  label='Projection points')
    return b_m, Z, X_projection


############ Run ############
def main():
    # Preparing data
    x1, x2 = df.gererating_data(number_of_data)

    # Step 1 + Step 2: Nomalization
    x1, x2 = df.nomalization(x1, x2)
    X = np.matrix([x1, x2])
    N = X.shape[0]*X.shape[1]

    # Step 3: Eigendecomposition of the covariance matrix
    # λ, B = eigendecomposition_of_S(X)
    λ, B = SVD(X) # reduce time to calculate S - covariance matrix

    # Step 4: Projection
    b_m, Z, X_projection = projection(B, λ, X)

    # Average squared reconstruction error
    error_sum = np.power(X-X_projection, 2).sum()/N
    print("Average squared reconstruction error: " + str(error_sum))

    ### Drawing
    plt.axis('equal')
    plt.grid(linestyle='--')
    plt.title("The number of points = " + str(number_of_data), loc = 'left', color = "blue")

    plt.legend(loc="lower left")
    plt.show()

main()