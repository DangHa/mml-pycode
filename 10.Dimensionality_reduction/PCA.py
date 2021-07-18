# Principal component analysis (PCA) with 2 dimensions
import numpy as np
import matplotlib.pyplot as plt
# other instruction
# http://www.math.union.edu/~jaureguj/PCA.pdf

number_of_data = 50

## generating data
def gererating_data(number_point):
    x1 = np.random.uniform(low=0.0, high=5.0, size=number_point)

    # y = a*x + b
    a = np.random.uniform(low=-2.0, high=2.0)
    b = np.random.uniform(low=-3.0, high=3.0)
    x2 = a*x1 + b - x1[::-1]*2/3 

    plt.plot(x1, x2, 'k+', label='Original data')
    return x1,x2

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

    plt.plot(x1, x2, 'b+', label='Nomalized data')
    return x1, x2

## Step 3: Eigendecomposition of the covariance matrix
# compute the principal components (B) 
# S = (1/N)*X*Xt - covariance matrix
def eigendecomposition_of_S(X):
    S = 1/(X.shape[1])*X*X.T

    λ, B = np.linalg.eig(S)

    # draw 2 eigenvectors*eigenvalues
    b1λ1 = B[:,0]*λ[0]
    b2λ2 = B[:,1]*λ[1]
    
    x_origin = [0,0]    # origin point
    y_origin = [0,0]    # origin point

    # [b1λ1[0], b2λ2[0]] = x of b1 and b2; [b1λ1[1], b2λ2[1]] = y of b1 and b2
    v = plt.quiver(x_origin, y_origin, [b1λ1[0,0], b2λ2[0,0]], [b1λ1[1,0], b2λ2[1,0]], color=['r'], width=0.005, scale=5)
    plt.quiverkey(v, .08, .23, .21, 'Eigenvectors', color='r', labelpos='E')
    
    return λ, B

## Step 4: Projection 
def projection(B, X):
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

# Preparing data
x1, x2 = gererating_data(number_of_data)

# Step 1 + Step 2: Nomalization
x1, x2 = nomalization(x1, x2)
X = np.matrix([x1, x2])

# Step 3: Eigendecomposition of the covariance matrix
λ, B = eigendecomposition_of_S(X)

# Step 4: Projection
b_m, Z, X_projection = projection(B, X)

# Average squared reconstruction error
error_sum = np.power(X-X_projection, 2).sum()
print("Average squared reconstruction error: " + str(error_sum))

# Drawing
plt.axis('equal')
plt.grid(linestyle='--')
plt.title("The number of points = " + str(number_of_data), loc = 'left', color = "blue")

plt.legend(loc="lower left")
plt.show()