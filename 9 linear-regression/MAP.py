# Maximum A Posteriori (MAP) estimation
import numpy as np
import matplotlib.pyplot as plt

# p(θ)      ~ N(0, b^2*I)
# p(y|x,θ)  ~ N(y| xt*θ, a^2)
# we hardly have the uncertainties of prior and likelihood (it's the problem of MAP)
a = 1
b = 4

# sample data
X = np.matrix([[1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]]).T
y = np.matrix([[100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]]).T


def map():
    
    ############ MAP
    λ = (a*a)/(b*b)

    M = 4   # k = M - 1

    Xbar = np.ones(X.shape)

    # already have 2 collumns [1], need + [x x^2 ... x^k]
    for k in range(1, M):
        X_k = np.power(X, k)
        Xbar = np.concatenate((Xbar, X_k), axis = 1)

    A = Xbar.T*Xbar
    B = Xbar.T*y
    I = np.ones(A.shape)

    ############ MAP
    W = (A + λ*I).I*B

    ############ Polynomial Regression
    W1 = A.I*B

    ## Drawning
    # MAP
    x0 = np.linspace(np.asarray(X)[0][0], np.asarray(X)[X.shape[0]-1][0], 60)
    y0 = 0 # traning line

    for k in range (M):
        w_k = np.asarray(W)[k][0]
        y0 += w_k * np.power(x0, k)

    # MLE
    x1 = np.linspace(np.asarray(X)[0][0], np.asarray(X)[X.shape[0]-1][0], 60)
    y1 = 0 # traning line

    for k in range (M):
        w_k = np.asarray(W1)[k][0]
        y1 += w_k * np.power(x1, k)

    plt.plot(X, y, 'ro')      # data 
    plt.plot(x0, y0, 'b-', label='MAP')
    plt.plot(x1, y1, 'g--', label='MLE')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("M = 3, λ = 1/16", loc = 'left', color = "blue")

    plt.legend()
    plt.show()


############ Run ############
def main():
    map()

main()