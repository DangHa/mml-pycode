import numpy as np
import matplotlib.pyplot as plt

# sample data
X = np.matrix([[1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]]).T
y = np.matrix([[100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]]).T


############ Linear regression ############
# Y = W*X ; A is a number vector
# Loss_function = 1/(2*N)*(Y-W*X)
# W = (Xt*X)^(-1)*(Xt*y)

def linear_regression():
    # Building X-b - W*[X, 1] = W(1->n)*X + W0 
    # W0 = b in y = wx + b
    one = np.ones((X.shape[0], 1))
    Xbar = np.concatenate((X, one), axis = 1)

    A = Xbar.T*Xbar
    B = Xbar.T*y

    W = A.I*B
    print("w: ", W)

    # w_1 = W
    # w_0 = b
    w = np.asarray(W)[0][0]
    b = np.asarray(W)[1][0]

    # first ele in X, last ele in X, the number of middle points
    x0 = np.linspace(np.asarray(X)[0][0], np.asarray(X)[X.shape[0]-1][0], 2) 
    y0 = w*x0 + b # traning line

    plt.plot(X, y, 'ro')     # data 
    plt.plot(x0, y0, 'y--')   # the fitting line
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


############ Polynomial Regression ############
# w = [[1 x ... x^k] [1 x ... x^k]2 ... [1 x ... x^k]n]
# k is function degree of y ; n is the number of x

def polynomial_regression(M_value):
    M = M_value   # k = M -1

    one = np.ones((X.shape[0], 1))
    Xbar = np.concatenate((one, X), axis = 1)

    # already have 2 collumns [1, x]
    for k in range(2, M):
        X_k = np.power(X, k)
        Xbar = np.concatenate((Xbar, X_k), axis = 1)

    A = Xbar.T*Xbar
    B = Xbar.T*y

    W = A.I*B
    print("w: ", W)

    # w_0 is weight of 1
    # w_1 is weight of x
    # w_2 is weight of x^2
    # ....
    # w_k is weight of x^k

    x0 = np.linspace(np.asarray(X)[0][0], np.asarray(X)[X.shape[0]-1][0], 60)
    y0 = 0 # traning line

    for k in range (M):
        w_k = np.asarray(W)[k][0]
        y0 += w_k * np.power(x0, k)
    

    plt.plot(X, y, 'ro')     # data 
    plt.plot(x0, y0, 'b-')   # the fitting line
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


############ Run ############
def main():
    # linear_regression()
    polynomial_regression(4)

main()


