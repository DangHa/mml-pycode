import numpy as np
import matplotlib.pyplot as plt

X = np.matrix([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.matrix([[ 45, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

############ Linear regression
# Y = W*X ; A is a number vector
# Loss_function = 1/(2*N)*(Y-W*X)
# W = (Xt*X)^(-1)*(Xt*y)


# Building X-b - W*[X, 1] = W(1->n)*X + W0 
# W0 = b in y = wx + b
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((X, one), axis = 1)

A = Xbar.T*Xbar
B = Xbar.T*y

W = A.I*B
print("w: ", W)

# W1 = W
# W0 = b
w = np.asarray(W)[0][0]
b = np.asarray(W)[1][0]

x0 = np.linspace(145, 185, 2)
y0 = w*x0 + b # traning line


plt.plot(X, y, 'ro')     # data 
plt.plot(x0, y0, 'y--')         # the fitting line
plt.xlabel('x')
plt.ylabel('y')

plt.show()


############ Polynomial Regression
#

