# Bayesian linear regression
# other instruction
# http://krasserm.github.io/2019/02/23/bayesian-linear-regression/
# https://jessicastringham.net/2018/01/10/bayesian-linreg-plots/
## https://github.com/takp/bayesian-linear-regression


import matplotlib.pyplot as plt
import numpy as np

# Data
X = np.array([0.02, 0.12, 0.19, 0.27, 0.42, 0.51, 0.64, 0.84, 0.88, 0.99])
y = np.array([0.20, 0.87, 0.94, 0.92, 0.54, -0.11, -0.78, -0.79, -0.89, -0.04])

# p(θ)      ~ N(θ| m_0, S_0)        -- prior
# p(y|x,θ)  ~ N(y| X*θ, alpha^2*I)  -- likelihood
# we assume that we have the uncertainties of prior and likelihood 

# m_0 = 1
# S_0 = 3
# alpha = 4

# posterior
# p(y|x,θ)  ~ N(θ | m_n, S_n)


############ Polynomial Regression ############
# Φ = [[1 x ... x^k] [1 x ... x^k]2 ... [1 x ... x^k]n]
# k is function degree of y ; n is the number of x
def phi(X0):
    Φ = np.asmatrix(np.ones([X0.shape[0],1]))

    # already have 2 collumns [1], need + [x x^2 ... x^k]
    for k in range(1, 5):
        x_k = np.power(X0, k)
        Φ = np.concatenate((Φ, np.asmatrix(x_k).T), axis = 1)
    
    return Φ

Φ = phi(X)
print("Φ = ")
print(Φ)

# Bayesian Linear Regression
alpha = 0.1 # assume
beta = 9.0  # assume


Sigma_N = np.linalg.inv(alpha*np.identity(Φ.shape[1]) + beta * np.dot(Φ.T, Φ))
mu_N = beta * np.dot(Sigma_N, np.dot(Φ.T, y).T)
print("mu_N (Bayesian linear regression) = ")
print(mu_N)


# Draw the graph
xlist = np.arange(0, 1, 0.01)

bayesian_lr = np.dot(phi(xlist), mu_N)


plt.plot(xlist, bayesian_lr, 'b')
plt.plot(X, y, 'k^') # => plot the data
plt.show()