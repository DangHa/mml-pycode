# Bayesian linear regression
# follow the instruction, 3.3 Bayesian linear regression - Bishop 2006
import matplotlib.pyplot as plt
import numpy as np

## Data
X = np.matrix([1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]).T
y = np.matrix([100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]).T

def bayesian_linear_regression():

    K = 5 # degree of polynomial tegression function
    ############ Polynomial Regression ############
    # Φ = [[1 x ... x^k] [1 x ... x^k]2 ... [1 x ... x^k]n]
    # k is function degree of y ; n is the number of x
    def phi(X0):
        Φ = np.ones(X0.shape)

        # already have 2 collumns [1], need + [x x^2 ... x^k]
        for k in range(1, K+1):
            x_k = np.power(X0, k)
            Φ = np.concatenate((Φ, x_k), axis = 1)
            
        return Φ

    Φ = phi(X)

    ### Bayesian Linear Regression
    # p(θ)      ~ N(θ| 0, alpha^-1*I)        -- Prior
    # p(y|x,θ)  ~ N(y| Φ*θ, sigma^2*I)       -- Likelihood
    # we assume that we have the uncertainties of prior and likelihood 
    alpha = 1.0 # assume
    sigma = 3.0 # assume

    beta = sigma*sigma  # sigma^2

    ## Posterior distribution
    # p(θ|x,y)  ~ N(θ | m_n, S_n)
    # S_n = (alpha^-1*I + sigma^-2*Φ.T*Φ).I
    # m_n = sigma^-2 * S_n * Φ.T * y
    S_n = np.linalg.inv(alpha*np.identity(Φ.shape[1]) + beta*Φ.T*Φ)
    m_n = beta * S_n* Φ.T * y


    ## Draw the graph
    xlist = np.asmatrix(np.arange(X[0], X[-1], 0.3)).T

    # cause p(θ) ~ N(θ| 0, alpha^-1*I)
    # => w_MAP = m_n (the mean line of bayesian_lr)
    Y_bayesian_lr_map = phi(xlist)*m_n

    # get 5 more random other θ in Gaussian distribution p(y|x,θ)
    θ_other = np.random.multivariate_normal(np.asarray(m_n).ravel(), S_n, 20).T
    Y_bayesian_lr = phi(xlist)*θ_other

    ## Graph
    plt.plot(xlist, Y_bayesian_lr, 'r-', label="")
    plt.plot(xlist, Y_bayesian_lr_map, 'b-', label="MAP (m_n)")
    plt.plot(X, y, 'k^') # => plot the data

    title = "M = " + str(K) + ", alpha = " + str(alpha) + ", sigma = " + str(sigma)
    plt.title(title, loc = 'left', color = "blue")
    plt.legend()
    plt.show()


def main():
    bayesian_linear_regression()

main()