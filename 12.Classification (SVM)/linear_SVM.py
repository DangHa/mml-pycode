# Linear - Support vector machine (Linear - SVM)
# other instruction
# https://github.com/CihanBosnali/Support-Vector-Machine-without-ML-libraries/blob/master/Support_Vector_Machine_from_Scratch.ipynb

import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn import svm


number_of_point = 150

### Generating data inside a circle
def gererating_data(size):

    # First cluster, y = 1
    X11 = torch.empty(size).normal_(mean=-1,std=2).numpy().reshape(size, 1)
    X12 = torch.empty(size).normal_(mean=-1,std=1).numpy().reshape(size, 1)
    plt.plot(X11.T, X12.T, color='red', marker='+')
    
    X1 = np.hstack((X11,X12))
    Y1 = np.full((size, 1), 1)

    # Second cluster, y = -1
    X21 = torch.empty(size).normal_(mean=10,std=2).numpy().reshape(size, 1)
    X22 = torch.empty(size).normal_(mean=0,std=2).numpy().reshape(size, 1)
    plt.plot(X21.T, X22.T, color='blue', marker='_')
    
    X2 = np.hstack((X21,X22))
    Y2 = np.full((size, 1), -1)

    return X1, X2, Y1, Y2

# Data
X1, X2, Y1, Y2 = gererating_data(int(number_of_point/2))
X = np.vstack((X1, X2))
Y = np.vstack((Y1, Y2)).reshape(number_of_point)

# SVM
clf = svm.SVC(kernel='linear', random_state = 1)
clf.fit(X,Y)

## the points are the closest to the line - lie on the margin lines (support vectors)


############ Visualization ############
# get the x,y of graph
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T


# plot decision boundary and margins
# decision_function gives the distences between the points and SVM-hyperplane
Z = clf.decision_function(xy).reshape(XX.shape)

# XX, YY is coordinate of contour
# Z is coordinate of 1 points to the origin point => its a line
# with Z = x^2 + y^2 => its a circle
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# plot support vectors - the points lies on the margin
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')


plt.grid(linestyle='--')
plt.show()