# generate and nomalize data
import numpy as np
import matplotlib.pyplot as plt

### generating data
def gererating_data(number_point):
    x1 = np.random.uniform(low=0.0, high=5.0, size=number_point)

    # y = a*x + b
    a = np.random.uniform(low=-2.0, high=2.0)
    b = np.random.uniform(low=-3.0, high=3.0)
    x2 = a*x1 + b - x1[::-1]*2/3 

    plt.plot(x1, x2, 'k+', label='Original data')
    return x1,x2

### Step 1: Mean subtraction
### Step 2: Standardization
#### S1 + S2 = Data Nomalization (Normalizing Inputs)
def nomalization(x1, x2):
    mean_x1 = np.mean(x1)
    std_x1 = np.std(x1)

    mean_x2 = np.mean(x2)
    std_x2 = np.std(x2)

    x1 = (x1 - mean_x1)/(std_x1)
    x2 = (x2 - mean_x2)/(std_x2)

    plt.plot(x1, x2, 'b+', label='Nomalized data')
    return x1, x2