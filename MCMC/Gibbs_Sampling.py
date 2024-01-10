import matplotlib.pyplot as plt
import numpy as np
import random
import math
from matplotlib.animation import FuncAnimation
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


"""
Below are helper functions
"""
def montecarlogibbssampling(data, niter, gamma_init, n0_init):
    # storing the results for niter of the simulation
    n0vals = np.zeros(niter)
    l1vals = np.zeros(niter)
    l2vals = np.zeros(niter)
    
    # store the updated value for each iterations
    a1, b1 = gamma_init[0], gamma_init[1]
    a2, b2 = gamma_init[0], gamma_init[1]
    l1, l2 = 0, 0
    n0 = n0_init

    # simulation process
    for j in range(niter):
        l1, a1, b1 = lambdacalc(data, n0, a1, b1, True) # update l1, a1, b1 values based on previous values
        l2, a2, b2 = lambdacalc(data, n0, a2, b2, False) # update l2, a2, b2 values based on previous values
        n0 = n0calc(data, l1, l2) # update n0 using updated l1, l2 values

        # storing l1, l2, n0 values in np array format for output
        l1vals[j] = l1
        l2vals[j] = l2
        n0vals[j] = n0

    return n0vals, l1vals, l2vals

def n0calc(data, l1, l2):
    # calculates the new value of n0 using the probability function from the slides with the lambda's fixed
    N = data.size
    n0list = np.linspace(start=0, stop=N-1, num=N)
    n0prob = np.zeros(N)
    sum = 0
    for i in data:
        element = math.factorial(i)
        element = float(element)
        sum = sum + np.log(element)
    for i in range(N):
        n0 = n0list[i]
        sum1 = np.sum(data[0:i+1])
        sum2 = np.sum(data[i+1:])
        exponent = np.log(l1) * sum1 - n0 * l1 + np.log(l2)*sum2  - (N-n0) * l2 - sum
        #print("exponent: ", exponent)
        u = np.exp(exponent)
        n0prob[i] = u
    return int(random.choices(n0list, weights=n0prob)[0])


def lambdacalc(data, n0, a, b, is1):
    # calculates the new lambda value using the probability function from the slides with the other lambda and n0 fixed
    if(is1):
        # use data from 0 to n0
        a1 = a + np.sum(data[:(n0+1)])
        b1 = b + n0
        #return np.random.gamma(a1, b1), a1, b1
        return np.random.gamma(a1,1/b1), a1, b1
    else:
        # use data from n0 + 1 to N
        a2 = a + np.sum(data[(n0+1):])
        b2 = b + (data.size -  n0)
        return np.random.gamma(a2,1/b2), a2, b2

"""
Execution function that utilize the above helper functions
"""
def execution(iterations, n0_init, gamma_init, file_name):

    # reading initial gravitational wave detection files
    df = pd.read_csv(file_name, header=None)
    years = df.iloc[:,0].to_numpy()
    gravData = df.iloc[:,1].to_numpy()
    # raw data plotting
    plt.scatter(years, gravData, s=30, facecolors='none', edgecolors='b', marker = "o")
    plt.title('Gravitational Wave Detection Record')
    plt.xlabel('Year')
    plt.ylabel('recorded gravitational wave events')
    plt.savefig("Raw Data.png")

    # MCMC Gibbs sampling to estimate n0, lambda1, and lambda2
    n0, lambda1, lambda2 = montecarlogibbssampling(gravData, iterations, gamma_init, n0_init)

    # plotting n0 distribution from MCMC Gibbs sampling
    plt.hist(2017+n0, range=(2017, 2200), density = True, bins=184, edgecolor='dimgrey', linewidth=1.2, color='indianred')
    plt.xlim([2060, 2180])
    plt.ylim([0, 0.25])
    plt.title('$n_0$ probability distribution')
    plt.xlabel('n')
    plt.ylabel('P($n_0|x_{1:N}$)',rotation=0)
    plt.savefig("n_0 histogram.png")

    # plotting lambda1 and lambda2 from MCMC Gibbs sampling
    plt.scatter(lambda1, lambda2, s=2, c='indianred', marker=".")
    plt.xlim([10.4, 10.6])
    plt.ylim([17.8, 18.4])
    plt.title('$\lambda_1$, $\lambda_2$ from Gibbs sampling')
    plt.xlabel('$\lambda_1$')
    plt.ylabel('$\lambda_2$',rotation=0)
    plt.savefig('Lambdas Scatterplot.png')

    # calculating mean
    n0_mean = np.mean(n0)
    lambda1_mean = np.mean(lambda1)
    lambda2_mean = np.mean(lambda2)

    return n0_mean, lambda1_mean, lambda2_mean


##################################################
# Execution and parameters assinging below
##################################################

iterations = 10000 # total MCMC Gibbs sampling iteration
n0_init = 5 # initial value of n0 we decide for when it starts the montecarlo stuff
gamma_init = [8, 1] # initial guess of alpha and beta value of the gamma distribution 
file_name = 'Grav_wave_data.csv' # file name path

n0, l1, l2 = execution(iterations, n0_init, gamma_init, file_name)
print('The mean value for n_0, lambda_1, and lambda_2 shown below:')
print('n_0: ', n0)
print('lambda_1: ', l1)
print('lambda_2: ', l2)