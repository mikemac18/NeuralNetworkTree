import numpy as np
from scipy.stats import truncnorm
import math
import matplotlib.pyplot as plt


'''
Below is a regular Gaussian distribution, without any ranges or bounds.
'''
mu = 10  #mean
sigma = 5  #standard deviation
s = abs(np.random.normal(mu, sigma, 50))
#print(s)


'''
To get a normal distirbution with bounds, we can use truncnorm().
From: https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy
'''

def get_truncated_normal(mean, sd, low, upp):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

X = get_truncated_normal(mean=30, sd=8, low=1, upp=60)
vals = X.rvs(10000)
#print(vals)

w = 0.45
n = math.ceil((vals.max() - vals.min())/w) #Create bins of size 0.10
plt.hist(vals,bins=n)
plt.xlabel("Label Number")
plt.ylabel("Frequency (count)")
plt.title("Weight: mean 30, std 8")
plt.savefig("Weight_Gaussian_10000_PNG.png",dpi=600,bbox_inches='tight')
plt.show()
np.savetxt("Weight_Gaussian_10000_CSV.csv",vals,delimiter=",",fmt='%f')
