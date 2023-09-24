import numpy as num
from numpy import zeros
import random
from scipy import stats
import matplotlib.pyplot as plt

#Set a sample size
sampleSize = 10000

#The array xs1 will be used to store Poisson variables
xs1 = zeros(sampleSize)
#The array xs2 will be used to store Binomial variables
xs2 = zeros(sampleSize)

#Set up the different trials
for i in range (sampleSize):
    #Generate a large value for n
    #This simulates n -> infinity
    n = random.randint(1000000, 10000000)
    #Set up a random value for lambda
    #1 <= lambda < n so lam/n > 1
    lam = random.randint(1, n)
    
    #Set up a Poisson model
    Poisson = stats.poisson(lam)

    #Get a range of different values from this model
    xsPoison = Poisson.rvs(sampleSize)

    #Store the mean as the result for this trial
    xs1[i] = num.mean(xsPoison)

    #Set up a Binomial model using n and lam/n for p
    Binom = stats.binom(n, lam/n)

    #Get a range of different values from this model
    xsBinom = Binom.rvs(sampleSize)

    #Store the mean as the result for this trial
    xs2[i] = num.mean(xsBinom)

#Plot the binomial and poisson values
fig, axs = plt.subplots(1, 2, tight_layout=True)
axs[0].hist(xs1, bins=20)
axs[1].hist(xs2, bins=20)
#Get the mean and variance for both
print("mean binom: " +  str(num.mean(xs2)))
print("variance binom: " +  str(num.var(xs2)))
print("mean pois: " + str(num.mean(xs1)))
print("variance pois: " +  str(num.var(xs1)))
plt.suptitle("Comparison between Poisson (left) and Binom (right)")
plt.show()

