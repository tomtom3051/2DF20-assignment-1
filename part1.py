import numpy as num
import random
from scipy import stats
import matplotlib.pyplot as plt

# change this to large n to see the difference in the plots
# for large n the distributions are similar
# i.e -> inf
n = 10000000

sampleSize = n

#Poisson setup
alpha = random.random()*1000
alpha = round(alpha, 1)
print(alpha)
Poisson = stats.poisson(alpha)
xs1 = Poisson.rvs(sampleSize)
xs1 = num.sort(xs1)

#Binom Setup
Binom = stats.binom(n, alpha/n)
xs2 = Binom.rvs(sampleSize)
xs2 = num.sort(xs2)


fig, axs = plt.subplots(1, 2, tight_layout=True)
axs[0].hist(xs1, bins=20)
axs[1].hist(xs2, bins=20)
print("mean binom: " +  str(num.mean(xs2)))
print("mean pois: " + str(num.mean(xs1)))
plt.title("Comparison between Poisson (left) and Binom (right)")
plt.show()
