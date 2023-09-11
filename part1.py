import numpy as num
import random
from scipy import stats
import matplotlib.pyplot as plt

# change this to large n
# i.e -> inf
n = 10

sampleSize = 10000

#Poisson setup
alpha = random.random()*n
alpha = round(alpha, 1)
print(alpha)
Poisson = stats.poisson(alpha)
xs1 = Poisson.rvs(sampleSize)
xs1 = num.sort(xs1)
print(xs1)

#Binom Setup
Binom = stats.binom(n, alpha/n)
xs2 = Binom.rvs(sampleSize)
xs2 = num.sort(xs2)
print(xs2)


#don't know the theory of why this works gonna try to explain this later
n = len(xs1)
ys = num.arange(1/n, 1+1/n, 1/n)

fig, axs = plt.subplots(1, 2, tight_layout=True)
axs[0].step(xs1, ys)
axs[1].step(xs2, ys)
plt.show()
print(num.mean(xs1))
print(num.mean(xs2))