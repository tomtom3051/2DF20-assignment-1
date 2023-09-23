import numpy as num
from numpy import zeros
import random
from scipy import stats
import matplotlib.pyplot as plt


# integer K number of houses > 0
k = 1000

# probability of being damaged 0 < Q < 1
q = 0.1

# probability of being insured 0 < P < 1
p = 0.1

# (A) What is the probability that 
# at most one house is damaged after the earthquake?

# the random variable nr Damaged has a Binomial distribution 
# with k number of trials and q success probability
# nrDamaged ~Bin(k, q) 
# P(X <= 1)?????
# Denote 1{.} the indicator function
# 1{a} = 1 if a holds
# 1{a} = 0 if a does not hold
# 1{a} is a bernoulli random variable
# and E[ 1{X <= 1} ] = P(X <= 1)

Bin = stats.binom(k, q) #Binomial distribution with parameters k and q
n = 100
z = Bin.rvs(n)          #Generate n Bin(k, q) random variates 


# estimation for P(X <= 1)
cdfResult = Bin.cdf(1)

formulaResult = ((1-q)**k) + (k*q*((1-q)**(k-1)))

print("(A) Cumulative Distribution Function probability of 1 or less house being damaged: " + str(cdfResult))
print("Derived equation probability of 1 or less house being damaged: " + str(formulaResult))


x = num.arange(0, 2)
cdf_values = Bin.cdf(x)


plt.plot(x, cdf_values, linestyle='-', color='b')
plt.xlabel('X')
plt.ylabel('CDF')
plt.title('(A)')
plt.grid(True)





# (B) Give the distribution of the number N of damaged houses.
# Also give the mean and probability distribution of the number of houses
# that receive exactly 5000 euros.
plt.figure()
plt.title("(B)")
plt.ylabel("probability")
#Divide by n later and add % sign
plt.xlabel("number of damaged houses")
print("(B) mean: " + str(num.mean(z)))
print("(B) variance: " + str(num.var(z)))
plt.hist(z)
#plt.show()

# (C) Take one arbitrary hosue. What is the probability
# that for this house an amount of more than 20 000 euros
# damage reimbursement will be paid?

# Let R denote the amount of reimbursement
# P(R > 20 000)????
# we know that a damage has q probability to be damager
# and that if the house is not damaged then it is not insured
# and cannot receive more than 20 000 reimbursement
# P(R > 20 000) <= q;
# now we look at the case where the house is damaged and receives insurance
# we know that then R ~ Exp() with mean 10 000

Exp = stats.expon(scale = 10000)
n = 10000
R = Exp.rvs(n)      #Generate n Exp(10 000) random variates
Rmean = num.mean(R)
print("Mean amount of damages: " + str(Rmean))

# estimation for P(R > 20 000)
estMoreThan20 = num.sum(R > 20000 )/n
print("Probability of more than 20" + str(estMoreThan20))
plt.figure()
plt.title("(C)")
plt.ylabel("probability")
#Divide by n later and add % sign
plt.xlabel("amount of damage")
print("(B) mean: " + str(num.mean(R)))
print("(B) variance: " + str(num.var(R)))
plt.hist(R)
# plt.show()

# now compute the probablity of the probability Q
# i.e P(R > 20 000) = estMoreThan20 * q

# (D) Give the distribution of the total amount
# that is going to be paid after the earthquake

totalPayment = 0
payments = zeros(k)

for i in range(k):
    damageInt = random.random()
    insuranceInt = random.random()

    if damageInt < q:
        payment = 0
    elif (damageInt > q and insuranceInt < p):
        payment = 5000
    elif (damageInt > q and insuranceInt > p):
        payment = stats.expon.rvs(scale=10000, size=1)
    
    payments[i] = payment
    totalPayment = totalPayment + payment


print("Total payment: " + str(totalPayment))
print(payments)
    
plt.figure()
plt.title("(D)")
plt.ylabel("probability")
#Divide by n later and add % sign
plt.xlabel("payment amount")
print("(D) mean: " + str(num.mean(payments)))
print("(D) variance: " + str(num.var(payments)))
plt.hist(payments)
plt.show()