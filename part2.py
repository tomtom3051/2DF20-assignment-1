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

# Exp distribution for R
Exp = stats.expon(scale = 10000)


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
n_damaged= Bin.rvs(n)          #Generate n Bin(k, q) random variates 



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
plt.xlabel("number of damaged houses")
print("(B) mean: " + str(num.mean(n_damaged)))
print("(B) variance: " + str(num.var(n_damaged)))
plt.hist(n_damaged, weights=num.ones(len(n_damaged)) / len(n_damaged))
plt.show()


# (B)(2)
n_sim = 100
n_5000 = num.empty(0)
for i in range(n_sim):
    n_damaged = Bin.rvs(1)
    
    Bin_insured = stats.binom(n_damaged, p)
    n_insured = Bin_insured.rvs(1)

    n_not_insured = n_damaged - n_insured
    n_5000 = num.append(n_5000, n_not_insured)

print(num.mean(n_5000))

plt.figure()
plt.title("(B)")
plt.ylabel("probability")
plt.xlabel("number of houses receiving exactly 5000")
plt.hist(n_5000, weights=num.ones(len(n_5000)) / len(n_5000))
plt.show()


# (C) Take one arbitrary hosue. What is the probability
# that for this house an amount of more than 20 000 euros
# damage reimbursement will be paid?

# Let R denote the amount of reimbursement
# P(R > 20 000)????
# we know that a damage has q probability to be damage
# and that for a has to received > 20000 it must be both damaged and insured
# thus P(R > 20 000) = q * p * P(R > 20 000 | Q and R)
# where Q and R are the events that the house is damaged and that the house is insured
# now we look at the case where the house is damaged and receives insurance
# we know that then R ~ Exp() with mean 10 000

#Exp = stats.expon(scale = 10000) needed for (B)(2) so it was relocated to the beginning of the py file
n = 1000000
R = Exp.rvs(n)      #Generate n Exp(10 000) random variates
#added to verify the mean
Rmean = num.mean(R)
print("Mean amount of damages: " + str(Rmean))

# estimation for P(R > 20 000 | Q and R)
estMoreThan20 = num.sum(R > 20000 )/n
print("Probability of more than 20: " + str(estMoreThan20))
print("(C) mean: " + str(num.mean(R)))
print("(C) variance: " + str(num.var(R)))


# (D) Give the distribution of the total amount
# that is going to be paid after the earthquake

#Set up the number of trials
n = 1000

#Set up an empty array to store the results of each trial
totalPayments = zeros(n)

#For each trial calculate the total payment value
for j in range(n):
    #Initially reset the total payment to 0
    totalPayment = 0

    #Then calculate the individual payment for each house
    for i in range(k):
        #Use float to simulate if house is damaged
        damageInt = random.random()

        #Use float to simulate if house is insured
        insuranceInt = random.random()

        #If undamaged no payment is made
        if damageInt < q:
            payment = 0
        
        #If damaged and uninsured payment = 5000
        elif (damageInt > q and insuranceInt < p):
            payment = 5000
        
        #If damaged and insured payment is calculated
        elif (damageInt > q and insuranceInt > p):
            payment = stats.expon.rvs(scale=10000, size=1)[0]
        
        #Add each payment to the total
        totalPayment = totalPayment + payment
    
    #Once all payments are calculated store the result
    totalPayments[j] = totalPayment

#Print out the mean value and the variance
print("(D) Mean total payment amount: " + str(num.mean(totalPayments)))
print("(D) Variance in total payment amount: " + str(num.var(totalPayments)))

plt.figure()
plt.title("(D)")
plt.ylabel("probability")
#Divide by n later and add % sign
plt.xlabel("Total payment amount")
plt.hist(totalPayments)
plt.show()