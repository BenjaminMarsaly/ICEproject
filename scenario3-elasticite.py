# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:09:51 2020

@author: Margue
"""
import numpy as np
from sklearn.linear_model import LinearRegression
import random
import math
import matplotlib.pyplot as plt 

import scipy.optimize

############## Elasticite ##############

def DD_FUNCTION(ALPHA,N1,N2):
    BETA = ALPHA*0.001
    N = N1*ALPHA*math.exp(-BETA*N2)
    return(N)


def POP_DYNAMICS(ALPHA, Coeff):
    ALPHA_resident = ALPHA # Alpha for resident
    ALPHA_invader = ALPHA_resident*Coeff # Alpha for invader
    Maxgen1 = 50 # Generations when only invader present
    Maxgen2 = 300 # Generations after invader introduced
    Tot_Gen = Maxgen1 + Maxgen2 # Total number of generations
    N_resident = np.zeros((Tot_Gen,1)) # Allocate space
    N_resident[0] = 1 # Initial number of resident
    N_invader = np.zeros((Tot_Gen,1))
    N_invader[Maxgen1] = 1 # Initial number of invader

    for Ige in range(1,Maxgen1+1): # Iterate over only resident
        N_resident[Ige] = DD_FUNCTION(ALPHA_resident, N_resident[Ige-1], N_resident[Ige-1])

    # Now add invader
    J = Maxgen1 + 1 # Staring generation of this period
    for Ig in range(J,Tot_Gen): # Iterate after introduction of invader
        N_total = N_resident[Ig-1]+ N_invader[Ig-1] # Total popn size
        # Resident population size
        N_resident[Ig] = DD_FUNCTION(ALPHA_resident, N_resident[Ig-1],N_total)# Invader population size
        N_invader[Ig] = DD_FUNCTION(ALPHA_invader, N_invader[Ig-1],N_total)
    
    Generation = np.arange (1, Tot_Gen+1) # Generation sequence
    Nstart = 10 + Maxgen1 # Starting point for regression
    # Regression model
    y = np.log(N_invader[Nstart:Tot_Gen])
    x = (np.array(Generation[Nstart:Tot_Gen])).reshape(-1,1)
    Invasion_Model = LinearRegression()
    Invasion_Model = Invasion_Model.fit(x,y)
    Elasticity = Invasion_Model.coef_ # Elasticity
    
    return(Elasticity)

def constrainedFunction(x, f, lower, upper, minIncr=0.001):
     x = np.asarray(x)
     lower = np.asarray(lower)
     upper = np.asarray(upper)
     xBorder = np.where(x<lower, lower, x)
     xBorder = np.where(x>upper, upper, xBorder)
     fBorder = f(xBorder)
     distFromBorder = (np.sum(np.where(x<lower, lower-x, 0.))
                      +np.sum(np.where(x>upper, x-upper, 0.)))
     return (fBorder + (fBorder
                       +np.where(fBorder>0, minIncr, -minIncr))*distFromBorder)

############### MAIN PROGRAM ###############

minA = 1
maxA = 10 # Limits for search

Best_Alpha = scipy.optimize.brentq(POP_DYNAMICS, minA, maxA, 0.995)

print(Best_Alpha) # Print out optimum

# Plot Elasticity vs alpha
N_int = 30 # Nos of intervals for plot
Alpha = np.linspace(start = minA, stop = maxA, num = N_int)

Elasticity = np.zeros((N_int,1))
for AL in range (0, len(Alpha)):
    Elasticity[AL] = POP_DYNAMICS(Alpha[AL], 0.995)



plt.plot(Alpha, Elasticity) ; plt.plot([minA, maxA], [0.0, 0.0], 'r-', lw=2)

# Plot Invasion exponent when resident is optimal
Coeff = Alpha/Best_Alpha # Convert alpha to coefficient
Invasion_coeff = np.zeros((N_int,1))# Allocate space

# Calculate invasion coefficient
for i in range(0,N_int):
    Invasion_coeff[i] =POP_DYNAMICS(Best_Alpha, Coeff[i])

plt.plot(Alpha, Invasion_coeff) # Plot invasion coeff on alpha
#plt.points(Best.Alpha,0, cex¼2) # Plot optimum alpha on graph

# Plot N(tþ1) on N(t) for optimum alpha
maxN = 1000
# Number of N
N_t = np.arange(1,maxN+1) # Values of N(t)

N_tplus1 = np.zeros((maxN,1)) # Allocate space for N(tþ1)
for i in range(1,maxN): # Iterate over values of N
    N_tplus1[i] = DD_FUNCTION(Best_Alpha, N_t[i], N_t[i])

plt.plot(N_t, N_tplus1)
plt.xlabel("N(t)")
plt.ylabel("N(t+1)")
# Plot N(t) on t
N = np.ones((100,1)) # Allocate space. Note reuse of N
for i in range(1,100):
    N[i] = DD_FUNCTION(Best_Alpha, N[i-1],N[i-1])
plt.plot(np.arange(1,100+1),N)
plt.xlabel("Generation")
plt.ylabel("Population")

print(Best_Alpha)
