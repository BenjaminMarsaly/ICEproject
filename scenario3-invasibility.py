# -*- coding: utf-8 -*-
"""
Master MODE 
UE ICE
Oct 2020
Clémence BLESTEL - Marine GUETTIER - Benjamin MARSALY

"""
    #####################################################
    #       Chapitre 3 - Invasibility Analysis          #
    #       Scenario 3 - Code 1/3                       #
    #       Pairwise Invasibility Analysis              #
    #####################################################

# import librairies
#-------------------
import numpy as np
import numpy 
from sklearn.linear_model import LinearRegression
import itertools
import math
import matplotlib.pyplot as plt 


# Functions 
#------------------

# Density Dependance function for trajectories
def DD_FUNCTION1(n,ALPHA,BETA):
    return (ALPHA*math.exp(-BETA*n))


# Density Dependance function for invasibility
def DD_FUNCTION2(ALPHA,N1,N2):
    BETA = ALPHA*0.001 # ALPHA BETA relationship
    N = N1*ALPHA*math.exp(-BETA*N2) # Ricker function
    return(N)


# POP_DYNAMICS specifying population dynamics
def POP_DYNAMICS(ALPHA):

    ALPHA_resident = ALPHA[0]           # Alpha for resident
    ALPHA_invader = ALPHA[1]            # Alpha for invader
    Maxgen1 = 50                        # Generations when only invader present
    Maxgen2 = 300                       # Generations after invader introduced
    Tot_Gen = Maxgen1 + Maxgen2         # Total number of generations
    N_resident = np.zeros((Tot_Gen,1))  # Allocate space
    N_resident[0] = 1                   # Initial number of resident
    N_invader = np.zeros((Tot_Gen,1))
    N_invader[Maxgen1] = 1              # Initial number of invader

    for Ige in range(1,Maxgen1+1):      # Iterate over only resident
        N_resident[Ige] = DD_FUNCTION2(ALPHA_resident, N_resident[Ige-1], N_resident[Ige-1])
    # End of resident only period
    # Now add invader
    J = Maxgen1+1                       # Staring generation of this period
    for Ig in range(J,Tot_Gen):         # Iterate after introduction of invader
        N_total = N_resident[Ig-1]+ N_invader[Ig-1] # Total popn size
        
        N_resident[Ig] = DD_FUNCTION2(ALPHA_resident, N_resident[Ig-1],N_total)# Invader population size
        N_invader[Ig] = DD_FUNCTION2(ALPHA_invader, N_invader[Ig-1],N_total) 
 
    Generation = np.arange (1, Tot_Gen+1) # Generation sequence
    Nstart = 10 + Maxgen1               # Starting point for regression
    # Regression model
    y = np.log(N_invader[Nstart:Tot_Gen])
    x = (np.array(Generation[Nstart:Tot_Gen])).reshape(-1,1)
    Invasion_Model = LinearRegression()
    Invasion_Model = Invasion_Model.fit(x,y)
    Elasticity = Invasion_Model.coef_ # Elasticity
    return(Elasticity)


# Combinations making combinations of pairs
def Combinations(A_Resident, A_Invader):
    d1 = list(itertools.product(A_Resident,A_Invader))
    d1 = np.array(d1)
    d = np.zeros((900,2))
    d[:,0] = d1[:,1]
    d[:,1] = d1[:,0]
    return (d)



# Population dynamics with the Ricker function
#----------------------------------------------

A = [2,10,20]                           # Values of alpha
for j in range(1,4):                    # Iterate over values of alpha
    N_t = range(0,1000+1)               # Population sizes 
    ALPHA = A[j-1]                      # alpha
    BETA = ALPHA*0.001                  # Beta
    # Plot N(t+1) vs N(t)
    N = len(N_t)                        # N number of values of N(t)
    N_tplus1 = np.zeros((N,1))          # Pre-allocate space for N(t+1)
    for i in range(0,N):                # Iterate over values of N
        N_tplus1[i] = N_t[i]*DD_FUNCTION1(N_t[i],ALPHA,BETA)
    # End of N(t+1) on N(t) calculation
    
    plt.subplot(2, 3,j)
    plt.plot (N_t, N_tplus1)
    plt.xlabel("N(t)")
    plt.ylabel("N(t+1)")
    plt.plot(N_t, N_t) # Plot the line of equality
    
    Maxgen = 100                        # Number of generations
    N = np.zeros((Maxgen,1))            # Pre-allocate space for N(t)
    N[0] =1                             # Initial value of N
    for Igen in range(1,Maxgen):        # Iterate over generations
        N[Igen] = N[Igen-1]*DD_FUNCTION1(N[Igen-1],ALPHA,BETA)
    Generation = range(1,Maxgen+1)      # Vector of generation numbers
    plt.ylim(0,500)
    plt.subplot(2, 3, j+3)
    plt.plot(Generation, N)             # Plot population trajectory



# Pairwise invasibility Analysis
#---------------------------------

N1 = 30                                 # Number of increments
A_Resident = np.linspace(start = 2, stop = 4, num = N1) # Resident alpha
A_Invader = np.linspace(start = 2, stop = 4, num = N1) # Invader alpha

d = Combinations(A_Resident, A_Invader) # Combinations

# Generate values at combinations
z = np.zeros((900,1))
for k in range(0,900):
    z[k] = POP_DYNAMICS (d[k,:])        # Loop to using DD_FUNCTION
z_matrix = z.reshape(N1, N1) # Convert to a matrix

# Plot contours
plt.contour(A_Resident, A_Invader,z_matrix)
plt.scatter(2.725109, 2.725109, c = 'white', edgecolor = 'black', s = 130)
plt.xlabel("Resident")
plt.ylabel("Invader")
plt.show()


