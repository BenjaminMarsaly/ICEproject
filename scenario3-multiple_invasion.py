# -*- coding: utf-8 -*-
"""
Master MODE 
UE ICE
Oct 2020
Clémence BLESTEL - Marine GUETTIER - Benjamin MARSALY

"""
    #####################################################
    #       Chapitre 3 - Invasibility Analysis          #
    #       Scenario 3 - Code 3/3                       #
    #       Multiple Invasibility Analysis              #
    #####################################################

# import librairies
#-------------------
import numpy as np
import random
import math
import matplotlib.pyplot as plt 
import numpy 


# Functions 
#------------------

# Density Dependance Ricker function
def DD_FUNCTION(X, N_total):
    ALPHA = X[0]
    N = X[1]
    BETA = ALPHA*0.001
    N = N*ALPHA*math.exp(-BETA*N_total)
    return (N)


# Main program
#-----------------


random.seed(10)                         # Initialize the random number seed
Maxgen = 5000                           # Number of generations run
Stats = np.zeros((Maxgen,3))            # Allocate space for statistics
MaxAlpha = 4                            # maximum value of alpha
Ninc = 50                               # Number of classes for alpha

# Allocate space to store data for each generation
Store =  np.zeros((Maxgen,Ninc))        # matrix full of 0 with Maxgen rows and Ninc columns
# Allocate space for alpha class and population size
Data = np.zeros((Ninc,2))               # matrix full of 0 with Ninc rows and 2 columns
Data[23,1] = 1                          # initial population size and alpha class

ALPHA =  np.linspace(start = 2, stop = MaxAlpha, num = Ninc) # Set alpha
 
Data[:,0] = ALPHA                       # alpha in the first column

for Igen in range(0,Maxgen):            # Iterate over generations
    N_total = sum(Data[:,1])            # Total population size
    for d in range(0,len(Data[:,1])):   # New cohort
        Data[d,1] = DD_FUNCTION(Data[d,:],N_total) #loop using DD_FUNCTION
    Store[Igen,:] = Data[:,1]           # Store values for this generation
    Stats[Igen,1] = sum(Data[:,0]*Data[:,1])/sum(Data[:,1]) 
    # Keep track of population size, mean trait value and SD of trait value
    S = sum(Data[:,1])                  # population size  
    Stats[Igen,0] = S                   # population size 
    s1 = np.zeros((len(Data[:,1]),1))
    s2 = np.zeros((len(Data[:,1]),1))
    for h in range(0,len(Data[:,1])):
        s1[h] = ((Data[h,0])**2)*Data[h,1]
        s2[h] = Data[h,0]*Data[h,1]
    SX1 = sum(s1)
    SX2 = (sum(s2))**2/S 
    Stats[Igen,2] = np.sqrt((SX1-SX2)/(S-1)) # SD of trait
    # Introduce a mutant by picking a random integer between 0 and 49
    Mutant = random.randint(0,49) 
    Data[Mutant,1] = Data[Mutant,1]+1   # Add mutant to class

plt.subplot(2, 2, 1)                    # Split graphics page into quadrats
# Plot last row of Store
plt.plot(ALPHA,Store[Maxgen-1,:])
plt.xlabel("ALPHA")
plt.ylabel("Number")


Generation = np.arange(1, Maxgen + 1)
N0 = 1
plt.subplot(2, 2, 2)
plt.plot(Generation[N0:Maxgen], Stats[N0:Maxgen,0], label ="Population size")
plt.xlabel("Generation")
plt.ylabel("Population Size")
plt.subplot(2, 2, 3)
plt.plot(Generation[N0:Maxgen], Stats[N0:Maxgen,1], label ="Mean")
plt.xlabel("Generation")
plt.ylabel("Mean of alpha")
plt.subplot(2, 2, 4)
plt.plot(Generation[N0:Maxgen], Stats[N0:Maxgen,2], label ="SD")
plt.xlabel("Generation")
plt.ylabel("SDof alpha")


print("Mean alpha in last gen =", Stats[Maxgen-1,1])
print("SD of alpha in last gen = ", Stats[Maxgen-1,2])