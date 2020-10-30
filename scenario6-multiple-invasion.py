# -*- coding: utf-8 -*-
"""
Master MODE 
UE ICE
Oct 2020
Clémence BLESTEL - Marine GUETTIER - Benjamin MARSALY

"""
    #####################################################
    #       Chapitre 3 - Invasibility Analysis          #
    #       Scenario 6 - Code 4/4                       #
    #       Multiple Invasibility Analysis              #
    #####################################################

# import librairies
#-------------------
import math
import numpy as np
import random
import matplotlib.pyplot as plt


# Functions
#-------------------

# Density Dependance function
def DD_FUNCTION(X,N_total):
    ALPHA = X[0]
    N = X[1]
    Amin = 1
    Amax = 50
    Bmin = 0
    Bmax = 0.9
    a = 20
    BETA = 0.01
    AA = (ALPHA-Amin)/(Amax-Amin)               # For convenience
    S = (Bmax-Bmin)*(1-AA)/(1 + a*AA)           # Survival
    N = N*(ALPHA*math.exp(-BETA*N_total)+ S)    # new population
    N = max(0,N)                                # N cannot be negative
    return(N)


# Main program
#-----------------

random.seed(10)                 # Initialize the random number seed
Maxgen = 10000                  # Number of generations run
Stats = np.zeros((Maxgen,3))    # Allocate space for statistics
MaxAlpha = 50                   # maximum value of alpha
Ninc = 50                       # Number of classes for alpha
# Allocate space to store data for each generation
Store =  np.zeros((Maxgen,Ninc))
# Allocate space for alpha class and population size
Data = np.zeros((Ninc,2))
Data[23,1] = 1                  # Initial population size and alpha class

ALPHA =  np.linspace(start = 1, stop = MaxAlpha, num = Ninc) # Set Alpha
Data[:,0] = ALPHA               # Place alpha in first column

for Igen in range(0,Maxgen):    # Iterate over generations
    N_total = sum(Data[:,1])    # total population size
    for d in range(len(Data[:,1])):
        Data[d,1] = DD_FUNCTION(Data[d,:], N_total) # New cohort
    Store[Igen,:] = Data[:,1]   # Store values for this generation
    # Keep track of population size, mean trait value and SD of trait value
    Stats[Igen,1] = sum(Data[:,0]*Data[:,1])/sum(Data[:,1]) # Mean
    S = sum(Data[:,1])          # Population size
    Stats[Igen,0] = S           # Population size
    s1 = np.zeros((len(Data[:,1]),1))
    s2 = np.zeros((len(Data[:,1]),1))
    for h in range(0,len(Data[:,1])):
        s1[h] = ((Data[h,0])**2)*Data[h,1]
        s2[h] = Data[h,0]*Data[h,1]
    SX1 = sum(s1)
    SX2 = (sum(s2))**2/S 
    Stats[Igen,2] = np.sqrt((SX1-SX2)/(S-1)) # SD of trait
    # Introduce a mutant by picking a random integer between 1 and 50
    Mutant = random.randint(0,49)  
    Data[Mutant,1] = Data[Mutant,1]+1 #Add mutant to class
    
for Row in range(9997,Maxgen):  # Select rows to be plotted
        plt.show()
        plt.plot(ALPHA,Store[Row,:])
        plt.xlabel("Alpha")
        plt.ylabel("Number")
    # End of frequency polygon plots

Generation = np.arange(1, Maxgen + 1) # Vector of generations
N0 = 9900                       # Starting value for plots
plt.subplot(1, 3, 1)
plt.plot(Generation[N0:Maxgen+1], Stats[N0:Maxgen+1,0], label ="Population size")
plt.xlabel("Population size")
plt.ylabel("Generation")
plt.subplot(1, 3, 2)
plt.plot(Generation[N0:Maxgen+1], Stats[N0:Maxgen+1,1], label ="Mean")
plt.xlabel("Mean")
plt.ylabel("Generation")
plt.subplot(1,3,3)
plt.plot(Generation[N0:Maxgen+1], Stats[N0:Maxgen+1,2], label ="SD")
plt.xlabel("SD")
plt.ylabel("Generation")

print("Mean alpha in last gen =", Stats[Maxgen-1,1])
print("SD of alpha in last gen = ", Stats[Maxgen-1,2])
   
