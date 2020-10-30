# -*- coding: utf-8 -*-
"""
Master MODE 
UE ICE
Oct 2020
Clémence BLESTEL - Marine GUETTIER - Benjamin MARSALY

"""
    #####################################################
    #       Chapitre 3 - Invasibility Analysis          #
    #       Scenario 6 - Code 3/4                       #
    #       Elasticity Invasibility Analysis              #
    #####################################################

# import librairies
#-------------------
import math
import numpy as np
import matplotlib.pyplot as plt 


# Functions
#-------------------

# Density Dependance function
def DD_FUNCTION(ALPHA, N1, N2):
    Amin = 1
    Amax = 50
    Bmin = 0
    Bmax = 0.9
    a = 20
    BETA = 0.01
    AA = (ALPHA-Amin)/(Amax-Amin)               # For convenience
    S = (Bmax-Bmin)*(1-AA)/(1 + a*AA)           # Survival
    N = N1*(ALPHA*math.exp(-BETA*N2)+ S)        # new population
    return(N)

# population dynamics function
def POP_DYNAMICS(ALPHA,Coeff):
    
    ALPHA_resident = ALPHA
    ALPHA_invader = ALPHA*Coeff
    Maxgen1 = 50
    Maxgen2 = 300
    Tot_Gen = Maxgen1 + Maxgen2
    
    N_resident = np.zeros((Tot_Gen,1))
    N_resident[0] = 1
    N_invader = np.zeros((Tot_Gen,1))
    N_invader[Maxgen1-1] = 1
    
    for Ige in range(1,Maxgen1):
        N = N_resident[Ige-1]
        N_resident[Ige] = DD_FUNCTION(ALPHA_resident, N, N)
    J = Maxgen1
    for Igen in range(J,Tot_Gen):
        N = N_resident[Igen-1] + N_invader[Igen-1]
        N_resident[Igen] = DD_FUNCTION(ALPHA_resident, N_resident[Igen-1], N)
        N_invader[Igen] = DD_FUNCTION(ALPHA_invader, N_invader[Igen-1], N)
  
    Generation = np.arange(1,Tot_Gen+1)                                  
    
    # New elements in the function: plots
    plt.subplot(1, 2, 1)
    plt.plot(Generation, N_resident)
    plt.xlabel("Generation")
    plt.ylabel("Resident")
    
    plt.subplot(1, 2, 2)
    plt.plot(Generation, N_invader)
    plt.xlabel("Generation")
    plt.ylabel("Invader")

Best_A = 24.21635           # Best alpha from elasticity analysis
Invader_A = 10              # Alpha for invader
Coeff = Invader_A/Best_A    # Calculate relevant coefficient
POP_DYNAMICS(Best_A,Coeff)  # Call functiom
