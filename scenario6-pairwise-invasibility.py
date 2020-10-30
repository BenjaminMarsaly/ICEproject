# -*- coding: utf-8 -*-
"""
Master MODE 
UE ICE
Oct 2020
Clémence BLESTEL - Marine GUETTIER - Benjamin MARSALY

"""
    #####################################################
    #       Chapitre 3 - Invasibility Analysis          #
    #       Scenario 6 - Code 1/3                       #
    #       Pairwise Invasibility Analysis              #
    #####################################################

# import librairies
#-------------------
import math
import numpy as np
from sklearn.linear_model import LinearRegression
import itertools
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
    AA = (ALPHA-Amin)/(Amax-Amin)           # For convenience
    S = (Bmax-Bmin)*(1-AA)/(1 + a*AA)       # Survival
    N = N1*(ALPHA*math.exp(-BETA*N2)+ S)    # new population
    return(N)

# population dynamics function
def POP_DYNAMICS(ALPHA):
    
    ALPHA_resident = ALPHA[0]               # Resident value
    ALPHA_invader = ALPHA[1]                # Invader value
    Maxgen1 = 50                            # Nos of generations for resident only
    Maxgen2 = 300                           # Nos of generations after invasion
    Tot_Gen = Maxgen1 + Maxgen2             # Total number of generations
    N_resident = np.zeros((Tot_Gen,1))      # Allocate space
    N_resident[0] = 1                       # Initial pop size of resident
    N_invader = np.zeros((Tot_Gen,1))       
    N_invader[Maxgen1-1] = 1                # Initial pop size of invader
    
    for Ige in range(1,Maxgen1):            # Iterate over generations with resident only
        N = N_resident[Ige-1]
        N_resident[Ige] = DD_FUNCTION(ALPHA_resident, N, N)
    print(N)
    # Now add invader
    J = Maxgen1                             # Starting generation
    for Igen in range(J,Tot_Gen):           # Iterate over generations with invader
        N = N_resident[Igen-1] + N_invader[Igen-1] # Total pop size
        N_resident[Igen] = DD_FUNCTION(ALPHA_resident, N_resident[Igen-1], N)
        N_invader[Igen] = DD_FUNCTION(ALPHA_invader, N_invader[Igen-1], N)
  
    Generation = np.arange(1,Tot_Gen+1)     # Vector of generation numbers                        
    N0 = 10 + Maxgen1                       # Gen at which to start regression analysis
    y = np.log(N_invader[N0-1:Tot_Gen])     # Linear regression
    x = (np.array(Generation[N0-1:Tot_Gen])).reshape(-1,1)
    Invasion_model = LinearRegression()
    Invasion_Model = Invasion_model.fit(x,y)
    Elasticity = Invasion_Model.coef_
    return(Elasticity)

# Combinations making combinations of pairs
def Combinations(A_Resident, A_Invader):
    d1 = list(itertools.product(A_Resident,A_Invader))
    d1 = np.array(d1)
    d = np.zeros((900,2))
    d[:,0] = d1[:,1]
    d[:,1] = d1[:,0]
    return (d)


# Main program 
#------------------

N1 = 30                                      # Nos of increments
A_Resident = np.linspace(start = 5, stop = 45, num = N1) # Resident alphas
A_Invader = np.linspace(start = 5, stop = 45, num = N1)  # Invader alphas

d = Combinations(A_Resident,A_Invader)      # Combinations

z = np.zeros((900,1))
for i in range(0,900):
    z[i] = POP_DYNAMICS(d[i,:])
z_matrix = z.reshape(N1, N1) # Convert to a matrix

plt.contour(A_Resident, A_Invader,z_matrix) # Plot contours
plt.scatter(24.21837, 24.21837, c = 'white', edgecolor = 'black', s = 130)
plt.xlabel("Resident")
plt.ylabel("Invader")
plt.show()


