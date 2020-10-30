# -*- coding: utf-8 -*-
"""
Master MODE 
UE ICE
Oct 2020
Clémence BLESTEL - Marine GUETTIER - Benjamin MARSALY

"""
    #####################################################
    #       Chapitre 3 - Invasibility Analysis          #
    #       Scenario 6 - Code 2/4                       #
    #       Elasticity Analysis              #
    #####################################################

# import librairies
#-------------------
import math
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
import scipy.optimize


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
    
    ALPHA_resident = ALPHA                      # Resident value
    ALPHA_invader = ALPHA*Coeff                 # Resident value   
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
    print(N)
    J = Maxgen1
    for Igen in range(J,Tot_Gen):
        N = N_resident[Igen-1] + N_invader[Igen-1]
        N_resident[Igen] = DD_FUNCTION(ALPHA_resident, N_resident[Igen-1], N)
        N_invader[Igen] = DD_FUNCTION(ALPHA_invader, N_invader[Igen-1], N)
  
    Generation = np.arange(0,Tot_Gen)                                  
    N0 = 10 + Maxgen1
    y = np.log(N_invader[N0-1:Tot_Gen])
    x = (np.array(Generation[N0-1:Tot_Gen])).reshape(-1,1)
    Invasion_model = LinearRegression()
    Invasion_Model = Invasion_model.fit(x,y)
    Elasticity = Invasion_Model.coef_
    return(Elasticity)


# Main program
#-----------------

minA = 10
maxA = 40                           # Limits of search for alpha
Best_A = scipy.optimize.brentq(POP_DYNAMICS, minA, maxA, 0.995)
# Find optimum alpha and store it 
print(Best_A)                       # print out optimum
N_int =30                           # Nos of intervals for plot
A = np.linspace(start = minA, stop = maxA, num = N_int)

Elasticity = np.zeros((N_int,1))
for AL in range (0, len(A)):
    Elasticity[AL] = POP_DYNAMICS(A[AL], 0.995)

# Now plot Invasion exponent when resident is optimal
plt.subplot(2, 2, 1)
plt.plot(A, Elasticity)
plt.plot([minA, maxA], [0.0, 0.0], 'r-', lw=2)  # Add horizontal line at zero
plt.xlabel("A")
plt.ylabel("Elasticity")

Coeff = A/Best_A                    # Convert alpha to coefficient
Invasion_coeff = np.zeros((N_int,1))# Allocate space

# Calculate invasion coefficient
for i in range(0,N_int):
    Invasion_coeff[i] =POP_DYNAMICS(Best_A, Coeff[i])

plt.subplot(2, 2, 2)
plt.plot(A, Invasion_coeff)         # plot invasion coeff vs alpha
plt.scatter(Best_A, 0, c = 'red')   # add predicted optimum
plt.xlabel("A")
plt.ylabel("Invasion coeff")



# Plot N(t+1) on N(t) for best model
N = 1000                            # Nos of data points
N_t = np.arange(1,N+1)              # Values of N(t)
N_tplus1 = np.zeros((N,1))          # Allocate space
for i in range(1,N):                # Iterate over values of N
    N_tplus1[i] = DD_FUNCTION(Best_A, N_t[i], N_t[i])

plt.subplot(2, 2, 3)
plt.plot(N_t, N_tplus1)             # Plot N(tþ1) on N(t)
plt.xlabel("N(t)")
plt.ylabel("N(t+1)")

# Plot N(t) on t
N = np.ones((100,1))                # Allocate space. Note reuse of N
for i in range(1,100):
    N[i] = DD_FUNCTION(Best_A, N[i-1],N[i-1])
plt.subplot(2, 2, 4)
plt.plot(np.arange(1,100+1),N)
plt.xlabel("Generation")
plt.ylabel("Population")

print(Best_A)


