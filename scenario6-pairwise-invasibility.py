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
import pandas as pd

import matplotlib.pyplot as plt 

def DD_FUNCTION(ALPHA, N1, N2):
    Amin = 1
    Amax = 50
    Bmin = 0
    Bmax = 0.9
    a = 20
    BETA = 0.01
    AA = (ALPHA-Amin)/(Amax-Amin)
    S = (Bmax-Bmin)*(1-AA)/(1 + a*AA)
    N = N1*(ALPHA*math.exp(-BETA*N2)+ S)
    return(N)

def POP_DYNAMICS(ALPHA):
    
    ALPHA_resident = ALPHA[0]
    ALPHA_invader = ALPHA[1]
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
  
    
    Generation = np.arange(1,Tot_Gen+1)                                  
    N0 = 10 + Maxgen1
 
    y = np.log(N_invader[N0-1:Tot_Gen])
    x = (np.array(Generation[N0-1:Tot_Gen])).reshape(-1,1)
    Invasion_model = LinearRegression()
    Invasion_Model = Invasion_model.fit(x,y)
    Elasticity = Invasion_Model.coef_
    return(Elasticity)

N1 = 30
A_Resident = np.linspace(start = 5, stop = 45, num = N1)
A_Invader = np.linspace(start = 5, stop = 45, num = N1)

def Combinations(A_Resident, A_Invader):
    d1 = list(itertools.product(A_Resident,A_Invader))
    d1 = np.array(d1)
    d = np.zeros((900,2))
    d[:,0] = d1[:,1]
    d[:,1] = d1[:,0]
    return (d)

d = Combinations(A_Resident,A_Invader)

z = np.zeros((900,1))
for i in range(0,900):
    z[i] = POP_DYNAMICS(d[i,:])
z
z_matrix = z.reshape(N1, N1) # Convert to a matrix
z_matrix

plt.contour(A_Resident, A_Invader,z_matrix)
plt.xlabel("Resident")
plt.ylabel("Invader")
plt.show()


