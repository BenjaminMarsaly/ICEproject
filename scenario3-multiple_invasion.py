# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:08:53 2020

@author: Margue
"""

# package


import numpy as np
from sklearn.linear_model import LinearRegression
import random
import math
import matplotlib.pyplot as plt 


import numpy 
import itertools


# fonction de densite
def DD_FUNCTION(X, N_total):
    ALPHA = X[0]
    N = X[1]
    BETA = ALPHA*0.001
    N = N*ALPHA*math.exp(-BETA*N_total)
    return (N)



random.seed(10)      # nombre depart initialise
Maxgen = 5000 # Nombre de generation
Stats = np.zeros((Maxgen,3))            # Initialise une matrice de Maxgen ligne et 3 colonnes
MaxAlpha = 4 # maximum valeur de alpha
Ninc = 50 # Nombre de classe pour alpha

Store =  np.zeros((Maxgen,Ninc))   # matrice 0 de row= Maxgen, col = Ninc

Data = np.zeros((Ninc,2)) # matrice 0 de row= Ninc, col = 2
Data[23,1] = 1 # initial pop et class de alpha

ALPHA =  np.linspace(start = 2, stop = MaxAlpha, num = Ninc) # matrice avec 1 col et row =Ninc et alpha variant de 2 a 4
 
Data[:,0] = ALPHA # A

for Igen in range(0,Maxgen): 
    N_total = sum(Data[:,1]) # totale de l'effectif
    for d in range(0,len(Data[:,1])):
        Data[d,1] = DD_FUNCTION(Data[d,:],N_total) #retourne N via la fonction DD
    Store[Igen,:] = Data[:,1]
    Stats[Igen,1] = sum(Data[:,0]*Data[:,1])/sum(Data[:,1]) # Moyenne
    S = sum(Data[:,1])
    Stats[Igen,0] = S 
    s1 = np.zeros((len(Data[:,1]),1))
    s2 = np.zeros((len(Data[:,1]),1))
    for h in range(0,len(Data[:,1])):
        s1[h] = ((Data[h,0])**2)*Data[h,1]
        s2[h] = Data[h,0]*Data[h,1]
    SX1 = sum(s1)
    SX2 = (sum(s2))**2/S 
    Stats[Igen,2] = np.sqrt((SX1-SX2)/(S-1)) # trouver sqrt fonction
    Mutant = random.randint(0,49) # faut le transformer 
    Data[Mutant,1] = Data[Mutant,1]+1 
  
plt.subplot(2, 2, 1)
plt.plot(ALPHA,Store[Maxgen-1,:])
plt.xlabel("ALPHA")
plt.ylabel("Number")


Generation = np.arange(1, Maxgen + 1)
N0 = 1
plt.subplot(2, 2, 2)
plt.plot(Generation[N0:Maxgen], Stats[N0:Maxgen,0], label ="Population size")
plt.subplot(2, 2, 3)
plt.plot(Generation[N0:Maxgen], Stats[N0:Maxgen,1], label ="Mean")
plt.subplot(2, 2, 4)
plt.plot(Generation[N0:Maxgen], Stats[N0:Maxgen,2], label ="SD")

print("Mean alpha in last gen =", Stats[Maxgen-1,1])
print("SD of alpha in last gen = ", Stats[Maxgen-1,2])