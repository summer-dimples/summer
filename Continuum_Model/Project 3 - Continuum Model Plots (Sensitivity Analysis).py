# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:27:18 2025

@author: Joseph Singh-Best
"""

import math
import numpy as np
import matplotlib.pyplot as plt 

# Range of speed limits v_max values & the maximum traffic density p_max
v_max = np.array([30, 40, 50, 60, 70])
p_max = np.array([200])

# Range of desnity values used to plot against speed, flow, wavespeed
density_rho = p = np.linspace(0, 300, 10000)

# Range of speed-density relationship order values for NL-LWR model
m = np.array([0.25, 0.5, 1, 2, 4])

# p giving maximum flow q_max
'p = p_max * ((m + 1) ** (-1 / m))   gives   q_max = v_max * p_max * ( (m+1) ** (-1 / m) - (m+1) ** -((m+1) / m) )'

#%% 1
""" LWR Speed-Density """

# initialise speed v
v = np.zeros((len(v_max),len(p)))
v = np.ndarray((len(v_max),len(p)))

# Determine speed values by the linear v-p relationship
for k in range(len(p_max)): 
    for i in range(len(v_max)):
        for j in range(len(p)):
                if p[j] < p_max[0]:
                    v[i,j] = v_max[i] * (1 - p[j] / p_max[0])
                else: 
                    v[i,j] = 0
    
    # Plot LWR speed-density for varied v_max
    fig = plt.figure(dpi = 500)
    plt.xlabel("Density, $\\rho$  (vehicles per mile)", fontsize = 18)
    plt.ylabel("Speed, v  (mph)", fontsize = 18)
    plt.tick_params(axis = 'both', labelsize = 16)
    plt.grid(color='lightgray', linestyle='dashed')
    for n in range(len(v_max)):
        plt.plot(p, v[n], label = "$v_{max}$ = " f"{v_max[n]}")
        plt.text(p_max[0] * .65, 30, "$\\rho_{max}$ = " f"{p_max[0]}", fontsize = 18)
        plt.xlim(0,p_max[0] * 1.1)
        plt.ylim(0,max(v_max) * 1.1)
    plt.axvline(p_max[0], - (v_max[i] * 2), v_max[i] * 2, color = 'black')
    plt.legend(framealpha = 1, loc = 'upper right', fontsize = 14)

plt.show()

#%% 2
""" LWR Flow-Density """

# Initialise flow q
q = np.zeros((len(v_max),len(p)))
q = np.ndarray((len(v_max),len(p)))

# Determine flow values by the linear v-p relationship 
for k in range(len(p_max)): 
    for i in range(len(v_max)):
        for j in range(len(p)):
            if v_max[i] * p[j] * (1 - p[j] / p_max[0]) >= 0: 
                q[i,j] = np.round(v_max[i] * p[j] * (1 - p[j] / p_max[0]),0)
            else: 
                q[i,j] = 0

    # Plot LWR speed-density for varied v_max
    fig = plt.figure(dpi = 500)
    plt.xlabel("Density, $\\rho$  (vehicles per mile)", fontsize = 18)
    plt.ylabel("Flow, q  (vehicles per hour)", fontsize = 18)
    plt.tick_params(axis = 'both', labelsize = 12)
    plt.grid(color='lightgray', linestyle='dashed')
    for n in range(len(v_max)):
        plt.plot(p, q[n], label = "$v_{max}$ = " f"{v_max[n]}")
        plt.scatter( (p_max[0] / 2), max(q[n]), color='black', s = 20, zorder = 5)
        plt.text( (p_max[0] / 2) * 0.7, max(q[n]) + (p_max[0] * 0.3), "$q_{max}$ = " f"{int(max(q[n]))}", fontsize = 11.5)
        plt.text(p_max[0] * 1.025, p_max[0] * 5.8, "$\\rho_{max}$ = " f"{p_max[0]}", fontsize = 16)
        plt.xlim(0,p_max[0] * 1.41) 
        plt.ylim(0,max(q[n]) * 1.1)
    plt.axvline(p_max[0], - (v_max[i] * 2), v_max[i] * 2, color = 'black')
    plt.legend(framealpha = 1, loc = 'upper right', fontsize = 12)

plt.show()
    
#%% 3
""" NL-LWR Speed-Density """

# Initialise speed v
v = np.zeros((len(v_max),len(p)))
v = np.ndarray((len(v_max),len(p)))

# Determine speed values by the non-linear v-p relationship
for l in range(len(p_max)): 
    for k in range (len(m)):
        for i in range(len(v_max)):
            for j in range(len(p)):
                if v_max[i] * (1 - ( p[j] / p_max[0] ) ** m[k] ) >= 0: 
                    v[i,j] = v_max[i] * (1 - ( p[j] / p_max[0] ) ** m[k] )
                else: 
                    v[i,j] = 0

        # Plot NL-LWR speed-density for varied v_max, m
        fig = plt.figure(dpi = 500)
        plt.xlabel("Density, $\\rho$  (vehicles per mile)", fontsize = 14)
        plt.ylabel("Speed, v  (mph)", fontsize = 14)
        plt.tick_params(axis = 'both', labelsize = 12)
        plt.grid(color='lightgray', linestyle='dashed')
        for n in range(len(v_max)):
            plt.plot(p, v[n], label = "$v_{max}$ = " f"{v_max[n]}")
            plt.text(p_max[0] * 1.025, v_max[i] * 0.34, "$\\rho_{max}$ = " f"{p_max[0]}", fontsize = 14)
            plt.xlim(0,p_max[0] * 1.4)
            plt.ylim(0,max(v_max) * 1.1)
        plt.axvline(p_max[0], - (v_max[i] * 2), v_max[i] * 2, color = 'black')
        plt.legend(framealpha = 1, loc = 'upper right', fontsize = 14)

plt.show()

#%% 4
""" NL-LWR Flow-Density """

# Initiialse flow q
q = np.zeros((len(v_max),len(p)))
q = np.ndarray((len(v_max),len(p)))

# Determine flow values by the non-linear v-p relationship
for l in range(len(p_max)): 
    for k in range(len(m)):
        for i in range(len(v_max)):
            for j in range(len(p)):
                if v_max[i] * p[j] * (1 - ( p[j] / p_max[0] ) ** m[k] ) >= 0:
                    q[i,j] = np.round(v_max[i] * p[j] * (1 - ( p[j] / p_max[0] ) ** m[k] ),0)
                else: 
                    q[i,j] = 0

        # Plot NL-LWR flow-density for varied v_max, m
        fig = plt.figure(dpi = 500)
        plt.xlabel("Density, $\\rho$  (vehicles per mile)", fontsize = 14)
        plt.ylabel("Flow, q  (vehicles per hour)", fontsize = 14)
        plt.tick_params(axis = 'both', labelsize = 12)
        plt.grid(color='lightgray', linestyle='dashed')
        for n in range(len(v_max)):
            plt.plot(p, q[n], label = "$v_{max}$ = " f"{v_max[n]}")
            plt.scatter( p_max[0] * ((m[k]+ 1) ** (-1 / m[k])), max(q[n]), color='black', s = 20, zorder = 5)
            plt.text( p_max[0] * ((m[k]+ 1) ** (-1 / m[k])) - 27, max(q[n]) * 1.04, "$q_{max}$ = " f"{int(max(q[n]))}", fontsize = 10)
            if n == len(m) - 1: 
                plt.text(p_max[0] * 1.025, max(q[n]) * .4, "$\\rho_{max}$ = " f"{p_max[0]}", fontsize = 12)
                plt.text(p_max[0] * .35, v_max[i] * m[k] ** (3/4), "v-$\\rho$ order = " f"{m[k]}", fontsize = 12)
            plt.xlim(0,p_max[0] * 1.3)
            plt.ylim(0,max(q[n]) * 1.125)
        plt.axvline(p_max[0], - (v_max[i] * 2), v_max[i] * 2, color = 'black')
        plt.legend(framealpha = 1, loc = 'upper right', fontsize = 12)

plt.show() 
    
#%% 5
""" LWR & NL-LWR Speed-Density : v_max fixed """
    
# Initialise speed v
v = np.zeros((len(m),len(p)))
v = np.ndarray((len(m),len(p)))

# Range of density values p_max 
p_max = np.array([40, 80, 120, 160, 200])

# Plot NL-LWR speed density for varied m, p_max
for l in range(len(p_max)): 
    fig = plt.figure(dpi = 500)
    plt.xlabel("Density, $\\rho$  (vehicles per mile)", fontsize = 14)
    plt.ylabel("Speed, v  (mph)", fontsize = 14)
    plt.tick_params(axis = 'both', labelsize = 12)
    plt.grid(color='lightgray', linestyle='dashed')
    for k in range (len(m)):
        for j in range(len(p)):
            if v_max[4] * (1 - ( p[j] / p_max[l] ) ** m[k] ) >= 0: 
                v[k,j] = v_max[4] * (1 - ( p[j] / p_max[l]) ** m[k] )
            else: 
                v[k,j] = 0

        for n in range(len(m)): 
            if k == len(m) - 1: 
                plt.plot(p, v[n], label = "v-$\\rho$ relation order = " f"{m[n]}")
            
            plt.text(p_max[l] / 100, v_max[4] * 1.05, "$v_{max}$ = " f"{v_max[4]}", fontsize = 12)
            plt.text(p_max[l] * 0.77, 50, "$\\rho_{max}$ = " f"{p_max[l]}", fontsize = 12)
            plt.xlim(0,p_max[l] * 1.1)
            plt.ylim(0,v_max[4] * 1.4)
    plt.axvline(p_max[l], - (v_max[4] * 2), v_max[4] * 2, color = 'black')
    plt.axhline(v_max[4], 0, p_max[0] * 2, color = 'black')
    plt.legend(framealpha = 1, fontsize = 10)

plt.show()
    
p_max = np.array([200])
#%% 6
""" LWR & NL-LWR Flow-Density : p_max fixed """

# Initialise flow q
q = np.zeros((len(m),len(p)))
q = np.ndarray((len(m),len(p)))

# Plot NL-LWR flow-density for varied m, v_max
for l in range(len(v_max)): 
    fig = plt.figure(dpi = 500)
    plt.xlabel("Density, $\\rho$  (vehicles per mile)", fontsize = 14)
    plt.ylabel("Flow, q  (vehicles per hour)", fontsize = 14)
    plt.tick_params(axis = 'both', labelsize = 12)
    plt.grid(color='lightgray', linestyle='dashed')
    for k in range (len(m)):
        for j in range(len(p)):
            if v_max[l] * p[j] * (1 - ( p[j] / p_max[math.floor(len(p_max) / 2)] ) ** m[k] ) >= 0: 
                q[k,j] = np.round(v_max[l] * p[j] * (1 - ( p[j] / p_max[math.floor(len(p_max) / 2)] ) ** m[k] ),0)
            else: 
                q[k,j] = 0
   
        for n in range(len(m)): 
            if k == len(m) - 1: 
                plt.plot(p, q[n], label = "v-$\\rho$ relation order = " f"{m[n]}")
            
                if n == len(m) - 1: 
                    plt.text(205, max(q[n]) * 0.05, "$\\rho_{max}$ = " f"{p_max[0]}", fontsize = 12)

                plt.scatter( p_max[0] * ((m[n] + 1) ** (-1 / m[n])), max(q[n]), color='black', s = 20, zorder = 5)
                plt.text( p_max[0] * ((m[n] + 1) ** (-1 / m[n])) - 22.5, max(q[n]) + q[n,4] * 35 , "$q_{max}$ = " f"{int(max(q[n]))}", fontsize = 10)

                plt.xlim(0,p_max[0] * 1.5)
                plt.ylim(0,max(q[n]) * 1.2)
    plt.text(max(p_max) * 1.075, max(q[n]) * .45, "$v_{max}$ = " f"{v_max[l]}", fontsize = 16)
    plt.axvline(p_max[0], - (v_max[2] * 2), v_max[2] * 2, color = 'black')
    plt.legend(framealpha = 1, fontsize = 8.5)
        
plt.show()

#%% 7
""" LWR & NL-LWR Wavespeed-Density : p_max fixed """

# Initialise wavespeed c
c = np.zeros((len(m),len(p)))
c = np.ndarray((len(m),len(p)))

# Plot NL-LWR wavespeed-density for varied m, v_max
for l in range(len(v_max)): 
    fig = plt.figure(dpi = 500)
    plt.xlabel("Density, $\\rho$  (vehicles per mile)", fontsize = 14)
    plt.ylabel("Wavespeed (mph)", fontsize = 14)
    plt.tick_params(axis = 'both', labelsize = 12)
    plt.grid(color='lightgray', linestyle='dashed')
    for k in range (len(m)):
        for j in range(len(p)):
            c[k,j] = v_max[l] * (1 - (m[k] + 1) * ( p[j] / p_max[math.floor(len(p_max) / 2)] ) ** m[k] )
            if abs(c[k,j]) > v_max[l]:
                if c[k,j] < 0:
                    c[k,j] = -v_max[l]
                
        for n in range(len(m)):
            if k == len(m) - 1: 
                plt.plot(p, c[n], label = "v-$\\rho$ relation order:  " f"{m[n]}") 
            if n == len(m) - 1: 
                plt.text(205, v_max[l] * .25, "$\\rho_{max}$ = " f"{p_max[0]}", fontsize = 12)
                plt.text(3, v_max[l] * 1.125, "$v_{max}$ = " f"{v_max[l]}", fontsize = 12)
                plt.text(2, -v_max[l] * .875, "$-v_{max}$ = " f"{v_max[l]}", fontsize = 12)
                plt.text(3, -v_max[l] * .175, "c = 0", fontsize = 12)
            
            plt.xlim(0,p_max[0] * 1.3)
            plt.ylim(-(max(c[n]) * 1.2), 0.01 + max(c[n]) * 1.5)
    plt.legend(framealpha = 1, loc = 'upper right', fontsize = 10)
    plt.axhline(max(c[k]), 0, p_max[0] * 2, color = 'black')
    plt.axhline(c[k,j], 0, p_max[0] * 2, color = 'black')
    plt.axhline(0, 0, p_max[0] * 2, color = 'grey')
    plt.axvline(p_max[0], - (v_max[l] * 2), v_max[l] * 2, color = 'black')

plt.show()
