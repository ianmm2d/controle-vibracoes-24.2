# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:01:38 2024

@author: PC
"""
## ============================================================================
## IMPORTS
## ============================================================================
import locale
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import matplotlib.font_manager as font_manager
from scipy.sparse.linalg import spsolve
import pandas as pd
from scipy.optimize import fsolve
from scipy.signal import find_peaks
## ============================================================================
## RC parameters
## ============================================================================
locale.setlocale(locale.LC_NUMERIC,'de_DE')
mpl.rcParams['axes.formatter.use_locale'] = False
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#303030'])*
                                   cycler(linestyle = ['-','--']))
mpl.rcParams['figure.constrained_layout.use'] = True
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'serif'
font = font_manager.FontProperties(family='serif',
                                   style='normal', size=12)

#Dataframes
df1 = pd.read_csv('Grupo - 3 9, Signal 2.csv', header=17, sep=";", decimal=',')
df1.rename(columns = {"'": 'Frequency', '(m/s^2)/N': 'FRF_real', '(m/s^2)/N.1': 'FRF_imag'}, 
           inplace = True)
df1 = df1.apply(pd.to_numeric, errors='coerce')
df1["FRF_abs"] = np.sqrt(df1['FRF_real']**2 + df1['FRF_imag']**2)
df1['Frequency'] = df1['Frequency']*2*np.pi


df2 = pd.read_csv('Grupo - 3 9, Signal 3.csv', header=17, sep=";", decimal=',')
df2.rename(columns = {"'": 'Frequency', '(m/s^2)/N': 'FRF_real', '(m/s^2)/N.1': 'FRF_imag'}, 
           inplace = True)
df2 = df2.apply(pd.to_numeric, errors='coerce')
df2["FRF_abs"] = np.sqrt(df2['FRF_real']**2 + df2['FRF_imag']**2)
df2['Frequency'] = df2['Frequency']*2*np.pi


df3 = pd.read_csv('Grupo - 3 9, Signal 4.csv', header=17, sep=";", decimal=',')
df3.rename(columns = {"'": 'Frequency', '(m/s^2)/N': 'FRF_real', '(m/s^2)/N.1': 'FRF_imag'}, 
           inplace = True)
df3 = df3.apply(pd.to_numeric, errors='coerce')
df3["FRF_abs"] = np.sqrt(df3['FRF_real']**2 + df3['FRF_imag']**2)
df3['Frequency'] = df3['Frequency']*2*np.pi

peaks1, _ = find_peaks(df1['FRF_abs'], height=1e-1)
peaks2, _ = find_peaks(df2['FRF_abs'], height=1e-1)
peaks3, _ = find_peaks(df3['FRF_abs'], height=1e-1)

# Propriedades do Material
densidade_al = 2700 # kg/m^3
E = 69e9 # Pa

# Dimensões do Piso
D = 300e-3 # m
L = 300e-3 # m
H = 25.4e-3 # m

# Dimensões da viga:
d = 25.4 * 1e-3 # m
l = 6 * 1e-3 # m
h = 150 * 1e-3 # m

# Informações do piso:
volume_piso = (D * L * H) # m^3
massa_piso = densidade_al * volume_piso # kg

# Informações da viga
I_viga = (d * l ** 3)/12
k_viga = (12*E*I_viga)/((h + H/2) ** 3)
k_eq = 4 * k_viga

volume_viga = (d * l * h) #m^3
massa_vigas = (48/35) * volume_viga * densidade_al # kg

# Parameters
N = 3

# Masses
m_e1 = 646e-3 # kg
m_e2 = 1187e-3 # kg
m_e3 = 1635e-3 # kg

m_1 = massa_piso + m_e1 #+ massa_vigas
m_2 = massa_piso + m_e2 #+ massa_vigas
m_3 = massa_piso + m_e3 #+ massa_vigas

# Stiffnesses
k1 = k_eq
k2 = k_eq
k3 = k_eq

# Mass and stiffness matrices
K = np.array([[k1 + k2, -k2    , 0  ],
              [-k2    , k2 + k3, -k3],
              [0      , -k3    , k3]])

Mv = np.array([m_1, m_2, m_3])
M = np.diag(Mv)

# Eigenvalues and eigenvectors
W, V = np.linalg.eig(np.linalg.inv(M) @ K)

idx = np.argsort(W)
W = W[idx]
V = V[:, idx]

for i in range(V.shape[1]):
    norm_factor = np.sqrt(V[:, i].T @ M @ V[:, i])
    V[:, i] /= norm_factor
    
# Natural frequencies
wi = np.sqrt(W)
fi = wi / (2 * np.pi)

# Frequency response functions
f = np.arange(0, 201, 1) 
w = 2*np.pi*f  # Angular frequency


# def mean_absolute_error(measured, predicted):
#     mae = np.mean(np.abs(np.array(measured) - np.array(predicted)))
#     return mae

def calcular_mse(curva_exp, curva_sim):
    mse = np.mean((np.array(curva_exp) - np.array(curva_sim))**2)
    return mse


# def rSquared(measured,predicted):
#     estimateError = ((predicted-measured)**2).sum()
#     meanOfMeasured = measured.sum()/len(measured)
#     variability = ((measured-meanOfMeasured)**2).sum()
#     return 1-estimateError/variability

H12 = np.zeros_like(w, dtype=complex)
x1 = 0  
xf1 = 1

H22 = np.zeros_like(w, dtype=complex)
x2 = 1
xf2 = 1

H32 = np.zeros_like(w, dtype=complex)
x3 = 2
xf3 = 1

# def meia_potência(modo, df, peaks):    
#     meia_pot = df['FRF_abs'][peaks[modo]]/np.sqrt(2)
#     df['diff'] = abs(df['FRF_abs'] - meia_pot)
#     closest_w = df.loc[df['diff'].idxmin()]
#     print(closest_w)
#     #sign_changes = np.where(np.diff(np.sign(diff)))[0]
#     # w1_1 = df['Frequency'][min(sign_changes)]
#     # w2_1 = df['Frequency'][min(sign_changes)+1]
#     # qsi = (w2_1 - w1_1)/(2*df['Frequency'][peaks[modo]])
#     return qsi

def meia_potencia(modo, df, peaks):
    bandwidths = []
    for peak_idx in peaks:
        peak_value = df['FRF_abs'].iloc[peak_idx]
        meia_pot = peak_value/ np.sqrt(2)
        
        # find w1 (left side of bandwidth)
        w1 = None
        for i in range(peak_idx -1, -1, -1):
            if df['FRF_abs'].iloc[i] < meia_pot:
                w1 = df['Frequency'].iloc[i]
                break
            
        w2 = None
        for i in range(peak_idx + 1, len(df['FRF_abs'])):
            if df['FRF_abs'].iloc[i] < meia_pot:
                w2 = df['Frequency'].iloc[i]
                break
            
        if w1 is not None and w2 is not None:
            bandwidths.append((w1,w2))
        else:
            bandwidths.append((None, None))
            
    return bandwidths

mse = 1
#Hm1 = np.zeros((len(w), N), dtype=complex)

# while mse > 1e-2:
#     i += 1

alpha = 15
beta = 1e-5

qsi = np.zeros(N)
qsi2 = np.zeros(N)
qsi3 = np.zeros(N)

for j in range(len(w)):
    soma1 = 0
    soma2 = 0
    soma3 = 0
    #qsi = 0.01
    for k in range(N):
        bands = meia_potencia(k, df1, peaks1)[k]
        qsi[k] = (bands[1] - bands[0])/(2*df1['Frequency'][peaks1[k]])
        # qsi1[k] = meia_potência(k, df1, peaks1)
        # qsi2[k] = meia_potência2(k)
        # qsi3[k] = meia_potência3(k)
        #qsi = (alpha + wi[k]**2*beta)/(2*wi[k])
        term1 = V[x1, k] * V[xf1, k] / (wi[k]**2 - w[j]**2 + 1j * 2 * qsi[k] * wi[k] * w[j])
        soma1 += term1
        #Hm1[j,k] = V[x1, k] * V[xf1, k] / (wi[k]**2 - w[j]**2 + 1j * 2 * qsi * wi[k]*w[j])
        
        # term2 = V[x2, k] * V[xf2, k] / (wi[k]**2 - w[j]**2 + 1j * 2 * qsi2[k] * wi[k] * w[j])
        # soma2 += term2
        
        # term3 = V[x3, k] * V[xf3, k] / (wi[k]**2 - w[j]**2 + 1j * 2 * qsi3[k] * wi[k] * w[j])
        # soma3 += term3
        
    H12[j] = soma1 * w[j]**2
    H22[j] = soma2
    H32[j] = soma3



#Hm12 = np.abs(Hm1)*w[:, None]**2
FRF_12 = np.abs(H12)
FRF_22 = np.abs(H22)
FRF_32 = np.abs(H32)
mse = calcular_mse(df1['FRF_abs'][2:120], FRF_12[2:120])
print(mse)

# Plot the magnitude of H
plt.figure(1)
plt.semilogy(w[2:120], FRF_12[2:120], 'k', linewidth=2)
plt.semilogy(df1['Frequency'][2:120], df1['FRF_abs'][2:120], linewidth=2)
#plt.semilogy(w, teste)
plt.grid()
plt.xlabel(r'$\omega$ [rad/s]', fontsize=20)
plt.ylabel(r'$|H_{ij}(\omega)|$', fontsize=20)
plt.show()

# plt.figure(2)
# plt.semilogy(w[2:120], FRF_22[2:120], 'k', linewidth=2)
# plt.semilogy(df2['Frequency'][2:120], df2['FRF_abs'][2:120], linewidth=2)
# plt.grid()
# plt.xlabel(r'$\omega$ [rad/s]', fontsize=20)
# plt.ylabel(r'$|H_{ij}(\omega)|$', fontsize=20)
# plt.show()

# plt.figure(3)
# plt.semilogy(w[2:120], FRF_32[2:120], 'k', linewidth=2)
# plt.semilogy(df3['Frequency'][2:120], df3['FRF_abs'][2:120], linewidth=2)
# plt.grid()
# plt.xlabel(r'$\omega$ [rad/s]', fontsize=20)
# plt.ylabel(r'$|H_{ij}(\omega)|$', fontsize=20)
# plt.show()

# # Plot the phase of H
# plt.figure(2)
# plt.plot(w, np.angle(H), 'k', linewidth=2)
# plt.grid()
# plt.xlabel(r'$\omega$ [rad/s]', fontsize=20)
# plt.ylabel(r'$\phi_{ij}(\omega)$ [rad]', fontsize=20)
# plt.show()

# #Plot magnitude of individual mode responses
# plt.figure(3)
# #plt.semilogy(w, FRF_12[:,1], '--r', linewidth=2, label='Mode 1')
# plt.semilogy(w, Hm12[:, 0], '--g', linewidth=2, label='Mode 1')
# plt.semilogy(w, Hm12[:, 1], '--r', linewidth=2, label='Mode 2')
# plt.semilogy(w, Hm12[:, 2], '--b', linewidth=2, label='Mode 3')
# plt.semilogy(w, FRF_12, 'k', linewidth=2, label='Total')
# plt.grid()
# plt.xlabel(r'$\omega$ [rad/s]', fontsize=14)
# plt.ylabel(r'$|H_{ij}(\omega)|$', fontsize=14)
# plt.legend()
# plt.show()

# # Frequency response for different damping ratios
# qsi1 = 0.01
# qsi2 = 0.05
# qsi3 = 0.1
# H1 = np.zeros_like(w, dtype=complex)
# H2 = np.zeros_like(w, dtype=complex)
# H3 = np.zeros_like(w, dtype=complex)

# for j in range(len(w)):
#     soma1, soma2, soma3 = 0, 0, 0
#     for k in range(N):
#         soma1 += V[x1, k] * V[xf1, k] / (wi[k]**2 - w[j]**2 + 1j * 2 * qsi1 * wi[k] * w[j])
#         soma2 += V[x1, k] * V[xf1, k] / (wi[k]**2 - w[j]**2 + 1j * 2 * qsi2 * wi[k] * w[j])
#         soma3 += V[x1, k] * V[xf1, k] / (wi[k]**2 - w[j]**2 + 1j * 2 * qsi3 * wi[k] * w[j])
#     H1[j], H2[j], H3[j] = soma1, soma2, soma3

# plt.figure(4)
# plt.semilogy(df1['Frequency'], df1['FRF_abs'], 'k', linewidth=2)
# plt.semilogy(w, np.abs(H1)*w**2, '-r', linewidth=2, label=r'$\xi = 0.01$')
# plt.semilogy(w, np.abs(H2)*w**2, '--g', linewidth=2, label=r'$\xi = 0.05$')
# plt.semilogy(w, np.abs(H3)*w**2, '-.b', linewidth=2, label=r'$\xi = 0.1$')
# plt.grid()
# plt.xlabel(r'$\omega$ [rad/s]', fontsize=14)
# plt.ylabel(r'$|H_{ij}(\omega)|$', fontsize=14)
# plt.legend()
# plt.show()
