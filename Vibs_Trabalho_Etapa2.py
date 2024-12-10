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
from scipy.optimize import minimize
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
df1 = pd.read_csv('data/etapa_2/Medição Livre 6, Signal 2.csv', header=17, sep=";", decimal=',')
df1.rename(columns = {"'": 'Frequency', '(m/s^2)/N': 'FRF_real', '(m/s^2)/N.1': 'FRF_imag'}, 
           inplace = True)
df1 = df1.apply(pd.to_numeric, errors='coerce')
df1["FRF_abs"] = np.sqrt(df1['FRF_real']**2 + df1['FRF_imag']**2)
df1['Frequency'][0] = 1
df1['Frequency'] = df1['Frequency']*2*np.pi
df1["FRF_pos"] = df1['FRF_abs']/(df1['Frequency']**2)


df2 = pd.read_csv('data/etapa_2/Medição Livre 6, Signal 3.csv', header=17, sep=";", decimal=',')
df2.rename(columns = {"'": 'Frequency', '(m/s^2)/N': 'FRF_real', '(m/s^2)/N.1': 'FRF_imag'}, 
           inplace = True)
df2 = df2.apply(pd.to_numeric, errors='coerce')
df2["FRF_abs"] = np.sqrt(df2['FRF_real']**2 + df2['FRF_imag']**2)
df2['Frequency'][0] = 1
df2['Frequency'] = df2['Frequency']*2*np.pi
df2["FRF_pos"] = df2['FRF_abs']/(df2['Frequency']**2)


df3 = pd.read_csv('data/etapa_2/Medição Livre 6, Signal 4.csv', header=17, sep=";", decimal=',')
df3.rename(columns = {"'": 'Frequency', '(m/s^2)/N': 'FRF_real', '(m/s^2)/N.1': 'FRF_imag'}, 
           inplace = True)
df3 = df3.apply(pd.to_numeric, errors='coerce')
df3["FRF_abs"] = np.sqrt(df3['FRF_real']**2 + df3['FRF_imag']**2)
df3['Frequency'][0] = 1
df3['Frequency'] = df3['Frequency']*2*np.pi
df3["FRF_pos"] = df3['FRF_abs']/(df3['Frequency']**2)


#Identificação dos Picos
peaks1, _ = find_peaks(df1['FRF_pos'], height=1e-6)
peaks2, _ = find_peaks(df2['FRF_pos'], height=1e-6)
peaks3, _ = find_peaks(df3['FRF_pos'], height=2e-6)

#Funções 
def calculate_ssi(observed, calculated):
    observed_norm = (observed - np.mean(observed)) / np.std(observed)
    calculated_norm = (calculated - np.mean(calculated)) / np.std(calculated)
    similarity = np.dot(observed_norm, calculated_norm) / len(observed)
    return similarity

def meia_potencia(df, peaks):
    bandwidths = []
    for peak_idx in peaks:
        peak_value = df['FRF_pos'].iloc[peak_idx]
        meia_pot = peak_value/ np.sqrt(2)
        
        # find w1 (left side of bandwidth)
        w1 = None
        for i in range(peak_idx -1, -1, -1):
            if df['FRF_pos'].iloc[i] < meia_pot:
                w1 = df['Frequency'].iloc[i]
                break
       
        w2 = None
        for i in range(peak_idx + 1, len(df['FRF_pos'])):
            if df['FRF_pos'].iloc[i] < meia_pot:
                w2 = df['Frequency'].iloc[i]
                break
        
        if w1 is not None and w2 is not None:
            bandwidths.append((w1,w2))
        else:
            bandwidths.append((df1['Frequency'][peak_idx + (peak_idx-i)], df1['Frequency'][i]))   
    return bandwidths


# def func_erro(qsi, df_1, df_2, df_3, wi, V, w):
#     for j in range(1, len(w)):
#         soma1 = 0
#         soma2 = 0
#         soma3 = 0
#         for k in range(N):
#             term1 = V[x1, k] * V[xf1, k] / (wi[k]**2 - w[j]**2 + 1j * 2 * qsi[k] * wi[k] * w[j])
#             soma1 += term1            
            
#             term2 = V[x2, k] * V[xf2, k] / (wi[k]**2 - w[j]**2 + 1j * 2 * qsi[k] * wi[k] * w[j])
#             soma2 += term2
            
#             term3 = V[x3, k] * V[xf3, k] / (wi[k]**2 - w[j]**2 + 1j * 2 * qsi[k] * wi[k] * w[j])
#             soma3 += term3
                        
#         H12_ot[j] = soma1 * w[j]**2
#         H22_ot[j] = soma2 * w[j]**2
#         H32_ot[j] = soma3 * w[j]**2
        
#     FRF_12_ot = np.abs(H12_ot)
#     FRF_22_ot = np.abs(H22_ot)
#     FRF_32_ot = np.abs(H32_ot)
#     func_erro.FRF_12_ot = FRF_12_ot
#     func_erro.FRF_22_ot = FRF_22_ot
#     func_erro.FRF_32_ot = FRF_32_ot

#     ssi1_ot = calculate_ssi(df1['FRF_abs'], FRF_12_ot)
#     ssi2_ot = calculate_ssi(df2['FRF_abs'], FRF_22_ot)
#     ssi3_ot = calculate_ssi(df3['FRF_abs'], FRF_32_ot)
    
#     return -(ssi1_ot+ssi2_ot+ssi3_ot)/3


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
massa_vigas = 4 * (48/35) * volume_viga * densidade_al # kg


# Graus de Liberdade
N = 4

# Massas
m_b = massa_piso
m_1 = massa_piso +  massa_vigas
m_2 = massa_piso + massa_vigas
m_3 = massa_piso + massa_vigas


# Rigidez
k1 = k_eq
k2 = k_eq
k3 = k_eq


# Matriz de Rigidez e de Massa
K = np.array([[ k1,     -k1,       0,   0],
              [-k1, k1 + k2,     -k2,   0],
              [  0,     -k2, k2 + k3, -k3],
              [  0,       0,     -k3,  k3]])

Mv = np.array([m_b, m_1, m_2, m_3])
M = np.diag(Mv)


# Autovalores e Autovetores
W, V = np.linalg.eig(np.linalg.inv(M) @ K)

idx = np.argsort(W)
W = W[idx]
V = V[:, idx]

for i in range(V.shape[1]):
    norm_factor = np.sqrt(V[:, i].T @ M @ V[:, i])
    V[:, i] /= norm_factor
   
W[0] = 0


# Frequências Naturais
wi = np.sqrt(W)
fi = wi / (2 * np.pi)


# # Frequências Analisadas
f = np.arange(0, 201, 1) 
w = 2*np.pi*f  # Angular frequency


# #Criação de Vetores
H12 = np.zeros_like(w, dtype=complex)
x1 = 1  
xf1 = 2

H22 = np.zeros_like(w, dtype=complex)
x2 = 2
xf2 = 2

H32 = np.zeros_like(w, dtype=complex)
x3 = 3
xf3 = 2

Hm1 = np.zeros((len(w), N), dtype=complex)
Hm2 = np.zeros((len(w), N), dtype=complex)
Hm3 = np.zeros((len(w), N), dtype=complex)

H12_ot = np.zeros_like(w, dtype=complex)
H22_ot = np.zeros_like(w, dtype=complex)
H32_ot = np.zeros_like(w, dtype=complex)

qsi_banda = np.zeros(N)


#Cálculo dos qsi
for k in range(len(peaks1)):
    bands = meia_potencia(df1, peaks1)[k]
    qsi_banda[k+1] = (bands[1] - bands[0])/(2*df1['Frequency'][peaks1[k]])

qsi = qsi_banda

#Cálculos das FRFs
for j in range(5,len(w)):
    soma1 = 0
    soma2 = 0
    soma3 = 0
    for k in range(N):
        term1 = V[x1, k] * V[xf1, k] / (wi[k]**2 - w[j]**2 + 1j * 2 * qsi[k] * wi[k] * w[j])
        soma1 += term1
        Hm1[j,k] = V[x1, k] * V[xf1, k] / (wi[k]**2 - w[j]**2 + 1j * 2 * qsi[k] * wi[k]*w[j])
        
        
        term2 = V[x2, k] * V[xf2, k] / (wi[k]**2 - w[j]**2 + 1j * 2 * qsi[k] * wi[k] * w[j])
        soma2 += term2
        Hm2[j,k] = V[x2, k] * V[xf2, k] / (wi[k]**2 - w[j]**2 + 1j * 2 * qsi[k] * wi[k]*w[j])
        
        term3 = V[x3, k] * V[xf3, k] / (wi[k]**2 - w[j]**2 + 1j * 2 * qsi[k] * wi[k] * w[j])
        soma3 += term3
        Hm3[j,k] = V[x3, k] * V[xf3, k] / (wi[k]**2 - w[j]**2 + 1j * 2 * qsi[k] * wi[k]*w[j])
        
    H12[j] = soma1
    H22[j] = soma2
    H32[j] = soma3


Hm12 = np.abs(Hm1)
Hm22 = np.abs(Hm2)
Hm32 = np.abs(Hm3)

FRF_12 = np.abs(H12)
FRF_22 = np.abs(H22)
FRF_32 = np.abs(H32)


# #Cálculo das similaridades
# ssi1 = calculate_ssi(df1['FRF_pos'], FRF_12)
# ssi2 = calculate_ssi(df2['FRF_pos'], FRF_22)
# ssi3 = calculate_ssi(df3['FRF_pos'], FRF_32)
# print((ssi1+ssi2+ssi3)/3)


# #Otimização
# res = minimize(func_erro, qsi_banda, args=(df1, df2, df3, wi, V, w), method='L-BFGS-B', bounds=[(0, 0.99)] * N)
# print(res.fun*-1)
# qsi_otimizado = res.x
# FRF_12_otimizado = func_erro.FRF_12_ot
# FRF_22_otimizado = func_erro.FRF_22_ot
# FRF_32_otimizado = func_erro.FRF_32_ot


FRF_32_rms = np.zeros_like(FRF_32)
FRF_32_rms_exp = np.zeros_like(FRF_32)
FRF_sum = 0
FRF_sum_exp = 0
F0 = 1 #Força unitária do delta de Dirac

for k in range(5, len(w)):
    FRF_32_rms[k] = FRF_32[k] ** 2
    FRF_sum += FRF_32_rms[k]*2*np.pi #Passo delta_w é 2*pi
    FRF_32_rms_exp[k] = df3['FRF_pos'][k]**2
    FRF_sum_exp += FRF_32_rms_exp[k]*2*np.pi
    
    
    
x_rms = F0 * np.sqrt(FRF_sum)
x_rms_exp = F0 * np.sqrt(FRF_sum_exp)
print(x_rms, x_rms_exp)


# Plot da FRF_12
plt.figure(1)
#plt.semilogy(w, FRF_12_otimizado, '--r', linewidth=2, label = 'Otimizado')
plt.semilogy(w[5:], FRF_12[5:], '--k', linewidth=2, label= 'Modelo')
plt.semilogy(df1['Frequency'][5:], df1['FRF_pos'][5:], '--b', linewidth=2, label = 'Experimental')
plt.grid()
plt.xlabel(r'$\omega$ [rad/s]', fontsize=20)
plt.ylabel(r'$|H_{12}(\omega)|$', fontsize=20)
plt.legend()
plt.show()


# Plot da FRF_22
plt.figure(2)
# #plt.semilogy(w, FRF_22_otimizado, '--r', linewidth=2, label = 'Otimizado')
plt.semilogy(w[5:], FRF_22[5:], '--k', linewidth=2, label= 'Modelo')
plt.semilogy(df2['Frequency'][5:], df2['FRF_pos'][5:], '--b', linewidth=2, label = 'Experimental')
plt.grid()
plt.xlabel(r'$\omega$ [rad/s]', fontsize=20)
plt.ylabel(r'$|H_{22}(\omega)|$', fontsize=20)
plt.legend()
plt.show()


# Plot da FRF_32
plt.figure(3)
#plt.semilogy(w, FRF_32_otimizado, '--r', linewidth=2, label = 'Otimizado')
plt.semilogy(w[5:], FRF_32[5:], '--k', linewidth=2, label= 'Modelo')
plt.semilogy(df3['Frequency'][5:], df3['FRF_pos'][5:], '--b', linewidth=2, label = 'Experimental')
plt.grid()
plt.xlabel(r'$\omega$ [rad/s]', fontsize=20)
plt.ylabel(r'$|H_{32}(\omega)|$', fontsize=20)
plt.legend()
plt.show()


# # Plot dos Modos na FRF12
# plt.figure(4)
# plt.semilogy(w, Hm12[:, 1], '--g', linewidth=2, label='Mode 1')
# plt.semilogy(w, Hm12[:, 2], '--r', linewidth=2, label='Mode 2')
# plt.semilogy(w, Hm12[:, 3], '--b', linewidth=2, label='Mode 3')
# plt.semilogy(w, FRF_12, 'k', linewidth=2, label='Total')
# plt.grid()
# plt.xlabel(r'$\omega$ [rad/s]', fontsize=14)
# plt.ylabel(r'$|H_{ij}(\omega)|$', fontsize=14)
# plt.legend()
# plt.show()


# # Plot dos Modos na FRF22
# plt.figure(5)
# plt.semilogy(w, Hm22[:, 1], '--g', linewidth=2, label='Mode 1')
# plt.semilogy(w, Hm22[:, 2], '--r', linewidth=2, label='Mode 2')
# plt.semilogy(w, Hm22[:, 3], '--b', linewidth=2, label='Mode 3')
# plt.semilogy(w, FRF_22, 'k', linewidth=2, label='Total')
# plt.grid()
# plt.xlabel(r'$\omega$ [rad/s]', fontsize=14)
# plt.ylabel(r'$|H_{ij}(\omega)|$', fontsize=14)
# plt.legend()
# plt.show()


# # Plot dos Modos na FRF32
# plt.figure(6)
# plt.semilogy(w, Hm32[:, 1], '--g', linewidth=2, label='Mode 1')
# plt.semilogy(w, Hm32[:, 2], '--r', linewidth=2, label='Mode 2')
# plt.semilogy(w, Hm32[:, 3], '--b', linewidth=2, label='Mode 3')
# plt.semilogy(w, FRF_32, 'k', linewidth=2, label='Total')
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

# plt.figure(7)
# plt.semilogy(df1['Frequency'], df1['FRF_abs'], 'k', linewidth=2)
# plt.semilogy(w, np.abs(H1)*w**2, '-r', linewidth=2, label=r'$\xi = 0.01$')
# plt.semilogy(w, np.abs(H2)*w**2, '--g', linewidth=2, label=r'$\xi = 0.05$')
# plt.semilogy(w, np.abs(H3)*w**2, '-.b', linewidth=2, label=r'$\xi = 0.1$')
# plt.grid()
# plt.xlabel(r'$\omega$ [rad/s]', fontsize=14)
# plt.ylabel(r'$|H_{ij}(\omega)|$', fontsize=14)
# plt.legend()
# plt.show()
