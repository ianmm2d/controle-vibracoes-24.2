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
df1 = pd.read_csv('Medição Livre 6, Signal 2.csv', header=17, sep=";", decimal=',')
df1.rename(columns = {"'": 'Frequency', '(m/s^2)/N': 'FRF_real', '(m/s^2)/N.1': 'FRF_imag'}, 
           inplace = True)
df1 = df1.apply(pd.to_numeric, errors='coerce')
df1["FRF_abs"] = np.sqrt(df1['FRF_real']**2 + df1['FRF_imag']**2)
df1['Frequency'][0] = 1
df1['Frequency'] = df1['Frequency']*2*np.pi
df1["FRF_pos"] = df1['FRF_abs']/(df1['Frequency']**2)


df2 = pd.read_csv('Medição Livre 6, Signal 3.csv', header=17, sep=";", decimal=',')
df2.rename(columns = {"'": 'Frequency', '(m/s^2)/N': 'FRF_real', '(m/s^2)/N.1': 'FRF_imag'}, 
           inplace = True)
df2 = df2.apply(pd.to_numeric, errors='coerce')
df2["FRF_abs"] = np.sqrt(df2['FRF_real']**2 + df2['FRF_imag']**2)
df2['Frequency'][0] = 1
df2['Frequency'] = df2['Frequency']*2*np.pi
df2["FRF_pos"] = df2['FRF_abs']/(df2['Frequency']**2)


df3 = pd.read_csv('Medição Livre 6, Signal 4.csv', header=17, sep=";", decimal=',')
df3.rename(columns = {"'": 'Frequency', '(m/s^2)/N': 'FRF_real', '(m/s^2)/N.1': 'FRF_imag'}, 
           inplace = True)
df3 = df3.apply(pd.to_numeric, errors='coerce')
df3["FRF_abs"] = np.sqrt(df3['FRF_real']**2 + df3['FRF_imag']**2)
df3['Frequency'][0] = 1
df3['Frequency'] = df3['Frequency']*2*np.pi
df3["FRF_pos"] = df3['FRF_abs']/(df3['Frequency']**2)


#Plots de figuras
fig1, ax1 = plt.subplots(1,1)
fig2, ax2 = plt.subplots(1,1)
fig3, ax3 = plt.subplots(1,1)
fig4, ax4 = plt.subplots(1,1)
fig5, ax5 = plt.subplots(1,1)
fig6, ax6 = plt.subplots(1,1)
fig7, ax7 = plt.subplots(1,1)
fig8, ax8 = plt.subplots(1,1)
fig9, ax9 = plt.subplots(1,1)
fig10, ax10 = plt.subplots(1,1)


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


def func_err(Mv, df_1, df_2, df_3, wi, V, w):
    M_ot = np.diag(Mv)
    # K_eq_ot = k_eq_var[0]
    # K_ot = np.array([[ K_eq_ot,     -K_eq_ot,       0,   0],
    #               [-K_eq_ot, K_eq_ot + K_eq_ot,     -K_eq_ot,   0],
    #               [  0,     -K_eq_ot, K_eq_ot + K_eq_ot, -K_eq_ot],
    #               [  0,       0,     -K_eq_ot,  K_eq_ot]])
    W_ot, V_ot = np.linalg.eig(np.linalg.inv(M_ot) @ K)
    idx = np.argsort(W_ot)
    W_ot = W_ot[idx]
    V_ot = V_ot[:, idx]
    for i in range(V_ot.shape[1]):
        norm_factor = np.sqrt(V_ot[:, i].T @ M_ot @ V_ot[:, i])
        V_ot[:, i] /= norm_factor  
    W_ot[0] = 0
    wi_ot = np.sqrt(W_ot)
    for j in range(5, len(w)):
        soma3_ot = 0
        for k in range(N):
            term3_ot = V_ot[x3, k] * V_ot[xf3, k] / (wi_ot[k]**2 - w[j]**2 + 1j * 2 * qsi[k] * wi_ot[k] * w[j])
            soma3_ot += term3_ot
                        
        H32_ot[j] = soma3_ot
        
    FRF_32_ot = np.abs(H32_ot)
    FRF_32_rms_ot = np.zeros_like(FRF_32_ot)
    FRF_sum_ot = 0
    for k in range(5, len(w)):
        FRF_32_rms_ot[k] = FRF_32_ot[k] ** 2
        FRF_sum_ot += FRF_32_rms_ot[k]*2*np.pi #Passo delta_w é 2*pi
    
    func_err.FRF_32_ot = FRF_32_ot
    x_rms_ot = F0 * np.sqrt(FRF_sum_ot)
    return x_rms_ot

def erro(k_eq_var, df_1, df_2, df_3, wi, V, w):
    K_eq_ot = k_eq_var[0]
    K_ot = np.array([[ K_eq_ot,     -K_eq_ot,       0,   0],
                  [-K_eq_ot, K_eq_ot + K_eq_ot,     -K_eq_ot,   0],
                  [  0,     -K_eq_ot, K_eq_ot + K_eq_ot, -K_eq_ot],
                  [  0,       0,     -K_eq_ot,  K_eq_ot]])
    W_ot, V_ot = np.linalg.eig(np.linalg.inv(M) @ K_ot)
    idx = np.argsort(W_ot)
    W_ot = W_ot[idx]
    V_ot = V_ot[:, idx]
    for i in range(V_ot.shape[1]):
        norm_factor = np.sqrt(V_ot[:, i].T @ M @ V_ot[:, i])
        V_ot[:, i] /= norm_factor  
    W_ot[0] = 0
    wi_ot = np.sqrt(W_ot)
    for j in range(5, len(w)):
        soma3_ot = 0
        for k in range(N):
            term3_ot = V_ot[x3, k] * V_ot[xf3, k] / (wi_ot[k]**2 - w[j]**2 + 1j * 2 * qsi[k] * wi_ot[k] * w[j])
            soma3_ot += term3_ot
                        
        H32_ot[j] = soma3_ot
        
    FRF_32_ot_k = np.abs(H32_ot)
    FRF_32_rms_ot = np.zeros_like(FRF_32_ot_k)
    FRF_sum_ot = 0
    for k in range(5, len(w)):
        FRF_32_rms_ot[k] = FRF_32_ot_k[k] ** 2
        FRF_sum_ot += FRF_32_rms_ot[k]*2*np.pi #Passo delta_w é 2*pi
    
    erro.FRF_32_ot_k = FRF_32_ot_k
    x_rms_ot = F0 * np.sqrt(FRF_sum_ot)
    return x_rms_ot

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
m_1 = massa_piso + massa_vigas
m_2 = massa_piso + massa_vigas
m_3 = massa_piso + massa_vigas 


# Rigidez
k1 = k_eq
k2 = k_eq
k3 = k_eq


F0 = 1 #Força unitária do delta de Dirac


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


#Cálculo das similaridades
ssi1 = calculate_ssi(df1['FRF_pos'][5:], FRF_12[5:])
ssi2 = calculate_ssi(df2['FRF_pos'][5:], FRF_22[5:])
ssi3 = calculate_ssi(df3['FRF_pos'][5:], FRF_32[5:])
#print((ssi1+ssi2+ssi3)/3)


#Otimização
res_1 = minimize(func_err, Mv, args=(df1, df2, df3, wi, V, w), method='Nelder-Mead', bounds=[(m_b/2, 3*m_b/2)] * N)#'Nelder-Mead'))
# print(res_1.x)
# print(res_1.fun)

res_2 = minimize(erro, k_eq, args=(df1, df2, df3, wi, V, w), method='Nelder-Mead', bounds=[(k_eq, 3*k_eq)])
# print(res_2.x)
# print(res_2.fun)

FRF_32_otimizado = func_err.FRF_32_ot
FRF_32_otimizado_k = erro.FRF_32_ot_k
FRF_32_rms = np.zeros_like(FRF_32)
FRF_32_rms_exp = np.zeros_like(FRF_32)
FRF_sum = 0
FRF_sum_exp = 0


for k in range(5, len(w)):
    FRF_32_rms[k] = FRF_32[k] ** 2
    FRF_sum += FRF_32_rms[k]*2*np.pi #Passo delta_w é 2*pi
    FRF_32_rms_exp[k] = df3['FRF_pos'][k]**2
    FRF_sum_exp += FRF_32_rms_exp[k]*2*np.pi
    
    
x_rms = F0 * np.sqrt(FRF_sum)
x_rms_exp = F0 * np.sqrt(FRF_sum_exp)
print(x_rms, x_rms_exp)

GL = np.zeros(N)
for i in range(N):
    GL[i] = i
 
V0 = [V[0][0], V[1][0], V[2][0], V[3][0]]    
V1 = [V[0][1], V[1][1], V[2][1], V[3][1]]
V2 = [V[0][2], V[1][2], V[2][2], V[3][2]]
V3 = [V[0][3], V[1][3], V[2][3], V[3][3]]


# Plot da FRF_12
ax1.semilogy(w[5:], FRF_12[5:], '--k', linewidth=2, label= 'Modelo')
ax1.semilogy(df1['Frequency'][5:], df1['FRF_pos'][5:], '--b', linewidth=2, label = 'Experimental')
ax1.grid()
ax1.set_xlabel(r'$\omega$ [rad/s]', fontsize=20)
ax1.set_ylabel(r'$|H_{12}(\omega)|$', fontsize=20)
ax1.legend()
fig1.savefig('FRF_pos12_Etapa2.jpg', dpi = 300)


# Plot da FRF_22
ax2.semilogy(w[5:], FRF_22[5:], '--k', linewidth=2, label= 'Modelo')
ax2.semilogy(df2['Frequency'][5:], df2['FRF_pos'][5:], '--b', linewidth=2, label = 'Experimental')
ax2.grid()
ax2.set_xlabel(r'$\omega$ [rad/s]', fontsize=20)
ax2.set_ylabel(r'$|H_{22}(\omega)|$', fontsize=20)
ax2.legend()
fig2.savefig('FRF_pos22_Etapa2.jpg', dpi = 300)



# Plot da FRF_32
ax3.semilogy(w[5:], FRF_32_otimizado[5:], '--r', linewidth=2, label = 'Otimizado M')
ax3.semilogy(w[5:], FRF_32_otimizado_k[5:], '--c', linewidth=2, label = 'Otimizado K')
ax3.semilogy(w[5:], FRF_32[5:], '--k', linewidth=2, label= 'Modelo')
ax3.semilogy(df3['Frequency'][5:], df3['FRF_pos'][5:], '--b', linewidth=2, label = 'Experimental')
ax3.grid()
ax3.set_xlabel(r'$\omega$ [rad/s]', fontsize=20)
ax3.set_ylabel(r'$|H_{32}(\omega)|$', fontsize=20)
ax3.legend()
fig3.savefig('FRF_pos32_Etapa2.jpg', dpi = 300)


# Plot dos Modos na FRF12
ax4.semilogy(w[5:], Hm12[:, 0][5:], '--c', linewidth=2, label='Mode 0')
ax4.semilogy(w[5:], Hm12[:, 1][5:], '--g', linewidth=2, label='Mode 1')
ax4.semilogy(w[5:], Hm12[:, 2][5:], '--r', linewidth=2, label='Mode 2')
ax4.semilogy(w[5:], Hm12[:, 3][5:], '--b', linewidth=2, label='Mode 3')
ax4.semilogy(w[5:], FRF_12[5:], 'k', linewidth=2, label='Total')
ax4.grid()
ax4.set_xlabel(r'$\omega$ [rad/s]', fontsize=14)
ax4.set_ylabel(r'$|H_{12}(\omega)|$', fontsize=14)
ax4.legend()
fig4.savefig('FRF12_Modos_Etapa2.jpg', dpi = 300)


# Plot dos Modos na FRF22
ax5.semilogy(w[5:], Hm22[:, 0][5:], '--c', linewidth=2, label='Mode 0')
ax5.semilogy(w[5:], Hm22[:, 1][5:], '--g', linewidth=2, label='Mode 1')
ax5.semilogy(w[5:], Hm22[:, 2][5:], '--r', linewidth=2, label='Mode 2')
ax5.semilogy(w[5:], Hm22[:, 3][5:], '--b', linewidth=2, label='Mode 3')
ax5.semilogy(w[5:], FRF_22[5:], 'k', linewidth=2, label='Total')
ax5.grid()
ax5.set_xlabel(r'$\omega$ [rad/s]', fontsize=14)
ax5.set_ylabel(r'$|H_{22}(\omega)|$', fontsize=14)
ax5.legend()
fig5.savefig('FRF22_Modos_Etapa2.jpg', dpi = 300)


# Plot dos Modos na FRF32
ax6.semilogy(w[5:], Hm32[:, 0][5:], '--c', linewidth=2, label='Mode 0')
ax6.semilogy(w[5:], Hm32[:, 1][5:], '--g', linewidth=2, label='Mode 1')
ax6.semilogy(w[5:], Hm32[:, 2][5:], '--r', linewidth=2, label='Mode 2')
ax6.semilogy(w[5:], Hm32[:, 3][5:], '--b', linewidth=2, label='Mode 3')
ax6.semilogy(w[5:], FRF_32[5:], 'k', linewidth=2, label='Total')
ax6.grid()
ax6.set_xlabel(r'$\omega$ [rad/s]', fontsize=14)
ax6.set_ylabel(r'$|H_{32}(\omega)|$', fontsize=14)
ax6.legend()
fig6.savefig('FRF32_Modos_Etapa2.jpg', dpi = 300)

#Plot Autovetores
Circle00 = plt.Circle((GL[0], V0[0]), 0.02, color='b', fill =True)
Circle10 = plt.Circle((GL[1], V0[1]), 0.02, color='b', fill=True)
Circle20 = plt.Circle((GL[2], V0[2]), 0.02, color='b', fill=True)
Circle30 = plt.Circle((GL[3], V0[3]), 0.02, color='b', fill=True)
ax7.plot(GL, V0, '--k', linewidth= 2)
ax7.grid()
ax7.set_xlabel(r'Grau de Liberdade', fontsize= 20)
ax7.set_ylabel(r'Modo 0', fontsize= 20)
ax7.add_patch(Circle00)
ax7.add_patch(Circle10)
ax7.add_patch(Circle20)
ax7.add_patch(Circle30)
ax7.set_aspect('equal', adjustable='datalim')
fig7.savefig('Modos0_Etapa2.jpg', dpi = 300)

Circle01 = plt.Circle((GL[0], V1[0]), 0.02, color='b', fill=True)
Circle11 = plt.Circle((GL[1], V1[1]), 0.02, color='b', fill=True)
Circle21 = plt.Circle((GL[2], V1[2]), 0.02, color='b', fill=True)
Circle31 = plt.Circle((GL[3], V1[3]), 0.02, color='b', fill=True)
ax8.plot(GL, V1, '--k', linewidth= 2)
ax8.grid()
ax8.set_xlabel(r'Grau de Liberdade', fontsize= 20)
ax8.set_ylabel(r'Modo 1', fontsize= 20)
ax8.add_patch(Circle01)
ax8.add_patch(Circle11)
ax8.add_patch(Circle21)
ax8.add_patch(Circle31)
ax8.set_aspect('equal', adjustable='datalim')
fig8.savefig('Modos1_Etapa2.jpg', dpi = 300)

Circle02 = plt.Circle((GL[0], V2[0]), 0.02, color='b', fill=True)
Circle12 = plt.Circle((GL[1], V2[1]), 0.02, color='b', fill=True)
Circle22 = plt.Circle((GL[2], V2[2]), 0.02, color='b', fill=True)
Circle32 = plt.Circle((GL[3], V2[3]), 0.02, color='b', fill=True)
ax9.plot(GL, V2, '--k', linewidth= 2)
ax9.grid()
ax9.set_xlabel(r'Grau de Liberdade', fontsize= 20)
ax9.set_ylabel(r'Modo 2', fontsize= 20)
ax9.add_patch(Circle02)
ax9.add_patch(Circle12)
ax9.add_patch(Circle22)
ax9.add_patch(Circle32)
ax9.set_aspect('equal', adjustable='datalim')
fig9.savefig('Modos2_Etapa2.jpg', dpi = 300)

Circle03 = plt.Circle((GL[0], V3[0]), 0.02, color='b', fill=True)
Circle13 = plt.Circle((GL[1], V3[1]), 0.02, color='b', fill=True)
Circle23 = plt.Circle((GL[2], V3[2]), 0.02, color='b', fill=True)
Circle33 = plt.Circle((GL[3], V3[3]), 0.02, color='b', fill=True)
ax10.plot(GL, V3, '--k', linewidth= 2)
ax10.grid()
ax10.set_xlabel(r'Grau de Liberdade', fontsize= 20)
ax10.set_ylabel(r'Modo 3', fontsize= 20)
ax10.add_patch(Circle03)
ax10.add_patch(Circle13)
ax10.add_patch(Circle23)
ax10.add_patch(Circle33)
ax10.set_aspect('equal', adjustable='datalim')
fig10.savefig('Modos3_Etapa2.jpg', dpi = 300)


