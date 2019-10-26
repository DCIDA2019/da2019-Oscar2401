import numpy as np
import math as mt
import matplotlib.pyplot as plt
import scipy 
from scipy.integrate import trapz,simps
from colossus.cosmology import cosmology

# FUNCIONES A USAR:
#==========================================================================
#Modelo:

def model(Om,b2,k,z=0.57):
    cosmo = cosmology.setCosmology('planck15',)
    cosmo.Om0 = Om
    return b2**2*cosmo.matterPowerSpectrum(k,z) #función

#==========================================================================
#Función para calcular la chi cuadrada:

def chisq_2(theta,data):
    om, b2 = theta
    x, y, yerr = data
    sigma2 = yerr**2 
    #Sumamos para encontrar el logaritmo del likelihood
    return -0.5*np.sum((y-model(om,b2,x,z=0.57))**2/sigma2+np.log(sigma2))

#==========================================================================
#Función para el metodo Metropolis:

def metrop_2(om_ini,b_ini,data,sigm,ite):
    np.random.seed(1)
    #Reordenamos el arreglo de a y b iniciales.
    om_ini = om_ini.T.reshape((len(om_ini),1))
    b_ini = b_ini.T.reshape((len(b_ini),1))
    #f_ini = f_ini.T.reshape((len(f_ini),1))
    data = np.array(data)
    
    ch_ini = [] #iniciamos lista para los valores de chi iniciales.
    
    #Llenamos chi_ini con los valores iniciales 
    for i in range(len(om_ini)):
        ch_ini.append(chisq_2([om_ini[i][0],b_ini[i][0]],data)+ log_prior([om_ini[i][0],b_ini[i][0]])) #prior
    
    #Transformamos a_ini de array a list para usar la función append.
    om = om_ini.tolist()
    b = b_ini.tolist()
    
    for i in range(len(om_ini)):
        ch_0 = ch_ini[i] 
        k = 0
        for j in range(ite): 
            om_af = np.random.normal(om[i][k],sigm) #creamos un valor de a y b aleatorios
            b_af = np.random.normal(b[i][k],sigm) #calculamos el logaritmo del likelihood de para los parametros aleatorios
            #f_af = np.random.normal(f[i][k],sigm) 
            
            lg = log_prior([om_af,b_af])
            
            if lg != 0:
                ch = lg
            else:
                ch = chisq_2([om_af,b_af],data) + lg
            
            if ch > ch_0: #Comparamos las dos dos logaritmos 
                om[i].append(om_af) #guardamos los parametros creados 
                b[i].append(b_af)
                #f[i].append(f_af)
                k = k+1
                ch_0 = ch # si el nuevo logaritmo es es mayor que el anterior lo tomamos como el nuevo
            else:
                r = np.log(np.random.uniform(0,1)) 
                diff = ch-ch_0
                if diff > r: #Si la diferencia entre los logaritmos (anterior y nuevo) es menor al dicho valor
                    om[i].append(om_af)
                    b[i].append(b_af)
                    #f[i].append(f_af)
                    k = k+1
                    ch_0 = ch #Tomamos el nuevo logaritmo
                else:  #si la direfencia es mayor, sólo guardamos los valores de a y b creados
                    om[i].append(om[i][k]) 
                    b[i].append(b[i][k])
                    #f[i].append(f[i][k])
                    k = k+1
            

    # Parte de grafico
    plt.figure(figsize=(8,6))    
    
    #Gráfica de todos los puntos
    for i in range(len(om_ini)):     
        plt.scatter(om[i],b[i],s=1)
    plt.ylabel('b',fontsize=18)
    plt.xlabel('$\Omega_m$',fontsize=18)
    plt.title("Varios Caminadores",fontsize=18)
    return  om, b

#==========================================================================
#Función para solo tomar el cumulo de los caminadores y graficarlos con sus histograma:

def flags_data_1(a,b,flags):
    a_flg = np.array([])
    b_flg = np.array([])
    for i in range(len(a)): 
        a_flg = np.append(a_flg,a[i][flags:]) #Solo tomamos los valores de los parametros por arriba del número flags 
        b_flg = np.append(b_flg,b[i][flags:])
    
    flgs = np.array([a_flg,b_flg])
    n = len(flgs)
    m = []
    mp = []
    titulo = ['Histograma de $\Omega_m$', 'Histograma de b']
    
    #Sección para graficar:
    plt.figure(figsize=(10,10))  
    for i in range(n):
        for j in range(i+1):
            k = i-j
            if i == k:
                plt.subplot(n,n,((i*n)+k+1))
                hist = plt.hist(flgs[k], bins=20,facecolor='grey',alpha = 0.8)
                m.append(np.mean(hist[1]))
                mp.append(np.mean(hist[1]))
                plt.axvline(m[i],color='r',label = np.round(m[i],4))
                plt.title(titulo[i],fontsize=18)
                plt.legend();
            else:
                plt.subplot(n,n,((i*n)+k+1))
                plt.hexbin(flgs[k], flgs[i], gridsize=30, cmap='Greys')
                plt.axvline(m[k],color='r',label = np.round(m[k],4))
                plt.axhline(m[i],color='b',label = np.round(m[i],4))
                plt.legend();
    
    return a_flg , b_flg, m

#CODIGO

#==========================================================================
#Cargamos los datos:
pk_cmasdr12 = np.loadtxt('GilMarin_boss_data/post-recon/cmass/GilMarin_2016_CMASSDR12_measurement_monopole_post_recon.txt').T

#Cortamos la cantidad de datos:
li = 20
ls = len(pk_cmasdr12[0])-10
k, pk, pk_err = pk_cmasdr12[0][li:ls],pk_cmasdr12[1][li:ls],pk_cmasdr12[2][li:ls]

#Probamos el metodo:
cosmo = cosmology.setCosmology('planck15',)

data = [k,pk,pk_err]

sigma = 0.1
om_ini = np.array([0.3,0.2,0.4]) #Tres caminadores
b_ini = np.array([0.5,0.1,0.6])
ite = 10000

om_2,b_2= metrop_2(om_ini,b_ini,data,sigma,ite)
