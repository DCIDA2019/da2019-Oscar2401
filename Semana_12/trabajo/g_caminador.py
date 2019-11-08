import numpy as np
import math as mt
import matplotlib.pyplot as plt
import scipy 
from scipy.integrate import trapz,simps
import scipy.optimize as op
from colossus.cosmology import cosmology

# FUNCIONES A USAR:
#==========================================================================

#Modelo:
def model_pk(Om,b2,beta,k,z=0.57):
    cosmo = cosmology.setCosmology('planck15',)
    cosmo.Om0 = Om
    return (1+beta)*b2**2*cosmo.matterPowerSpectrum(k,z) #función

#==========================================================================

#Funcion para calcular la chi cuadrada
def chisq_pk(theta,data):
    om, b2, beta = theta
    #Nuestros datos
    x, y, yerr = data
    sigma2 = yerr**2 
    #Sumamos para encontrar el logaritmo del likelihood
    return -0.5*np.sum((y-model_pk(om,b2,beta,x,z=0.57))**2/sigma2 + np.log(sigma2))

#==========================================================================

#Funcion para calcular la chi cuadrada 
def chisq_min_pk(theta,data):
    om, b2, beta = theta
    #Nuestros datos
    x, y, yerr = data
    sigma2 = yerr**2 
    #Sumamos para encontrar el logaritmo del likelihood
    return 0.5*np.sum((y-model_pk(om,b2,beta,x,z=0.57))**2/sigma2 + np.log(sigma2)) 

#==========================================================================

# Función Prior
def log_prior(theta):
    om, b2, beta = theta
    if 0.05 < om < 1.0 and 0.05 < b2 < 5.0 and 0.001 < beta < 1.0:
        return 0.0
    return -np.inf

#==========================================================================
def metrop_pk(om0,b0,beta0,data,sigm,w,ite):
    np.random.seed(1)

    fp=open('caminadores.dat',"w")

    #Primer ajuste con función de python
    ini_points =  np.array([om0,b0,beta0])
    solu = op.minimize(chisq_min_pk, ini_points, args=data, method='L-BFGS-B',bounds=((0,1),(0,5),(0,1)))

    #Inicializamos los caminadores 
    om_ini = np.array([np.random.normal(solu.x[0],1/100) for y in range(w)])
    b_ini = np.array([np.random.normal(solu.x[1],1/100) for y in range(w)])
    beta_ini = np.array([np.random.normal(solu.x[2],1/100) for y in range(w)])

    #Reordenamos el arreglo de a y b iniciales.
    om_ini = om_ini.T.reshape((len(om_ini),1))
    b_ini = b_ini.T.reshape((len(b_ini),1))
    beta_ini = beta_ini.T.reshape((len(beta_ini),1))
    data = np.array(data)
    
    ch_ini = [] #iniciamos lista para los valores de chi iniciales.
    
    #Llenamos chi_ini con los valores iniciales 
    for i in range(len(om_ini)):
        ch_ini.append(chisq_pk([om_ini[i][0],b_ini[i][0],beta_ini[i][0]],data)+ log_prior([om_ini[i][0],b_ini[i][0],beta_ini[i][0]])) #prior
    
    #Transformamos a_ini de array a list para usar la función append.
    om = om_ini.tolist()
    b = b_ini.tolist()
    beta = beta_ini.tolist()
    
    for i in range(len(om_ini)):
        ch_0 = ch_ini[i] 
        k = 0
        for j in range(ite): 
            om_af = np.random.normal(om[i][k],sigm) #creamos un valor de a y b aleatorios
            b_af = np.random.normal(b[i][k],sigm) #calculamos el logaritmo del likelihood de para los parametros aleatorios
            beta_af = np.random.normal(beta[i][k],sigm) 
            
            lg = log_prior([om_af,b_af,beta_af])
            
            if lg != 0:
                ch = lg
            else:
                ch = chisq_pk([om_af,b_af,beta_af],data) + lg
            
            if ch > ch_0: #Comparamos las dos dos logaritmos 
                om[i].append(om_af) #guardamos los parametros creados 
                b[i].append(b_af)
                beta[i].append(beta_af)
                fp.write("%f \t%f \t%f \n" % (om_af,b_af,beta_af))
                k = k+1
                ch_0 = ch # si el nuevo logaritmo es es mayor que el anterior lo tomamos como el nuevo
            else:
                r = np.log(np.random.uniform(0,1)) 
                diff = ch-ch_0
                if diff > r: #Si la diferencia entre los logaritmos (anterior y nuevo) es menor al dicho valor
                    om[i].append(om_af)
                    b[i].append(b_af)
                    beta[i].append(beta_af)
                    fp.write("%f \t%f \t%f \n" % (om_af,b_af,beta_af))
                    k = k+1
                    ch_0 = ch #Tomamos el nuevo logaritmo
                else:  #si la direfencia es mayor, sólo guardamos los valores de a y b creados
                    om[i].append(om[i][k]) 
                    b[i].append(b[i][k])
                    beta[i].append(beta[i][k])
                    fp.write("%f \t%f \t%f \n" % (om[i][k],b[i][k],beta[i][k]))
                    k = k+1
            
    fp.close()
    # Parte de grafico
    plt.figure(figsize=(8,6))    
    
    #Gráfica de todos los puntos
    for i in range(len(om_ini)):     
        plt.scatter(om[i],b[i],s=1)
    plt.ylabel('b',fontsize=18)
    plt.xlabel('$\Omega_m$',fontsize=18)
    plt.title("Varios Caminadores",fontsize=18)
    plt.savefig('caminadores.png')
    plt.show()
    
    return  om, b, beta


#CODIGO

#==========================================================================
#Cargamos los datos:
pk_cmasdr12 = np.loadtxt('/home/echeveste/Mis_trabajos/analisis_datos/da2019-Oscar2401/Semana_12/GilMarin_boss_data/post-recon/cmass/GilMarin_2016_CMASSDR12_measurement_monopole_post_recon.txt').T

#Cortamos la cantidad de datos:
li = 20
ls = len(pk_cmasdr12[0])-10
k, pk, pk_err = pk_cmasdr12[0][li:ls],pk_cmasdr12[1][li:ls],pk_cmasdr12[2][li:ls]

#Probamos el metodo:
cosmo = cosmology.setCosmology('planck15',)

data = [k,pk,pk_err]

sigma = 0.01
om_ini = 0.36
b_ini = 2
beta_ini = 0.5
ite = 50000

om_2,b_2,beta_2= metrop_pk(om_ini,b_ini,beta_ini,data,sigma,8,ite)




