import numpy as np
import math as mt
import matplotlib.pyplot as plt
import scipy 
from scipy.integrate import trapz,simps
from colossus.cosmology import cosmology

# FUNCIONES A USAR:
#==========================================================================
#Modelo:

def model(Om,b2,beta,k,z=0.57):
    cosmo = cosmology.setCosmology('planck15',)
    cosmo.Om0 = Om
    return (1-beta)*b2**2*cosmo.matterPowerSpectrum(k,z) #función

#==========================================================================
#Función para solo tomar el cumulo de los caminadores y graficarlos con sus histograma
def flags_data_1(a,b,beta,flags):
    a_flg = np.array([])
    b_flg = np.array([])
    beta_flg = np.array([])
    
    
    
    for i in range(len(a)): 
        a_flg = np.append(a_flg,a[i][flags:]) #Solo tomamos los valores de los parametros por arriba del número flags 
        b_flg = np.append(b_flg,b[i][flags:])
        beta_flg = np.append(beta_flg,beta[i][flags:])
    
    flgs = np.array([a_flg,b_flg,beta_flg])
    n = len(flgs)
    m = []
    mp = []
    titulo = ['Histograma de $\Omega_m$', 'Histograma de b','Histograma de $beta$']
    
    #Sección para graficar:
    plt.figure(figsize=(10,10))  
    for i in range(n):
        for j in range(i+1):
            k = i-j
            if i == k:
                plt.subplot(n,n,((i*n)+k+1))
                hist = plt.hist(flgs[k], bins=20,facecolor='grey',alpha = 0.8)
                m.append(hist[1][np.where(hist[0]==np.max(hist[0]))])
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

    plt.savefig('graficas.png')
    plt.show()
    
    return a_flg , b_flg, beta_flg, m

#CODIGO
#==========================================================================
#Cargamos los datos:

camin = np.loadtxt('caminadores.dat')

om = camin.T[0]
om = om.reshape((8,50000))
b = camin.T[1]
b = b.reshape((8,50000))
beta = camin.T[2]
beta = beta.reshape((8,50000))

print('El número de pasos es', len(om[0]),'\n')
#print('Escoge la cantidad de los primeros pasos a eliminar => ')
flags = int(input('Escoge la cantidad de los primeros pasos a eliminar => '))
omg_flg,b_flg,beta_flg, m = flags_data_1(om,b,beta,flags)

pk_cmasdr12 = np.loadtxt('GilMarin_boss_data/post-recon/cmass/GilMarin_2016_CMASSDR12_measurement_monopole_post_recon.txt').T
li = 0
ls = len(pk_cmasdr12[0])-0
k, pk, pk_err = pk_cmasdr12[0][li:ls],pk_cmasdr12[1][li:ls],pk_cmasdr12[2][li:ls]



k_m = 10**np.linspace(-6,5,100000)
pk_m = model(m[0],m[1],m[2],k_m,z=0.57)

plt.figure()
plt.loglog()
plt.errorbar(k,pk,yerr = pk_err,fmt='.')
plt.plot(k_m,pk_m,'-')
plt.xlim(1e-3,0.8)
plt.ylim(1500,3e5)
plt.savefig('ajuste.png')
plt.show()





