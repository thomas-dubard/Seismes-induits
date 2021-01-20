import numpy as np

from init import *

##Calcul du gradient de la transformée de Hilbert (cf. doc)
def noyau(freq: np.ndarray,H: float) -> np.ndarray:
    """
    Optimisée d'un facteur 100 par Quentin Guitet
    """
    pos = freq==0 #on localise la position des zéros
    freq[pos] = 1 #On remplace les zéros par des non-zéros (1 par exemple)
    c = np.abs(freq)
    freq = c/np.tanh(2/H/c) #on applique l'opération (sans rencontrer de zéros donc)
    freq[pos] = 0 #on met des zéros aux positions des zéros du tableau de départ
    return freq

def Psi(f: np.ndarray, deltaX: float, H: float) -> np.ndarray:
    F=np.fft.fft(f)
    freq=np.fft.fftfreq(len(f),deltaX)
    K = noyau(freq, H)

    F=K*2*np.pi*F

    psi=np.fft.ifft(F)
    return psi.real*Lb

##Système différentiel
def phidot(phi, nu):
    """
    Calcul du premier terme du système différentiel adimensionné
    """
    v = np.exp(phi)
    vm = np.mean(v)
    theta = np.exp(nu)
    PSI = np.clip(Psi(v, deltaX, H), -1000, 1000)
    D = v - 1/theta - 1/2*PSI - gamma*(vm-np.ones(I+1))
    phid1 = D/(alpha1 + beta*v)
    phid2 = D/(alpha2 + beta*v)
    return np.concatenate((phid1[0:K1], phid2[K1:K2+1], phid1[K2+1:]))

def nudot(phi, nu):
    """
    Calcul du premier terme du système différentiel adimensionné
    """
    v = np.exp(phi)
    theta = np.exp(nu)
    return 1/theta - v

def F(y):
    """
    Calcul du système différentiel
    """
    phi = y[0]
    nu = y[1]
    F1 = phidot(phi, nu)
    F2 = nudot(phi, nu)
    return np.array([F1, F2])