import numpy as np

##Paramètres
mu = 30e9
sigma = 100e6
rho = 2400

a1 = 2e-3
a2 = 2e-4
b = 1e-3
v0 = 1e-9
dc = 1e-3

eta = np.sqrt(mu*rho/2) #=6000000
Lb = mu*dc/(b*sigma) #=300

deltaX = int(Lb/4)
N = 1000
I = 2**9
L = I * deltaX
H = 2*L
h = 1e-2

alpha1 = a1/b
alpha2 = a2/b
beta = eta*v0/(b*sigma) #=6e-08
gamma = mu*dc/(b*sigma*H)

##Conditions initiales
x = np.arange(-L//2, L//2 + 1, deltaX)
d = L//8 #largeur de la zone de glissement initiale
amp = 10 #amplitude max
temps = np.array([0])
K1 = (L//2-d)//deltaX
K2 = (L//2+d)//deltaX

def cond_init(forme: str) -> (list, list):
    """
    Cette fonction permet de choisir le profil de condition initiale.
    Elle est déterminée aussi par les constantes ci-dessus.
    """
    if forme == "elliptique":
        #Elliptique
        Ae = []
        for y in x :
            if y < d and y > -d:
                Ae.append((np.sqrt(d**2 - y**2)/d)*amp + 1)
            else :
                Ae.append(1)

        Ae = np.array(Ae)
        Be = 1/Ae
        return Ae, Be
    if forme == "gaussienne":
        #Gaussienne
        Ag = np.exp(-x**2/(d**2))*amp + np.ones(I+1)
        Bg = 1/Ag
        return Ag, Bg
    if forme == "carrée":
        #Carré
        Ac = []
        for y in x :
            if y < d and y > -d:
                Ac.append(amp)
            else :
                Ac.append(1)

        Ac = np.array(Ac)
        Bc = 1/Ac
        return Ac, Bc
    if forme == "uniforme":
        #Uniforme
        Au = np.ones(I+1)*amp
        Bu = 1/Au
        return Au, Bu