import time

from init import *
from solve import *

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
N = 10000
I = 2**9
L = I * deltaX
H = 2*L
h = 1e-2

alpha1 = a1/b
alpha2 = a2/b
beta = eta*v0/(b*sigma) #=6e-08
gamma = mu*dc/(b*sigma*H)

##Résolution
CIv, CItheta = cond_init("gaussienne")
resolution(CIv, CItheta, N, I, report="False", courbe="False", export="False")