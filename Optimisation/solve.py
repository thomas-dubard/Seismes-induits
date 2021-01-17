import numpy as np
import matplotlib.pyplot as plt
import time

from RKF45 import *
from couleur import *

def resolution(CIv, CItheta, N, I, report="False", courbe="False", export="False"):
    t_res0 = time.time()
    phik, nuk = np.log(CIv), np.log(CItheta)
    yk = np.array([phik, nuk])
    Phi, Nu = [phik], [nuk]
    Thau = [[0]*len(Phi[0])]

    if courbe:
        plt.figure(figsize=(16, 10))

    for n in range(N):
        if n%10000 == 0 and report:
            print(f"Itération {n}")
            print(f"Run time = {time.time() - t_res0}")

        res = RKF(F, yk, h)
        yk, h = res[0], res[1]
        temps = np.concatenate((temps, [temps[n]+h]))

        Phi.append(yk[0])
        Nu.append(yk[1])
        if n >= 1:
            vnn = np.exp(Phi[n])
            vn = np.exp(Phi[n-1])
            tnn = np.exp(Nu[n])
            tn = np.exp(Nu[n-1])
            delta_Thau = (a*np.log(vnn[x]/vn[x]) + b*np.log(tnn[x]/tn[x]))
            delta_v = (vnn[x] - vn[x])
            delta_t = (temps[n] - temps[n-1])
            Thau.append([Thau[-1][x] + delta_Thau/delta_t for x in range(I+1)])
            Delta.append([Delta[-1][x] + delta_v/delta_t for x in range(I+1)])

        if courbe:
