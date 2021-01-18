import numpy as np
import matplotlib.pyplot as plt
import time

from init import *
from RKF45 import *
from couleur import *

red_Phi = 0.5
red_Nu = 0.5
red_Thau = 2
red_Delta = 2

def rep(n, N):
    M = int(np.log10(N))
    m = int(np.log10(n))
    return "0"*(M-m) + str(n)

def resolution(CIv, CIThau, N, I, report="False", courbe="False", export="False", animate="False"):
    t_res0 = time.time()
    phik, nuk = np.log(CIv), np.log(CIThau)
    yk = np.array([phik, nuk])
    Phi, Nu = [phik], [nuk]
    Thau, Delta = [[0]*len(Phi[0])], [[0]*len(Phi[0])]
    max_Phi , max_Nu = max(Phi[0]), max(Nu[0])
    max_Thau, max_Delta = max(Thau[0]), max(Delta[0])

    if courbe:
        plt.figure(figsize=(16, 10))
        col = wavelength_to_rgb(380)
        plt.subplot(2, 2, 1)
        plt.plot(x, np.exp(Phi[0]), color=col)
        plt.subplot(2, 2, 2)
        plt.plot(x, np.exp(Nu[0]), color=col)
        plt.subplot(2, 2, 3)
        plt.plot(x, Delta[0], color=col)
        plt.subplot(2, 2, 4)
        plt.plot(x, Thau[0], color=col)

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

        if n%100000 == 0 and export:
            numpyPhi = np.array(Phi)
            np.savetxt(f"Phi-{rep(n, N)}.csv", numpyPhi, delimiter=',')
            Phi = [Phi[-1]]
            numpyNu = np.array(Nu)
            np.savetxt(f"Nu-{rep(n, N)}.csv", numpyNu, delimiter=',')
            Nu = [Nu[-1]]
            numpyThau = np.array(Thau)
            np.savetxt(f"Thau-{rep(n, N)}.csv", numpyThau, delimiter=',')
            Thau = [Thau[-1]]
            numpyDelta = np.array(Delta)
            np.savetxt(f"Delta-{rep(n, N)}.csv", numpyDelta, delimiter=',')
            Delta = [Delta[-1]]

        if n%(N//1000) == 0 and animate:
            col = wavelength_to_rgb(380 + (750-380)*temps[n]/temps[N])
            plt.plot(x, Phi[n])
            #plt.ylim(0, 21)
            plt.savefig(f"Phi-{rep(n, N)}.png", format='png')
            plt.clf()
            plt.plot(x, Nu[n])
            #plt.ylim(-20, 0)
            plt.savefig(f"Nu-{rep(n, N)}.png", format='png')
            plt.clf()
            plt.plot(x, Thau[n])
            #plt.ylim(0, 1)
            plt.savefig(f"Thau-{rep(n, N)}.png", format='png')
            plt.clf()
            plt.plot(x, Delta[n])
            #plt.ylim(0, 21)
            plt.savefig(f"Delta-{rep(n, N)}.png", format='png')
            plt.clf()

        if courbe:
            col = wavelength_to_rgb(380 + (750-380)*temps[n]/temps[N])
            maxn = max(Phi[n])
            if maxn > max_Phi + red_Phi or maxn < max_Phi - red_Phi:
                max_Phi = maxn
                plt.subplot(2, 2, 1)
                plt.plot(x, np.exp(Phi[n]), color=col)
            maxn = max(Nu[n])
            if maxn > max_Nu + red_Nu or maxn < max_Nu - red_Nu:
                max_Nu = maxn
                plt.subplot(2, 2, 2)
                plt.plot(x, np.exp(Nu[n]), color=col)
            maxn = max(Thau[n])
            if maxn > max_Thau * red_Thau or maxn < max_Thau / red_Thau:
                max_Thau = maxn
                plt.subplot(2, 2, 3)
                plt.plot(x, np.exp(Thau[n]), color=col)
            maxn = max(Delta[n])
            if maxn > max_Delta * red_Delta or maxn < max_Delta / red_Delta:
                max_Delta = maxn
                plt.subplot(2, 2, 4)
                plt.plot(x, np.exp(Delta[n]), color=col)
    if courbe:
        plt.subplot(2, 2, 1)
        plt.title('V(x)')
        plt.subplot(2, 2, 2)
        plt.title('Theta(x)')
        plt.subplot(2, 2, 3)
        plt.title('Thau(x)')
        plt.subplot(2, 2, 4)
        plt.title('Delta(x)')
        plt.savefig("Resultat_final.png", format='png')
        plt.show()