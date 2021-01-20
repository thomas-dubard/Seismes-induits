import numpy as np
import matplotlib.pyplot as plt
import time

from init import *
from hilbert import *
from RKF45 import *
from couleur import *

red_Phi = 0.5
red_Nu = 0.5
red_Thau = 2
red_Delta = 2

def rep(n: int, N: int) -> str:
    if n == 0:
        return "0"*len(str(N))
    M = int(np.log10(N))
    m = int(np.log10(n))
    return "0"*(M-m) + str(n)

def resolution(CIv: list, CIThau: list, N: int, I: int, report: bool, courbe: bool, export: bool, animate: bool) -> None:
    """
    Cette fonction permet la résolution du système différentiel.
    Elle est déterminée par les conditions initiales.
    Elle dépend aussi des positions de départ (x dans init).
    Elle dépend aussi de la taille du système (N temporel, I spatial).
    L'option "report" permet de suivre le bon déroulement de la fonction
    et son temps d'éxécution.
    L'option "courbe" permet de tracer un rendu des quatre variables principales
    décrivant le phénomène. Elle se fait selon un calcul du maximum et une
    représentation à chaque fois qu'il est multiplié ou divivsé par une valeur.
    L'option "export" permet d'exporter régulièrement les valeurs des quatre
    variables principales décrivant le phénomène. L'expérience montre que pour
    de grandes valeurs de N, les données deviennent trop pesantes pour le
    programme et l'éxécution ralentit, ainsi que le traitement des données après
    éxécution. C'est pour quoi on les exporte et formate.
    L'option "animate" permet d'exporter régulièrement l'état actuel des quatre
    variables principales décrivant le système afin de visualiser en vidéo leur
    évolution (on met les images bout à bout avec un logiciel externe).
    On pourra ajouter une option de visualisation de l'évolution du maximum.
    """
    t_res0 = time.time()
    phik, nuk = np.log(CIv), np.log(CIThau)
    yk = np.array([phik, nuk])
    Phi, Nu = [phik], [nuk]
    Thau, Delta = [[0]*len(Phi[0])], [[0]*len(Phi[0])]
    max_Phi , max_Nu = max(Phi[0]), max(Nu[0])
    max_Thau, max_Delta = max(Thau[0]), max(Delta[0])
    temps = np.array([1e-2])

    if courbe:
        #Initialisation des courbes si on les affiche
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

        if n == 0:
            res = RKF(F, yk)
        else:
            res = RKF(F, yk, h)
        yk, h = res[0], res[1]
        temps = np.concatenate((temps, [temps[n]+h]))

        Phi.append(yk[0])
        Nu.append(yk[1])
        if n >= 1:
            #Thau et Delta sont des variables intégrales.
            #Il faut donc au moins deux valeurs précédentes pour les calculer.
            #On les calcule avec la méthode des rectangles à gauche.
            vnn = np.exp(Phi[n])
            vn = np.exp(Phi[n-1])
            tnn = np.exp(Nu[n])
            tn = np.exp(Nu[n-1])
            Thau.append([Thau[n-1][y] + (a1*np.log(vnn[y]/vn[y]) + b*np.log(tnn[y]/tn[y]))/(temps[n] - temps[n-1]) for y in range(I+1)])
            Delta.append([Delta[n-1][y] + (vnn[y] - vn[y])/(temps[n] - temps[n-1]) for y in range(I+1)])

        if n%100000 == 0 and export:
            #On va régulièrement exporter les variables de calcul.
            #Cela permet de les formater et donc les "alléger".
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
            #On exporte 1000 clichés de l'évolution des variables.
            #Cela permet de visualiser dynamiquement leur évolution en vidéo.
            col = wavelength_to_rgb(380 + (750-380)*n/N)
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
            #On stocke le tracé des courbes si le maximum a suffisamment évolué.
            #Pour la lisibilité il faut le faire que sur un cycle.
            col = wavelength_to_rgb(380 + (750-380)*n/N)
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
        #En fin d'éxécution on peut afficher les courbes.
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