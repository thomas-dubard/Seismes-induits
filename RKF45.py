import matplotlib.pyplot as plt
import math
import time

def RKF(f, tk, yk, h, epsilon=1e-2):
    erreur = 1.
    cpt = 0
    while erreur > epsilon and cpt < 30:
        k1 = h * f(tk, yk)
        k2 = h * f(tk + h/4, yk + k1/4)
        k3 = h * f(tk + 3*h/8, yk + 3*k1/32 + 9*k2/32)
        k4 = h * f(tk + 12*h/13, yk + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
        k5 = h * f(tk + h, yk + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)
        k6 = h * f(tk + h/2, yk - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)
        ykk = yk + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5
        zkk = yk + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55
        erreur = abs(zkk - ykk)
        if erreur == 0:
            return ykk, h
        s = (h / (2 * erreur))**0.25
        h = s*h
        cpt += 1
    return ykk, s*h

def RKFixed(f, tk, yk, h, dtmin=1e-16, dtmax=1e-2, tol=1e-3):
    t0 = time.time()
    if h > dtmax:   #Cela empêche la fonction de marcher si c'est le cas.
        h = dtmax/2
    s = 1.
    ykk = yk
    cpt = 0
    while s*h < dtmax:
        cpt += 1
        print(f"iteration {cpt} pour {tk} et {yk}")
        k1 = h * f(tk, yk)
        k2 = h * f(tk + h/4, yk + k1/4)
        k3 = h * f(tk + 3*h/8, yk + 3*k1/32 + 9*k2/32)
        k4 = h * f(tk + 12*h/13, yk + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
        k5 = h * f(tk + h, yk + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)
        k6 = h * f(tk + h/2, yk - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)
        ykk = yk + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5
        zkk = yk + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55
        erreur = abs(zkk - ykk)
        if erreur < tol:
            tf = time.time()
            print(f"Run time = {tf - t0}s")
            return ykk, h
        s = (h / (2 * erreur))**0.25
        if s*h < dtmin:
            tf = time.time()
            print(f"Run time = {tf - t0}s")
            return ykk, h
        h = s*h
    tf = time.time()
    print(f"Run time = {tf - t0}s")
    return ykk, h

def ftest(tk, yk):
    return 1 + yk**2
T0 = time.time()
deltaT = 0.001
X = [k*deltaT for k in range(1400)]
Yref = [math.tan(t) for t in X]
Ytest = [0.0]
h = deltaT
for t in X[:-1]:
    ykk, hkk = RKFixed(ftest, t, Ytest[-1], h)
    Ytest.append(ykk)
    h = hkk
plt.plot(X, Yref, color='blue', label='tan')
plt.plot(X, Ytest, color='red', label='RKF')
Tf = time.time()
print(f"full run time = {Tf - T0}s")
plt.show()