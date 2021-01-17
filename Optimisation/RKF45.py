import time

def RKF(f, yk, h=1e-2, dtmin=1e-16, dtmax=1e2, tol=1e-10):
    """
    Cette fonction vise à résoudre une équation différentielle vectorielle.
    Elle utilise la méthode Runge-Kutta-Fehlberg d'ordre 5.
    Cela permet d'adapter le pas de temps pour avoir une meilleure précision.
    """
    K = 0
    t0 = time.time()
    if h > dtmax:
        #Cela empêche la fonction de marcher si c'est le cas.
        h = dtmax/2
    s = 1.
    ykk = yk
    while s*h < dtmax and K < 30:
        k1 = h * f(yk)
        k2 = h * f(yk + k1/4)
        k3 = h * f(yk + 3*k1/32 + 9*k2/32)
        k4 = h * f(yk + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
        k5 = h * f(yk + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)
        k6 = h * f(yk - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)
        ykk = yk + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5
        #print('ykk='+str(ykk))
        zkk = yk + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55
        erreur = np.linalg.norm(zkk - ykk)
        tf = time.time()
        K = K + 1
        s = (tol / (2 * erreur))**0.25
        print(erreur)
        print(K)
        if erreur < tol:
            #print(tf-t0)
            return ykk, s*h
        if s*h < dtmin:
            #print(tf-t0)
            return ykk, h
        h = s*h
        #print(tf-t0)
    if K==30 and erreur == tol:
        # Si on a trop d'itérations, on arrête l'éxécution.
        raise ValueError("Trop d'iterations")
    return ykk, h