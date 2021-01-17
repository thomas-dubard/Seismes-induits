##Conditions initiales
x = np.arange(-L//2, L//2 + 1, deltaX)
d = L//8 #largeur de la zone de glissement initiale
amp = 10 #amplitude max
temps = np.array([0])
K1 = (L//2-d)//deltaX
K2 = (L//2+d)//deltaX

def cond_init(forme):
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
    if forme == "uniforme"
        #Uniforme
        Au = np.ones(I+1)*amp
        Bu = 1/Au
        return Au, Bu