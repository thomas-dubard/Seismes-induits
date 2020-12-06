import matplotlib.pyplot as plt
import math

def RKF(f, tk, yk, h):
    k1 = h * f(tk, yk)
    k2 = h * f(tk + h/4, yk + k1/4)
    k3 = h * f(tk + 3*h/8, yk + 3*k1/32 + 9*k2/32)
    k4 = h * f(tk + 12*h/13, yk + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
    k5 = h * f(tk + h, yk + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)
    k6 = h * f(tk + h/2, yk - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)
    ykk = yk + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5
    zkk = yk + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55
    s = (h / (2 * abs(zkk - ykk)))**0.25
    return ykk, s*h

def ftest(tk, yk):
    return 1 + yk**2
deltaT = 0.01
X = [k*deltaT for k in range(100)]
Yref = [math.tan(t) for t in X]
Ytest = [0.0]
h = deltaT
for t in X[:-1]:
    ykk, hkk = RKF(ftest, t, Ytest[-1], h)
    Ytest.append(ykk)
plt.plot(X, Yref, color='blue', label='tan')
plt.plot(X, Ytest, color='red', label='RKF')
plt.show()