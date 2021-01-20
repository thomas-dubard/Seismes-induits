import time

from init import *
from solve import *


##Résolution
CIv, CItheta = cond_init("gaussienne")
resolution(CIv, CItheta, N, I, report=True, courbe=False, export=False, animate=False)