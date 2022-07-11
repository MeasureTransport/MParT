from mpart import *
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing

print(multiprocessing.cpu_count()) # or os.cpu_count()

opts = MapOptions()
map_order = 5
dim = 5
triMap = CreateTriangular(dim,dim,map_order,opts)

N=100000
X = np.loadtxt('X.txt',delimiter=',')
X=X.T

coeffs = np.loadtxt('coeffs.txt',delimiter=',')
triMap.SetCoeffs(coeffs) 

start = time.time()
Y = triMap.Evaluate(X)
dt = time.time()-start

print(dt)

