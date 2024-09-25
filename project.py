import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
#import quanstumsolver 
import qs

errors = np.array([0.005, 0.0062, 0.015, 3.0]) #E4He, Rp4He, E3H, fT3H
y_exp = [-28.296,1.4552,-8.482,1129.6]

def logPrior(cD, cE):
    mean = 0
    cov = 5
    normal_cD = np.log(stats.multivariate_normal.pdf(cD,mean,cov))
    normal_cE = np.log(stats.multivariate_normal.pdf(cE,mean,cov))
    return normal_cD + normal_cE

def logLikelihood(y_th, sigma_i):
    res = stats.multivariate_normal(mean = y_th,cov = sigma_i)
    return (res.pdf(y_exp))

def logLikelihood_driver(cD,cE):
    res = qs.fewnucleonEmulator(cD,cE)

    y_th = np.array([res[0],res[1],res[4],res[6]])
    cov_matrix = np.diag(errors)
    
    return logLikelihood(y_th,cov_matrix)

def logPosterior(theta):
    return logPrior(theta[0],theta[1]) + logLikelihood_driver(theta[0],theta[1])


cD_len = 72
cE_len = 72
cD = np.linspace(1.26,1.3175,cD_len)
cE = np.linspace(0.0175,0.0275 ,cE_len)
grid = [[(d,e) for d in cD] for e in cE]
grid = np.asarray(grid)

res = []
log_res = []
for row in grid:
    for theta in row:
        log_res.append(logPosterior(theta))

res = np.exp(log_res)
res = res.reshape(cE_len,cD_len)
#plt.imshow(res)

plt.contour(cD,cE,res)  
plt.show()