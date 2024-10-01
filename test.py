import numpy as np



# Given values

cD, cE = (0, 0)

y_th = np.array([-30.606054779785527, 1.4273732268259915, 1.6509090158635766, -8.06759017105565,

                 -8.846585876571671, 0.6692257098461925, 1172.5590093034943])

sigma = np.array([0.005, 0.0062, 0.015, 3.0])

y_exp = np.array([-28.296, 1.4552, -8.482, 1129.6])



# Defining the normal likelihood function (Gaussian) for each observable

def gaussian_likelihood(y_exp_i, y_th_i, sigma_i):

    return (1 / (np.sqrt(2 * np.pi) * sigma_i)) * np.exp(-0.5 * ((y_exp_i - y_th_i) / sigma_i) ** 2)



# Calculating the likelihood for each corresponding pair (y_exp_i, y_th_i, sigma_i)

likelihoods = [gaussian_likelihood(y_exp[i], y_th[i], sigma[i]) for i in range(len(y_exp))]



# The total likelihood is the product of individual likelihoods (since they are independent)

total_likelihood = np.prod(likelihoods)

print(total_likelihood)