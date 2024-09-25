from scipy.stats import multivariate_normal
import numpy as np

# Define mean and covariance matrix
mean = [ -30.60605478, 1.42737323, -8.84658588, 1172.5590093 ] # 2D mean vector
cov = [[0.005  ,0.     ,0.     ,0.    ], [0.     ,0.0062 ,0.     ,0.    ],  [0.     ,0.     ,0.015  ,0.    ], [0.     ,0.     ,0.     ,3.    ]]
  # 2x2 covariance matrix

start = [-28.296, 1.4552, -8.482, 1129.6]
# Create a multivariate normal distribution
rv = multivariate_normal(mean, cov)

# Generate a random sample
#sample = rv.rvs(size=1)
#print(sample)

# Compute the probability density at a given point
pdf_value = rv.pdf(start)
print(pdf_value)    