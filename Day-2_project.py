#Import Numpy
import numpy as np

#The Data
data = np.array([
    [101, 85, 90, 78, 88, 92],
    [102, 35, 45, 50, 40, 38],
    [103, 95, 92, 98, 90, 95],
    [104, 40, 38, 42, 35, 45],
    [105, 70, 75, 80, 85, 88]

])

#Calculating Functions
print(np.sum(data[:,1:], axis = 1))
print(np.mean(data[:,1:]))
