 #Importing Numpy
import numpy as np 

# Problem-1

# Create a 1D array with values from 1 to 25.
# Reshape this array into a 5x5 matrix.
# Extract the center 3x3 matrix from it.

arr1= np.arange(1,26).reshape(5,5)
print(arr1[1:4,1:4])

# Problem-2

# Create an array of 50 random integers between 1 and 100 
# Without using any for loops, identify all the even numbers in this array.
# Update these even numbers by adding 1 to them (making them odd).

arr2 = np.random.randint(1,101,50)
b = arr2[arr2 % 2 == 0 ] 
b = arr2[arr2 % 2 == 0 ] = b+1 

print(arr2)

# Problem-3

# Create an array of marks: data = np.array([10, 20, 50, 80, 100]).
# Normalize this array so that all values are transformed to a range between 0 and 1.
# You must use NumPy functions to find the minimum and maximum values to keep the calculation dynamic.

data = np.array([10, 20, 50, 80, 100])
a = (np.min(data))
b = (np.max(data))
c = (data - a) / (b - a)
print(c)

#Task-1 
arr = np.arange(1, 21)
mask = (arr > 5) & (arr < 15)
arr[mask] = -1
print(arr)


#Task-2
array2 = np.random.randint(1,100,25).reshape(5,5)
print(array2.mean())
print(array2.std())
print(array2.sum())

#Task-3
array3 = np.array([5, 12, 17, 20, 25, 30])
print(np.where(array3 > 20, 1, 0))

#Task-4
array4 = np.identity(4)
print(array4)

#Task-5
array5 = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(np.ravel(array5))
