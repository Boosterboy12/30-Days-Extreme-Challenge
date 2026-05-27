#Importing Numpy 
import numpy as np 

#Level-1

#Task-1 
array1 = np.arange(10,50)
print(array1)
print(array1.shape)
print(type(array1))

#Task-2 
print(array1[::-1])

#Task-3
array2 = np.arange(0,9)
print(array2.reshape(3,3))

#Task-4
array3 = np.identity(5)
print(array3)

#Task-5
array4 = np.random.randint(1,10,10) 
print(array4)

#Level-2

#Task-6
array5 = np.zeros((8,8))
array5[::2, 1::2] = 1  
array5[1::2, ::2] = 1 
print(array5)

#Task-7
array6 = np.ones((5,5))
array6[1:-1, 1:-1] = 0
print(array6)

#Task-8
array7 = np.arange(1,37).reshape(6,6)
print(array7[2:4, 2:4])

#Level-3

#Task-9
array8 = np.random.randint(1,101,50)
fil_array8 = array8[array8 > 50]
print(fil_array8)

#Task-10
array9 = np.random.randint(1,100,30)
array9[array9 % 3 == 0] = -1
print(array9)

#Task-11
A = np.random.randint(1,100,10)
B = np.random.randint(1,100,10)
c = A > B
print(c)

#Level-4 

#Task-12
array10 = np.random.randint(1,100,(4,4))
print(np.min(array10))
print(np.sum(array10, axis=1))

#Task-13
array11 = np.random.rand(20)
print(array11.mean())
print(array11.std())

#Task-14
array12 = np.array([[2,3,4],[5,6,7]])
print(np.transpose(array12))

#Task-15
array13 = np.arange(1,10)
print(array13**2)