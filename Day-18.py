# Importing The Dependencies
import numpy as np
import torch
import torch.nn as nn

# ========================================================================================================================== #
# ========================================================================================================================== #

# Task 1: Dimension Clash -> Create a random matrix of shape 3 * 2 and another of shape 2 * 4. Combine them mathematically to output a single matrix of shape 3 * 4.

# --- Solution --- #
random_1 = torch.rand(3, 2)
random_2 = torch.rand(2, 4)
mat = torch.matmul(random_1, random_2)
print(mat)

# ========================================================================================================================== #
# ========================================================================================================================== #

# Task 2: The GPU Leap -> Create a normal NumPy array containing [5.5, 6.5, 7.5]. Move this data entirely onto your GPU memory using dynamic device selection, and verify its final location.

# --- Solution --- #
arr1 = np.array([5.5, 6.5, 7.5])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_tensor = torch.from_numpy(arr1).to(device)

# --- Verify location cleanly --- #
if gpu_tensor.is_cuda:
    print(f"💥 Success! Tensor moved to GPU: {gpu_tensor}")
else:
    print(f"Running on CPU: {gpu_tensor}")

# ========================================================================================================================== #
# ========================================================================================================================== #

# Task 3: The Shape Shifter -> Create a matrix with 2 rows and 3 columns containing numbers from 1 to 6. Change its structure completely to turn it into a single vertical column of shape 6 * 1.

# --- Solution --- #
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
reshaped_matrix = matrix.reshape(6, 1)
print(reshaped_matrix)


layer = nn.Linear(in_features=3,out_features=2)
print(layer.weight)
print(layer.bias)

tensor = torch.tensor([2.00,3.00],requires_grad=True)
updated_tensor = tensor ** 2
print(updated_tensor)
# ========================================================================================================================== #
#                                                          0 END 1                                                           #
# ========================================================================================================================== #
