# -*- coding: utf-8 -*-
"""00_PyTorch_Fundamental_ZTM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1iU1tZyXJBvV36Jt3ZcvnDa74SlT45ZFe
"""

## OO. PyTorch Fundamental

import torch
print(torch.__version__)

"""# **Introduction to Tensors**

####Creating scalars
"""

# scalar
scalar = torch.tensor(6)
scalar

scalar.ndim

# Get tensor back as Python int
scalar.item()

# Vector
vector = torch.tensor([7,7])
vector

vector.shape

# MATRIX
MATRIX = torch.tensor([[7,8],
                       [6,8]])
MATRIX

MATRIX.shape

MATRIX.ndim

MATRIX[1]

# TENSOR
TENSOR = torch.tensor([[[1,2,3],
                        [3,6,9],
                        [2,4,5]]])

TENSOR

TENSOR.ndim

TENSOR.shape

# Output will be this:
# torch.Size([1, 3, 3])
# because In above while creating an TENSOR, there is one dimensional, with very outer square bracket in purple color showing 1D Tensor with 3x3 matrix data.

"""## **Random Tensors**"""

# Why Random tensors ?

# Random tensors are important because the way many neural networks learn is that they start with tensors full of random numbers and then adjust those random numbers to better represent the data.

# Start with random number -> look at data -> update random numbers -> look at data -> update random numbers.

# Create a random tensor of size (3,4)
random_tensor = torch.rand(3,4)
random_tensor

random_tensor.ndim

# Create a random tensor with similar shape to an image tensor.
random_image_size_tensor = torch.rand(size=(224,224,3)) # height, width, colour channels.
random_image_size_tensor.shape, random_image_size_tensor.ndim

torch.rand(size=(3,3))

"""#### **Zeros and Ones**"""

# Create a tensors of all zeros.
zeros = torch.zeros(size=(3,4))
zeros

zeros * random_tensor

# Create a tensors of all ones
ones = torch.ones(size=(3,4))
ones

ones.dtype

random_tensor.dtype

"""### **Ranges and Tensor Like Other Tensors**"""

# Use torch.range
torch.arange(start=1, end=11)

torch.__version__

one_to_ten = torch.arange(start=0, end=10, step=2)
one_to_ten

# Creating tensors like
# If we don't want to explictly define the shape of tensors.
ten_zeros = torch.zeros_like(input=one_to_ten)
ten_zeros

"""### **Tensor DataTypes**


`Note`: Tensor dataype is one of the 3 big error you will run into with PyTorch & deep learning.

1. Tensors not right datatype.

2. Tensors not right shape.

3. Tensors not on the right device.
"""

# Float 32 tensor.
# Default data type in Pytorch is float 32.
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.int, # What datatype is the tensor (e.g. float32, or float16)
                               device="cuda", # What device is your tensor on
                               requires_grad=False) # whether or not to track gradients with this tensors operations.
float_32_tensor

# Converting the float32 tensor to float16 tensors.
float_16_tensor = float_32_tensor.type(torch.float16)
float_16_tensor

# Multiplication of Two tensors.
float_16_tensor * float_32_tensor

"""### **Getting Tensor Attributes**"""

int_32_tensor = torch.tensor([3,6,9], dtype=torch.int32, device="cuda")
int_32_tensor

# Multiplication of two tensors with different dataypes.
float_32_tensor * int_32_tensor

long_32_tensor = torch.tensor([3,6,9], dtype=torch.long)
long_32_tensor

"""### **Getting Information about tensors**

a. Tensors not right dataype - to do get datatype from a tensor, can use `tensor.dtype`

b. Tensors not right shape - to get shape from a tensor, can use `tensor.shape`

c. Tensors not on the right device - to get device from a tensor, can use `tensor.device`
"""

# Creating a tensor.
some_tensor = torch.rand(3,4)
some_tensor

# Find out the details about the some tensor.
print(some_tensor)
print(f"Data Type of tensor:  {some_tensor.dtype}")
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Device tensor is on:  {some_tensor.device}")

"""### **Manipulating Tensors**


Tensor Operations include:

* Addition.

* Substraction.

* Multiplication (element-wise).

* Division.

* Matrix Multiplication.
"""

# Create a tensor and add 6 to it.
tensor = torch.tensor([1,2,3])
tensor = tensor + 6
tensor

# Multiply tensor by 6
tensor = tensor * 6
tensor

# Substract 6
tensor = tensor - 6
tensor

# Try out PyTorch in-built Functions.
torch.mul(tensor, 10)

torch.add(tensor, 5)

"""### **Matrix Multiplications**


Two mains ways of performing multiplication in neural networks and deep learning:

1. Element-wise multiplication.

2. Matrix Multiplication. (Dot product)


The main Two Rules of Matrix Multiplications

1. The **inner dimensions** must matched.
    
* `(3, 2) @ (3, 2)` won't work.

* `(2, 3) @ (3, 2)` will work.

* `(3, 2) @ (2, 3)` will work.


2. The resulting matrix has the shape of the **outer dimensions**:

* `(2, 3) @ (3, 2)` -> `(2, 2)`

* `(3, 2) @ (2, 3)` -> `(3, 3)`

"""

torch.matmul(torch.rand(3,2), torch.rand(2,3))

# Create a two matrix.
# 1. Element wise multiplication.
tensor = torch.tensor([4,5,6])
print(tensor, " * ", tensor)
print(f"Equals: {tensor * tensor}")

# 1. Matrix multiplication. (Dot Product)
tensor = torch.tensor([4,5,6])
print(tensor, " * ", tensor)
print(f"Equals: {torch.matmul(tensor, tensor)}")

# Commented out IPython magic to ensure Python compatibility.
# %%time
# value = 0
# for i in range(len(tensor)):
#     value += tensor[i] * tensor[i]
# 
# value

# Commented out IPython magic to ensure Python compatibility.
# %%time
# tensor1 = torch.tensor([3,4,5])
# tensor2 = torch.tensor([3,4,5])
# torch.matmul(tensor1, tensor2)

"""### **One of the Most Common Error in the deep learning: shape errors**


Cool thing about matrix multiplication: http://matrixmultiplication.xyz/


"""

# shapes for Matrix Multiplications.
tensor_A = torch.tensor([[1,2],
                         [3,4],
                         [5,6]])

tensor_B = torch.tensor([[5,6],
                         [7,8]])

# torch.mm is the same as torch.matmul (it's an alias for matmul)
# torch.mm(tensor_A, tensor_B)
print(tensor_A.shape)
print(tensor_B.shape)
print(f"\nMatrix Multiplication :\n {torch.mm(tensor_A, tensor_B)}")

""" To fix our tensor shape issues, we can manipulate the shape of one our tensors using a Transpose.

 A **transpose** switches the axes or dimensions of a given tensor.
"""

# To fix our tensor shape issues, we can manipulate the shape of one our tensors using a Transpose.
tensor_A = torch.tensor([[1,2],
                         [3,4],
                         [5,6]])

tensor_B = torch.tensor([[5,6],
                         [7,8],
                         [5,4]])

# Tranpose the tensor_B
tensor_B.T

# The matrix multiplication operation works when tensor_B is transposed.
print(f"Original shape: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}")
print(f"New shapes: tensor_A = {tensor_A.shape} (same as above), tensor_B.T = {tensor_B.T.shape}")

# tensor_A.shape, tensor_B.T.shape
# print(torch.mm(tensor_A, tensor_B.T))
print(f"Multiplying : {tensor_A.shape} @ {tensor_B.T.shape} <- Inner dimensions must matched.")
print("Output: \n")
output = torch.mm(tensor_A, tensor_B.T)
print(output)
print(f"\n Output shape: {output.shape}")

"""### **Finding the min, max, mean, sum, etc. (Tensor aggregation).**"""

# Create a tensor.
x = torch.arange(0, 100, 10)
x

# Find the min.
torch.min(x), x.min()

# Find the max.
torch.max(x), x.max()

# Find the mean. - note: the torch.mean() function requires a tensor of float32 datatype to work.
torch.mean(x.type(torch.float32)), x.type(torch.float32).mean()

# Find the sum.
torch.sum(x), x.sum()

"""### **Find the Positional Min and Max of Tensors**"""

x = torch.arange(3,32,2)

# Find the position in tensor that has the minimum value with argmin() -> return index position of target tensor where the minimum value occurs.
x.argmin()

x[0]

# Find the Position in tensor that has the maximum value with argmax()
x.argmax()

x[14]

"""### **Reshaping, stacking, squeezing, and unsqueezing tensors**

* Reshaping - reshapes an input tensor to a defined shape.

* View - Return a view of an input tensor of certain shape but keep the same memory as the original tensor.

* Stacking - Combine multiple tensors on top of each otehr (vstack) or (hstack or side by side).

* Squeeze - removes all `1` dimensions from a tensor.

* Unsqueese - add a `1` dimension to a target tensor.

* Permute - Return a view of the input dimensions permuted (swapped) in a certain way.

"""

# Let's create tensor.
import torch
x = torch.arange(1.,10.)
x, x.shape

# Add an extra dimension.
x_reshaped = x.reshape(1, 9)
x_reshaped, x_reshaped.shape

# Change the View -> view is quite similar to reshape.
# view shares the original memory with tensors.
z = x.view(1, 9)
z, z.shape

# Changing z changes x (because a view of a tensor shares the same memory as the original input)
z[:, 0] = 5
z, x

# Stack tensors on top of each other
# vstack -> dims=1
# hstack -> dims=0
x_stacked = torch.stack([x,x,x,x], dim=0)
x_stacked

# torch.squeeze() - removes all single dimensions from a target tensor.

# Returns a tensor with all dimensions of input of size 1 removed.

print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

# Remove extra dimensions from x_shaped.
x_squeezed = x_reshaped.squeeze()

print(f"\nNew tensor : {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")

# torch.unsqueeze() - adds a single dimension to a target tensor at a specific dimension

print(f"\nPrevious target : {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")

# Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)

print(f"\nNew tensor : {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")

# torch.permute - rearranges the dimensions of a target tensor in a specific order.
# returns a view of the original tensor input with its dimensions permuted.
x_original = torch.rand(size=(224, 224, 3)) # [height, width, colour_channels]

# Permute the original tensor to rearrange the axis (or dim) order.
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0
x_permuted.shape

print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}") # [colour_channels, height, width]

"""### **Indexing(Selecting Data From tensors)**

Indexing with PyTorch is similar to indexing with NumPy.
"""

# Create a tensor
import torch
x = torch.arange(1, 10).reshape(1, 3, 3)
x, x.shape

x[0]

# Let's index on the middle bracket (dim=1)
x[0][1]

# Let's index on the most inner bracket (last dimension)
x[0][0][2]

# We can also use ":" to select "all" of a target dimension.
x[:,0]

# Get all values of 0th and 1st dimensions but only index 1 of 2nd dimensions.
x[:, :, 1]

# Get all values of the 0th dimension but only the 1 index value of 1st and 2nd dimension.
x[:, 1, 1]

# Get index of 0th and 1st dimensions and all values of 2nd dimensions.
x[0, 0, :]

x

# Index on x to return  9
x[:, 2, 2]

# Index on x to return 3, 6, 9
x[0,:,2]

"""### **PyTorch Tensor and NumPy**

#### Numpy is a popular scientific Python numerical computing library.

* Data in NumPy, want in PyTorch tensorn -> `torch.from_numpy(ndarray)`

* PyTorch tensor -> NumPy -> `torch.Tensor.numpy()`
"""

import torch
import numpy as np

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array).type(torch.float32) # warning: when converting from numpy -> pytorch, pytorch reflects numpy's default datatype of float64 unless speficied otherwise.

array, tensor

array.dtype

torch.arange(1.0, 8.0).dtype

# Change the value of array, what will this do to `tensor` ?
array = array + 1
array, tensor

# Tensor to NumPy array.
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
tensor, numpy_tensor

# Change the tensor, what happens to `numpy_tensor` ?
tensor = tensor + 1
tensor, numpy_tensor

"""**Note:** ---- Changing the value of tensor does not change the value of the numpy because tensor and numpy does not share the same memory location so that.----

### **PyTorch Reproducibility (Taking the Random Out of Random)**

In short how a neural network learns:

`start with random numbers -> tensor operation -> update random numbers to try and make them better representations of the data -> again -> again -> again`.


To reduce the randomness in neural networks and PyTorch comes the concept of a **random seed**.

Essentially what the random seed does if "flavour" the randomness.
"""

import torch
import random

# Create two random tensor
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)

# Let's make some random but reproducible tensors.
import torch

# Set the random seed.
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)

print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)

"""### **Different ways of accessing the GPU**

Running tensors and PyTorch objects on the GPUs (and making faster computations)

GPUs = faster computation on numbers.
"""

# 1. Gettings GPUs.
# !nvidia-smi

# Check for GPU access with PyTorch
import torch
torch.cuda.is_available()

# Setup device agnostic code.
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# Count the number of devices.
torch.cuda.device_count()

"""### **Setting up Device Agnostic Code and Putting Tensors On and Off the GPU**

Putting tensors (and models) on the GPU.

The reasons we want our tensors/models on the GPU is because of using a GPU results in faster computations.
"""

# Create a tensor (default on the CPU)
tensor_on_cpu = torch.tensor([1,2,3], device='cpu')

# Tensor not on GPU.
print(tensor, tensor_on_cpu.device)

# Move tensor to GPU (if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

tensor_on_gpu = tensor_on_cpu.to(device)

tensor_on_gpu

# Moving tensors baxck to the GPU,.
# If tensor is on GPU, can't transform it to NumPy.
tensor_on_gpu.numpy()

# In order to fix the above GPU tensor with NumPy issue, we can first set it to the CPU.
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
tensor_back_on_cpu
s
tensor_on_gpu

