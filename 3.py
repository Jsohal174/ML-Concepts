import torch
import numpy as np
import pandas as pd

# From Python lists
x = torch.tensor([1, 2, 3])

print("FROM PYTHON LISTS:", x)
print("TENSOR DATA TYPE:", x.dtype)

# From a NumPy array
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
torch_tensor_from_numpy = torch.from_numpy(numpy_array)

print("TENSOR FROM NUMPY:\n\n", torch_tensor_from_numpy)

# From Pandas DataFrame
# Read the data from the CSV file into a DataFrame
df = pd.read_csv('./data.csv')

# Extract the data as a NumPy array from the DataFrame
all_values = df.values

# Convert the DataFrame's values to a PyTorch tensor
tensor_from_df = torch.tensor(all_values)

print("ORIGINAL DATAFRAME:\n\n", df)
print("\nRESULTING TENSOR:\n\n", tensor_from_df)
print("\nTENSOR DATA TYPE:", tensor_from_df.dtype)

# All zeros
zeros = torch.zeros(2, 3)

print("TENSOR WITH ZEROS:\n\n", zeros)

# Random numbers
random = torch.rand(2, 3)

print("RANDOM TENSOR:\n\n", random)

# Range of numbers
range_tensor = torch.arange(0, 10, step=1)

print("ARANGE TENSOR:", range_tensor)

# A 2D tensor
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print("ORIGINAL TENSOR:\n\n", x)
print("\nTENSOR SHAPE:", x.shape)

print("ORIGINAL TENSOR:\n\n", x)
print("\nTENSOR SHAPE:", x.shape)
print("-"*45)

# Add dimension
expanded = x.unsqueeze(0)  # Add dimension at index 0

print("\nTENSOR WITH ADDED DIMENSION AT INDEX 0:\n\n", expanded)
print("\nTENSOR SHAPE:", expanded.shape)

print("EXPANDED TENSOR:\n\n", expanded)
print("\nTENSOR SHAPE:", expanded.shape)
print("-"*45)

# Remove dimension
squeezed = expanded.squeeze()

print("\nTENSOR WITH DIMENSION REMOVED:\n\n", squeezed)
print("\nTENSOR SHAPE:", squeezed.shape)

print("ORIGINAL TENSOR:\n\n", x)
print("\nTENSOR SHAPE:", x.shape)
print("-"*45)

# Reshape
reshaped = x.reshape(3, 2)

print("\nAFTER PERFORMING reshape(3, 2):\n\n", reshaped)
print("\nTENSOR SHAPE:", reshaped.shape)

print("ORIGINAL TENSOR:\n\n", x)
print("\nTENSOR SHAPE:", x.shape)
print("-"*45)

# Transpose
transposed = x.transpose(0, 1)

print("\nAFTER PERFORMING transpose(0, 1):\n\n", transposed)
print("\nTENSOR SHAPE:", transposed.shape)

# Create two tensors to concatenate
tensor_a = torch.tensor([[1, 2],
                         [3, 4]])
tensor_b = torch.tensor([[5, 6],
                         [7, 8]])

# Concatenate along columns (dim=1)
concatenated_tensors = torch.cat((tensor_a, tensor_b), dim=1)


print("TENSOR A:\n\n", tensor_a)
print("\nTENSOR B:\n\n", tensor_b)
print("-"*45)
print("\nCONCATENATED TENSOR (dim=1):\n\n", concatenated_tensors)

# Create a 3x4 tensor
x = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print("ORIGINAL TENSOR:\n\n", x)
print("-" * 55)

# Get a single element at row 1, column 2
single_element_tensor = x[1, 2]

print("\nINDEXING SINGLE ELEMENT AT [1, 2]:", single_element_tensor)
print("-" * 55)

# Get the entire second row (index 1)
second_row = x[1]

print("\nINDEXING ENTIRE ROW [1]:", second_row)
print("-" * 55)

# Last row
last_row = x[-1]

print("\nINDEXING ENTIRE LAST ROW ([-1]):", last_row, "\n")

print("ORIGINAL TENSOR:\n\n", x)
print("-" * 55)

# Get the first two rows
first_two_rows = x[0:2]

print("\nSLICING FIRST TWO ROWS ([0:2]):\n\n", first_two_rows)
print("-" * 55)

# Get the third column of all rows
third_column = x[:, 2]

print("\nSLICING THIRD COLUMN ([:, 2]]):", third_column)
print("-" * 55)

# Every other column
every_other_col = x[:, ::2]

print("\nEVERY OTHER COLUMN ([:, ::2]):\n\n", every_other_col)
print("-" * 55)

# Last column
last_col = x[:, -1]

print("\nLAST COLUMN ([:, -1]):", last_col, "\n")

print("ORIGINAL TENSOR:\n\n", x)
print("-" * 55)

# Combining slicing and indexing (First two rows, last two columns)
combined = x[0:2, 2:]

print("\nFIRST TWO ROWS, LAST TWO COLS ([0:2, 2:]):\n\n", combined, "\n")

print("SINGLE-ELEMENT TENSOR:", single_element_tensor)
print("-" * 45)

# Extract the value from a single-element tensor as a standard Python number
value = single_element_tensor.item()

print("\n.item() PYTHON NUMBER EXTRACTED:", value)
print("TYPE:", type(value))

print("ORIGINAL TENSOR:\n\n", x)
print("-" * 55)

# Boolean indexing using logical comparisons
mask = x > 6

print("MASK (VALUES > 6):\n\n", mask, "\n")

# Applying Boolean masking
mask_applied = x[mask]

print("VALUES AFTER APPLYING MASK:", mask_applied, "\n")
print("ORIGINAL TENSOR:\n\n", x)
print("-" * 55)

# Fancy indexing

# Get first and third rows
row_indices = torch.tensor([0, 2])

# Get second and fourth columns
col_indices = torch.tensor([1, 3]) 

# Gets values at (0,1), (0,3), (2,1), (2,3)
get_values = x[row_indices[:, None], col_indices]

print("\nSPECIFIC ELEMENTS USING INDICES:\n\n", get_values, "\n")

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print("TENSOR A:", a)
print("TENSOR B", b)
print("-" * 60)

# Element-wise addition
element_add = a + b

print("\nAFTER PERFORMING ELEMENT-WISE ADDITION:", element_add, "\n")

print("TENSOR A:", a)
print("TENSOR B", b)
print("-" * 65)

# Element-wise multiplication
element_mul = a * b

print("\nAFTER PERFORMING ELEMENT-WISE MULTIPLICATION:", element_mul, "\n")

print("TENSOR A:", a)
print("TENSOR B", b)
print("-" * 65)

# Dot product
dot_product = torch.matmul(a, b)

print("\nAFTER PERFORMING DOT PRODUCT:", dot_product, "\n")

a = torch.tensor([1, 2, 3])
b = torch.tensor([[1],
                 [2],
                 [3]])

print("TENSOR A:", a)
print("SHAPE:", a.shape)
print("\nTENSOR B\n\n", b)
print("\nSHAPE:", b.shape)
print("-" * 65)

# Apply broadcasting
c = a + b


print("\nTENSOR C:\n\n", c)
print("\nSHAPE:", c.shape, "\n")

temperatures = torch.tensor([20, 35, 19, 35, 42])
print("TEMPERATURES:", temperatures)
print("-" * 50)

### Comparison Operators (>, <, ==)

# Use '>' (greater than) to find temperatures above 30
is_hot = temperatures > 30

# Use '<=' (less than or equal to) to find temperatures 20 or below
is_cool = temperatures <= 20

# Use '==' (equal to) to find temperatures exactly equal to 35
is_35_degrees = temperatures == 35

print("\nHOT (> 30 DEGREES):", is_hot)
print("COOL (<= 20 DEGREES):", is_cool)
print("EXACTLY 35 DEGREES:", is_35_degrees, "\n")

is_morning = torch.tensor([True, False, False, True])
is_raining = torch.tensor([False, False, True, True])
print("IS MORNING:", is_morning)
print("IS RAINING:", is_raining)
print("-" * 50)

### Logical Operators (&, |)

# Use '&' (AND) to find when it's both morning and raining
morning_and_raining = (is_morning & is_raining)

# Use '|' (OR) to find when it's either morning or raining
morning_or_raining = is_morning | is_raining

print("\nMORNING & (AND) RAINING:", morning_and_raining)
print("MORNING | (OR) RAINING:", morning_or_raining)

data = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
print("DATA:", data)
print("-" * 45)

# Calculate the mean
data_mean = data.mean()

print("\nCALCULATED MEAN:", data_mean, "\n")

# A batch of 4 grayscale images, each 3x3
image_batch = torch.rand(4, 3, 3)

print("ORIGINAL BATCH SHAPE:", image_batch.shape)
print("-" * 45)

### START CODE HERE ###

# 1. Add a channel dimension at index 1.
image_batch_with_channel = image_batch_with_channel = image_batch.unsqueeze(1)

# 2. Transpose the tensor to move the channel dimension to the end.
# Swap dimension 1 (channels) with dimension 3 (the last one).
image_batch_transposed = None

### END CODE HERE ###


print("\nSHAPE AFTER UNSQUEEZE:", image_batch_with_channel.shape)
print("SHAPE AFTER TRANSPOSE:", image_batch_transposed.shape)

# Sensor readings (5 time steps)
temperature = torch.tensor([22.5, 23.1, 21.9, 22.8, 23.5])
humidity = torch.tensor([55.2, 56.4, 54.8, 57.1, 56.8])

print("TEMPERATURE DATA: ", temperature)
print("HUMIDITY DATA:    ", humidity)
print("-" * 45)

### START CODE HERE ###

# 1. Concatenate the two tensors.
# Note: You need to unsqueeze them first to stack them vertically.
combined_data = torch.cat((temperature.unsqueeze(0), humidity.unsqueeze(0)), dim=0)

# 2. Create the weights tensor.
weights =  torch.tensor([0.6, 0.4])

# 3. Apply weights using broadcasting.
# You need to reshape weights to [2, 1] to broadcast across columns.
weighted_data = combined_data * weights.unsqueeze(1)

# 4. Calculate the weighted average for each time step.
weighted_average = torch.sum(weighted_data, dim=0)

### END CODE HERE ###

print("\nCOMBINED DATA (2x5):\n\n", combined_data)
print("\nWEIGHTED DATA:\n\n", weighted_data)
print("\nWEIGHTED AVERAGE:", weighted_average)