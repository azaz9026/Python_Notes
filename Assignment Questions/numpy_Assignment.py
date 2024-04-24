 

 # Numpy Assignment -----------------------------------------------------------------------------------------------------------



import numpy as np

# Task 1
arr = np.arange(6)
print("1. Data type of 'arr':", arr.dtype)

# Task 2
def check_float64(arr):
    return arr.dtype == np.float64

# Task 3
arr_complex = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)

# Task 4
def convert_to_float32(arr):
    return arr.astype(np.float32)

# Task 5
def convert_to_float32_precision(arr):
    return arr.astype(np.float32)

# Task 6
def array_attributes(arr):
    return arr.shape, arr.size, arr.dtype

# Task 7
def array_dimension(arr):
    return arr.ndim

# Task 8
def item_size_info(arr):
    return arr.itemsize, arr.nbytes

# Task 9
def shape_stride_relationship(arr):
    return arr.shape, arr.strides

# Task 10
def create_zeros_array(n):
    return np.zeros(n)

# Task 11
def create_ones_matrix(rows, cols):
    return np.ones((rows, cols))

# Task 12
def generate_range_array(start, stop, step):
    return np.arange(start, stop, step)

# Task 13
def generate_linear_space(start, stop, num):
    return np.linspace(start, stop, num)

# Task 14
def create_identity_matrix(n):
    return np.eye(n)

# Task 15
def list_to_numpy_array(lst):
    return np.array(lst)

# Task 16
def numpy_view_demo(arr):
    # Create a view of the array with a different shape
    new_arr = arr.view(np.int32)
    return new_arr

# Testing the functions
print("2. Is 'arr' of float64 data type?", check_float64(arr))
print("3. Complex array 'arr_complex':", arr_complex)
print("4. 'arr' converted to float32:", convert_to_float32(arr))
print("5. 'arr' converted to float32 to reduce precision:", convert_to_float32_precision(arr))
print("6. Attributes of 'arr':", array_attributes(arr))
print("7. Dimensionality of 'arr':", array_dimension(arr))
print("8. Item size and total size of 'arr':", item_size_info(arr))
print("9. Shape and strides of 'arr':", shape_stride_relationship(arr))
print("10. Zeros array of 5 elements:", create_zeros_array(5))
print("11. Ones matrix of size 3x3:", create_ones_matrix(3, 3))
print("12. Range array:", generate_range_array(1, 10, 2))
print("13. Linear space array:", generate_linear_space(1, 10, 5))
print("14. Identity matrix of size 4x4:", create_identity_matrix(4))
print("15. Numpy array from list:", list_to_numpy_array([1, 2, 3]))
print("16. Demo of numpy.view:", numpy_view_demo(arr))



import numpy as np
import pandas as pd

# Task 18
def concatenate_arrays(arr1, arr2, axis=0):
    return np.concatenate((arr1, arr2), axis=axis)

# Task 19
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8], [9, 10]])
concatenated_horizontal = np.concatenate((arr1, arr2), axis=1)

# Task 20
def vertical_stack_arrays(arrays):
    return np.vstack(arrays)

# Task 21
def create_array_with_range(start, stop, step):
    return np.arange(start, stop+1, step)

# Task 22
def create_equally_spaced_array(start, stop, num):
    return np.linspace(start, stop, num)

# Task 23
def create_log_spaced_array(start, stop, num):
    return np.logspace(np.log10(start), np.log10(stop), num)

# Task 24
def create_dataframe_from_array():
    data = np.random.randint(1, 101, size=(5, 3))
    return pd.DataFrame(data, columns=['A', 'B', 'C'])

# Task 25
def replace_negatives_with_zeros(df, column):
    df[column] = np.where(df[column] < 0, 0, df[column])
    return df

# Task 26
def access_3rd_element(arr):
    return arr[2]

# Task 27
def access_element_1_2(arr):
    return arr[1, 2]

# Task 28
def extract_elements_greater_than_5(arr):
    return arr[arr > 5]

# Task 29
def basic_slicing(arr):
    return arr[2:6]

# Task 30
def slice_sub_array(arr):
    return arr[1:3, :2]

# Test the functions
print("18. Concatenated arrays:", concatenate_arrays(arr1, arr2, axis=1))
print("19. Horizontally concatenated array:", concatenated_horizontal)
print("20. Vertically stacked arrays:")
print(vertical_stack_arrays([arr1, arr2]))
print("21. Array with range:", create_array_with_range(1, 10, 2))
print("22. Equally spaced array:", create_equally_spaced_array(0, 1, 10))
print("23. Logarithmically spaced array:", create_log_spaced_array(1, 1000, 5))
print("24. Pandas DataFrame:")
print(create_dataframe_from_array())
print("25. DataFrame with replaced negatives:")
df = pd.DataFrame({'A': [1, -2, 3, -4, 5]})
print(replace_negatives_with_zeros(df, 'A'))
print("26. 3rd element:", access_3rd_element(arr))
print("27. Element at index (1, 2):", access_element_1_2(arr_2d))
print("28. Elements greater than 5:", extract_elements_greater_than_5(arr))
print("29. Sliced array:", basic_slicing(arr))
print("30. Sub-array:", slice_sub_array(arr_2d))



import numpy as np

# Task 31
def extract_elements_by_indices(arr, indices):
    return arr[indices]

# Task 32
def filter_elements_greater_than_threshold(arr, threshold):
    return arr[arr > threshold]

# Task 33
def extract_elements_from_3d_array(arr, indices_dim1, indices_dim2, indices_dim3):
    return arr[indices_dim1, indices_dim2, indices_dim3]

# Task 34
def extract_elements_with_conditions(arr):
    return arr[(arr > 3) & (arr < 7)]

# Task 35
def extract_elements_by_indices_2d(arr, row_indices, col_indices):
    return arr[row_indices, col_indices]

# Task 36
def add_scalar_to_array(arr, scalar):
    return arr + scalar

# Task 37
def multiply_rows_by_elements(arr1, arr2):
    return arr1 * arr2

# Task 38
def add_array_to_each_row(arr1, arr2):
    return arr1 + arr2

# Task 39
def add_arrays_using_broadcasting(arr1, arr2):
    return arr1 + arr2

# Task 40
def multiply_using_broadcasting(arr1, arr2):
    return arr1 * arr2

# Task 41
def column_wise_mean(arr):
    return np.mean(arr, axis=0)

# Task 42
def max_value_in_each_row(arr):
    return np.max(arr, axis=1)

# Task 43
def indices_of_max_value_in_each_column(arr):
    return np.argmax(arr, axis=0)

# Task 44
def moving_sum_along_rows(arr):
    return np.apply_along_axis(lambda x: np.convolve(x, np.ones(3), mode='valid'), axis=1, arr=arr)

# Task 45
def check_if_all_elements_in_each_column_are_even(arr):
    return np.all(arr % 2 == 0, axis=0)

# Test the functions with provided arrays
arr_31 = np.array([[1, 2, 3], [4, 5, 6]])
arr_32 = np.array([[1, 2, 3], [4, 5, 6]])
arr_33 = np.array([[1, 2, 3], [4, 5, 6]])
arr_34 = np.array([[1, 2, 3], [4, 5, 6]])
arr_35 = np.array([[2, 4, 6], [3, 5, 7]])

print("31. Elements extracted by indices:", extract_elements_by_indices(arr_31, [0, 1]))
print("32. Elements greater than threshold:", filter_elements_greater_than_threshold(arr_32, 3))
print("33. Elements extracted from 3D array:", extract_elements_from_3d_array(arr_33, [0, 1], [0, 1], [1, 2]))
print("34. Elements with conditions:", extract_elements_with_conditions(arr_34))
print("35. Elements extracted by indices (2D):", extract_elements_by_indices_2d(arr_35, [0, 1], [1, 2]))
print("36. Scalar added to array:", add_scalar_to_array(arr_31, 5))
print("37. Rows multiplied by elements:", multiply_rows_by_elements(arr_31, arr_35))
print("38. Array added to each row:", add_array_to_each_row(arr_31.reshape(1, 3), arr_35))
print("39. Arrays added using broadcasting:", add_arrays_using_broadcasting(arr_33.reshape(3, 1), arr_34))
print("40. Arrays multiplied using broadcasting:", multiply_using_broadcasting(arr_31.reshape(2, 3), arr_32.reshape(2, 2)))
print("41. Column wise mean:", column_wise_mean(arr_31))
print("42. Maximum value in each row:", max_value_in_each_row(arr_35))
print("43. Indices of maximum value in each column:", indices_of_max_value_in_each_column(arr_34))
print("44. Moving sum along rows:", moving_sum_along_rows(arr_31))
print("45. Check if all elements in each column are even:", check_if_all_elements_in_each_column_are_even(arr_35))


import numpy as np

# Task 46
def reshape_array(arr, m, n):
    return np.reshape(arr, (m, n))

# Task 47
def flatten_matrix(matrix):
    return matrix.flatten()

# Task 48
def concatenate_arrays(arr1, arr2, axis):
    return np.concatenate((arr1, arr2), axis=axis)

# Task 49
def split_array(arr, num_sections, axis):
    return np.split(arr, num_sections, axis=axis)

# Task 50
def insert_and_delete_elements(arr, indices_insert, values_insert, indices_delete):
    arr = np.insert(arr, indices_insert, values_insert)
    arr = np.delete(arr, indices_delete)
    return arr

# Task 51
def element_wise_addition(arr1, arr2):
    return arr1 + arr2

# Task 52
def element_wise_subtraction(arr1, arr2):
    return arr1 - arr2

# Task 53
def element_wise_multiplication(arr1, arr2):
    return arr1 * arr2

# Task 54
def element_wise_division(arr1, arr2):
    return arr1 / arr2

# Task 55
def element_wise_exponentiation(arr1, arr2):
    return arr1 ** arr2

# Task 56
def count_substring_occurrences(arr, substring):
    return np.char.count(arr, substring)

# Task 57
def extract_uppercase_characters(arr):
    return np.char.upper(arr)

# Test the functions with provided arrays
arr_51 = np.random.randint(1, 100, size=(3, 3))
arr_52 = np.arange(10, 0, -1)
arr_53 = np.random.randint(1, 100, size=(3, 3))
arr_54 = np.arange(2, 11, 2)
arr_55 = np.arange(1, 6)
arr_56 = np.array(['hello', 'world', 'hello', 'numpy', 'hello'])
arr_57 = np.array(['Hello', 'World', 'OpenAI', 'GPT'])

print("46. Reshaped array:\n", reshape_array(arr_51, 2, 4))
print("47. Flattened matrix:", flatten_matrix(arr_51))
print("48. Concatenated arrays along axis 0:\n", concatenate_arrays(arr_51, arr_53, axis=0))
print("49. Split array along axis 1 into 3 sub-arrays:\n", split_array(arr_51, 3, axis=1))
print("50. Insert and delete elements:", insert_and_delete_elements(arr_52, [2, 5], [99, 99], [0, 7]))
print("51. Element-wise addition:\n", element_wise_addition(arr_51, arr_53))
print("52. Element-wise subtraction:\n", element_wise_subtraction(arr_52, np.arange(1, 11)))
print("53. Element-wise multiplication:\n", element_wise_multiplication(arr_51, arr_53))
print("54. Element-wise division:\n", element_wise_division(arr_54, np.arange(1, 6)))
print("55. Element-wise exponentiation:\n", element_wise_exponentiation(arr_55, arr_55[::-1]))
print("56. Count substring occurrences:", count_substring_occurrences(arr_56, 'hello'))
print("57. Extract uppercase characters:\n", extract_uppercase_characters(arr_57))


import numpy as np

# Task 58
def replace_substring(arr, old_substring, new_substring):
    return np.char.replace(arr, old_substring, new_substring)

# Task 59
def concatenate_strings(arr1, arr2):
    return np.char.add(arr1, arr2)

# Task 60
def longest_string_length(arr):
    return max(len(string) for string in arr)

# Task 61
dataset = np.random.randint(1, 1001, size=100)
mean = np.mean(dataset)
median = np.median(dataset)
variance = np.var(dataset)
std_deviation = np.std(dataset)

# Task 62
random_numbers = np.random.randint(1, 101, size=50)
percentile_25 = np.percentile(random_numbers, 25)
percentile_75 = np.percentile(random_numbers, 75)

# Task 63
set1 = np.random.rand(10)
set2 = np.random.rand(10)
correlation_coefficient = np.corrcoef(set1, set2)

# Task 64
matrix1 = np.random.rand(3, 3)
matrix2 = np.random.rand(3, 3)
matrix_multiplication_result = np.dot(matrix1, matrix2)

# Task 65
array_percentiles = np.percentile(np.random.randint(10, 1001, size=50), [10, 50, 90])

# Task 66
index_of_element = np.where(np.random.randint(1, 101, size=50) == 42)

# Task 67
random_array = np.random.randint(1, 101, size=50)
sorted_array = np.sort(random_array)

# Task 68
filtered_array = random_array[random_array > 20]

# Task 69
divisible_by_3 = random_array[random_array % 3 == 0]

# Task 70
between_20_and_40 = random_array[(random_array >= 20) & (random_array <= 40)]

# Task 71
byte_order = random_array.dtype.byteorder


import numpy as np

# Task 72
def perform_byte_swapping(arr):
    arr.byteswap(inplace=True)

# Task 73
def swap_byte_order(arr):
    new_arr = arr.newbyteorder()
    return new_arr

# Task 74
def conditional_byte_swap(arr):
    if arr.dtype.byteorder == '=':
        return arr
    else:
        return arr.newbyteorder()

# Task 75
def check_byte_swapping_necessary(arr):
    return arr.dtype.byteorder != '='

# Task 76
arr1 = np.arange(1, 11)
copy_arr = arr1.copy()
copy_arr[0] = 100
# Modifying copy_arr does not affect arr1

# Task 77
matrix = np.random.randint(1, 10, size=(3, 3))
view_slice = matrix[:2, :2]
view_slice[0, 0] = 100
# Modifying view_slice changes the original matrix

# Task 78
array_a = np.arange(1, 13).reshape(4, 3)
view_b = array_a[:2, :2]
view_b += 5
# Modifying view_b alters the original array_a

# Task 79
orig_array = np.arange(1, 9).reshape(2, 4)
reshaped_view = orig_array.reshape(4, 2)
reshaped_view[0, 0] = 100
# Modifying reshaped_view reflects changes in the original orig_array

# Task 80
data = np.random.randint(1, 10, size=(3, 4))
data_copy = data[data > 5].copy()
data_copy[0, 0] = 100
# Modifying data_copy does not affect the original data

# Task 81
A = np.random.randint(1, 10, size=(3, 3))
B = np.random.randint(1, 10, size=(3, 3))
addition_result = A + B
subtraction_result = A - B

# Task 82
C = np.random.randint(1, 10, size=(3, 2))
D = np.random.randint(1, 10, size=(2, 4))
multiplication_result = np.dot(C, D)

# Task 83
E = np.random.randint(1, 10, size=(3, 3))
transposed_E = np.transpose(E)

# Task 84
F = np.random.randint(1, 10, size=(3, 3))
determinant_F = np.linalg.det(F)

# Task 85
G = np.random.randint(1, 10, size=(3, 3))
try:
    inverse_G = np.linalg.inv(G)
except np.linalg.LinAlgError:
    inverse_G = "Matrix is singular, cannot compute its inverse."
