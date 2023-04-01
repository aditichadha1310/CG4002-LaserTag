import numpy as np

def numpy_to_cpp_array(np_array, destination_file):
    cpp_array = ""
    if np_array.ndim == 2:
        rows, cols = np_array.shape
        for i in range(rows):
            cpp_array += "{"
            for j in range(cols):
                cpp_array += str(np_array[i][j])
                if j < cols - 1:
                    cpp_array += ", "
            cpp_array += "}"
            if i < rows - 1:
                cpp_array += ",\n"

    else:
        for i in range(len(np_array)):
            cpp_array += str(np_array[i])
            if i < len(np_array) -1:
                cpp_array += ",\n"

    with open(destination_file, "w") as file:
        file.write(cpp_array)
    return cpp_array

# np_array = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9],[43,4,5]])
np_array = np.array([1, 2, 3])

print(np_array)
destination_file = "/Users/edly/Documents/GitHub/CG4002-LaserTag/hardware_ai/new/data_processing/myfile.txt"
cpp_array = numpy_to_cpp_array(np_array, destination_file)
print(cpp_array)
