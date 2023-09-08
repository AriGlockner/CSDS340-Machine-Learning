import numpy as np

# a
print('a)')

a = np.array([[2] * 6] * 4)
print(a)

# b
print('\nb)')
b = np.eye(4, 6, dtype=int) * 2 + np.ones((4, 6), int)
print(b)

# c
print('\nc)')

# Will work because a * b just multiplies the values in the same position of the two matrices
print(a * b)

# Will not work because the n dimension of a (6) is not the same as the m dimension of b (4).
# Since 6 != 4, the dot product cannot be computed.
# print(np.dot(a, b))

# d
print('\nd)')
# a^T is a 6x4 matrix and b is a 4x6 matrix, so the dot product will be a 6x6 matrix
print(np.dot(a.T, b))
print()
# a is a 4x6 matrix and b^T is a 6x4 matrix, so the dot product will be a 4x4 matrix
print(np.dot(a, b.T))

# I wrote out the matrix multiplication by hand. Let me know if that's needed.
# Not including due to not being a py file type
