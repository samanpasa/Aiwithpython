import numpy as np

# Define the matrix A
A = np.array([
    [1, 2, 3],
    [0, 1, 4],
    [5, 6, 0]
])

# Calculate the inverse of A
A_inv = np.linalg.inv(A)

# Verify by calculating A * A_inv and A_inv * A
identity1 = np.dot(A, A_inv)
identity2 = np.dot(A_inv, A)

# Print the results
print("Matrix A:")
print(A)

print("\nInverse of A:")
print(A_inv)

print("\nA * A_inv:")
print(identity1)

print("\nA_inv * A:")
print(identity2)

# Check if the results are close to the identity matrix
print("\nIs A * A_inv close to identity matrix?", np.allclose(identity1, np.eye(3)))
print("Is A_inv * A close to identity matrix?", np.allclose(identity2, np.eye(3)))
