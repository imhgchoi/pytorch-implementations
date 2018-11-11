import numpy as np


# Equation to generate dataset
# : y = 0.5*X[0] + 0.8*X[1] - 0.2*X[2] + 1.3*X[3] - 0.7
X = np.random.rand(1000,4)
X = np.insert(X, [4], [[1]]*1000, axis=1)
y = [0.5*x[0] + 0.8*x[1] - 0.2*x[2] + 1.3*x[3] - 0.7  for x in X]

# Theta = (X' * X)^(-1) * X^T * y
Theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(X), X)), np.matrix.transpose(X)), y)

print(Theta)
## OUTPUT : [ 0.5  0.8 -0.2  1.3 -0.7]

