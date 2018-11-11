import numpy as np
from matplotlib import pyplot as plt

SEED = 2
# Equation to generate dataset
# : y = 0.5*X[0] + 0.8*X[1] - 0.2*X[2] + 1.3*X[3] - 0.7
np.random.seed(SEED)
X = np.random.rand(1000,4)
X = np.insert(X, [4], [[1]]*1000, axis=1)
y = [0.5*x[0] + 0.8*x[1] - 0.2*x[2] + 1.3*x[3] - 0.7  for x in X]

# Adding noise to y
np.random.seed(SEED)
epsilon = np.random.randn(1,1000) * 0.0003
y = y + epsilon[0]

# Modeling
# Initialize parameters
np.random.seed(SEED)
param = np.random.rand(5)
alpha = 0.5
gamma = 0.01
def cost(X, param) : return 1/(2*len(X)) * (np.sum(np.power((np.matmul(X, param) - y), 2)) + gamma * np.sum(np.power(param, 2)))

iter = 1
iter_cost = []
while True:
    temp = [0]*len(param)
    for i in range(len(param)) :
        t = param[i] - alpha * ((1/len(X)) * np.sum((np.matmul(X, param) - y) * X[: , i]) + min(np.abs(i-4), 1) * gamma/len(X) * param[i])
        temp[i] = t
    param = temp.copy()
    print('iter',str(iter),':',cost(X, param))
    iter_cost.append([iter,cost(X, param)])
    iter += 1

    if cost(X, param) < .0002 :
        print(param)
        break

# Plot Learning Curve
plt.plot(iter_cost)
plt.title('Cost Curve for Regularized Gradient Descent')
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.ylim((0,0.1))
plt.savefig('regularized_gradient_descent__cost_curve.png', bbox_inches='tight')