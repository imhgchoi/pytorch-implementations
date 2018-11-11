import numpy as np
from random import shuffle
from matplotlib import pyplot as plt

SEED = 3
# Equation to generate dataset
# : y = 0.5*X[0] + 0.8*X[1] - 0.2*X[2] + 1.3*X[3] - 0.7
np.random.seed(SEED)
X = np.random.rand(1000,4)
X = np.insert(X, [4], [[1]]*1000, axis=1)
y = [0.5*x[0] + 0.8*x[1] - 0.2*x[2] + 1.3*x[3] - 0.7  for x in X]

# Adding some noise to y
np.random.seed(SEED)
epsilon = np.random.randn(1,1000) * 0.0003
y = y + epsilon[0]

# Modeling
# Initialize parameters
np.random.seed(SEED)
param = np.random.rand(5)
alpha = 0.1
def cost(X, param) : return 1/len(X) * np.sum(np.power((np.matmul(X, param) - y), 2))

iter = 1
epoch = 1
iter_cost = []
while True:
    flag = False
    for i in range(len(X)):
        temp = [0]*len(param)
        for j in range(len(param)):
            t = param[j] - alpha * (np.sum(X[i] * param) - y[i]) * X[i, j]
            temp[j] = t
        param = temp.copy()
        print('iter',str(iter),', epoch',str(epoch),':',cost(X, param))
        iter += 1
        iter_cost.append([iter,cost(X, param)])

        if cost(X, param) < .0000001 :
            print(param)
            flag = True
            break
    epoch += 1

    #shuffle
    shuffle = np.insert(X, [5], np.reshape(y,(1000,1)), axis=1)
    np.random.shuffle(shuffle)
    X = shuffle[:, :5]
    y = shuffle[:, 5]
    if flag == True:
        break



# Plot Learning Curve
plt.plot(iter_cost)
plt.title('Cost Curve for Stochastic Gradient Descent')
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.ylim((0,0.5))
plt.savefig('stochastic_gradient_descent__cost_curve.png', bbox_inches='tight')