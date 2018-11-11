from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# importing the Iris dataset -- only use dataset with target value 0 and 1
iris = datasets.load_iris()
X = iris.data[:100,:]
X = np.insert(X, [4], [[1]]*100, axis=1)
y = iris.target[:100]

# split data
np.random.seed(1)
shuffle = np.insert(X, [5], np.reshape(y,(100,1)), axis=1)
np.random.shuffle(shuffle)
X = shuffle[:, :5]
y = shuffle[:, 5]

X_train = X[:80]
y_train = y[:80]
X_test = X[80:100]
y_test = y[80:100]

def cost(theta, X, y):
    sum = 0
    for i in range(len(X)):
        sum += y[i] * np.log(1/(1 + np.exp(-np.sum(X[i]*theta)))) + (1-y[i]) * np.log(1 - 1/(1 + np.exp(-np.sum(X[i]*theta))))
    return (-1/len(X)) * sum

def derivative(theta, X, y, j):
    sum = 0
    for i in range(len(X)):
        sum += (1/(1+np.exp(-np.sum(X[i]*theta))) - y[i])*X[i, j]
    return sum

def accuracy(theta, X_train, y_train, X_test, y_test):
    prediction = 1/(1+np.exp(-np.matmul(X_train, theta)))
    train_accuracy = 100 - np.sum(np.power(np.round(prediction)-y_train, 2)) / len(y_train) * 100
    prediction = 1/(1+np.exp(-np.matmul(X_test, theta)))
    test_accuracy = 100 - np.sum(np.power(np.round(prediction)-y_test, 2)) / len(y_test) * 100
    return np.round(train_accuracy, 2), np.round(test_accuracy, 2)

alpha = 0.1
np.random.seed(1)
theta = np.random.rand(5)


iter = 1
iter_cost = []
iter_accuracy = []
while True :
    temp = [0] * 5
    for j in range(len(theta)):
        temp[j] = theta[j] - alpha * (1/len(X)) * derivative(theta, X_train, y_train, j)
    theta = temp.copy()
    tr_a, te_a = accuracy(theta, X_train, y_train, X_test, y_test)
    print('iter', str(iter), ':', cost(theta, X_train, y_train), "/ train set Accuracy :", tr_a,"%","/ test set Accuracy :", te_a,"%")
    iter_cost.append([iter, cost(theta, X_train, y_train), cost(theta, X_test, y_test)])
    iter_accuracy.append([iter, tr_a, te_a])
    iter += 1
    if cost(theta, X_train, y_train) < 0.05 :
        if cost(theta, X_train, y_train) > 0.1 :
            print('The model may be overfitted.')
        print(theta)
        break



iter_cost = np.array(iter_cost)
j_train = iter_cost[:,[0,1]]
j_test = iter_cost[:,[0,2]]
plt.figure(0)
plt.plot(j_train[:,0], j_train[:,1], label='train-set')
plt.plot(j_test[:,0], j_test[:,1], label='test-set')
plt.title("Cost Curve for Logistic Regression")
plt.xlabel("number of iterations")
plt.ylabel("cost")
plt.ylim(0,3)
plt.legend()
plt.savefig('logistic_regression__cost_curve.png', bbox_inches='tight')

iter_accuracy = np.array(iter_accuracy)
acc_train = iter_accuracy[:,[0,1]]
acc_test = iter_accuracy[:,[0,2]]
plt.figure(1)
plt.plot(acc_train[:,0], acc_train[:,1], label='train-set')
plt.plot(acc_test[:,0], acc_test[:,1], label='test-set')
plt.title("Accuracy Curve for Logistic Regression")
plt.xlabel("number of iterations")
plt.ylabel("accuracy (%)")
plt.legend()
plt.savefig('logistic_regression__accuracy_curve.png', bbox_inches='tight')
