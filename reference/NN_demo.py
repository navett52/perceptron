'''
This sample shows a very simple neural network using Python
this neural network contains 4-dimensional inputs

0, 0, 1, 0  --> 0
1, 1, 1, 0  --> 1
1, 0, 1, 1  --> 1
0, 1, 1, 1  --> 0
1, 0, 0, 1  --> 1
our goal is to predict a test sample [1, 0, 0, 0] ? (the label is 1)

CSC 425/525 Artificial Intelligence

'''
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork_sample():
    def __init__(self):
        np.random.seed(1)
        self.weights = 2 * np.random.rand(4, 1) - 1 #range -1 to 1

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, x):
        #Note that the derivative of sigmoid equals to output * (1-output)
        #The weight adjust value equals to error * input * output * (1-output)
        return x * (1-x)

    def forward_process(self, X):
        return self.sigmoid(np.dot(X, self.weights))

    def backpropogation_process(self, X, y, predict):
        error = y - predict
        adjustment = np.dot(X.T, error * self.sigmoid_derivative(predict))
        self.weights += adjustment
        return self.weights

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            predict = self.forward_process(X)
            self.weights = self.backpropogation_process(X, y, predict)

        # display predicted results
        print("Predicted results:\n", self.forward_process(X))

    def predict(self, test):
        predict = self.forward_process(test)
        return np.where(predict <= 0.5, 0, 1)



NN = NeuralNetwork_sample()
#print("Initial random weights: ", NN.weights)
'''
x = np.linspace(-10, 10, 100)
plt.plot(x, NN.sigmoid(x))
plt.grid()
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.show()
'''

# training dataset -- input train and label pairs
training_X = np.array([[0, 0, 1, 0], [1, 1, 1, 0], [1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]])
training_y = np.array([[0, 1, 1, 0, 1]]).T

# start to training the neural network
NN.train(training_X, training_y, 50)

print("Weights after training:\n", NN.weights)

# training accuracy performed on the training set
print("The predicted result for training samples are\n", NN.predict(training_X))

# prediction a test sample [1, 0, 0, 0] which has a true label = 1
print("The predicted result for test sample [1, 0, 0, 0] is\n", NN.predict(np.array([1, 0, 0, 0])))