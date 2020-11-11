import numpy as np

class Perceptron():
    def __init__(self):
        np.random.seed(1)
        # self.weights = 2 * np.random.rand(2, 1) - 1 #range -1 to 1
        self.weights = np.array([[1], [0.5], [-0.6]])
        self.learning_rate = .5
    
    def sign(self, X):
        predict = []
        for sum in X:
            if (sum < 0):
                predict.append(-1)
            else:
                predict.append(1)
        
        return np.asarray(predict)
    
    def forward_process(self, X):
        return self.sign(np.dot(X, self.weights))
    
    def backpropogation_process(self, X, y, predict):
        row_index = 0
        for row in X:
            adjustment = []
            error = y[row_index] - predict[row_index]
            for component in row:
                adjustment.append(self.learning_rate * error * component)
            
            adjustment_vector = np.asarray(adjustment)
            self.weights += adjustment_vector
            row_index += 1
        
        return self.weights
    
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            predict = self.forward_process(X)
            self.weights = self.backpropogation_process(X, y, predict)

        # display predicted results
        print("Predicted results:\n", self.forward_process(X))
    
    def predict(self, test):
        predict = self.forward_process(test)
        return np.where(predict <= 0, -1, 1)


perceptron = Perceptron()

# training dataset -- input train and label pairs
training_X = np.array([
    [1, 1.0, 1.0], [1, 9.4, 6.4], [1, 2.5, 2.1], [1, 8.0, 7.7], [1, 0.5, 2.2], [1, 7.9, 8.4], [1, 7.0, 7.0], [1, 2.8, 0.8], [1, 1.2, 3.0], [1, 7.8, 6.1]])
training_y = np.array([
    [1, -1, 1, -1, 1, -1, -1, 1, 1, -1]]).T

# start to training the neural network
perceptron.train(training_X, training_y, 50)

print("Weights after training:\n", perceptron.weights)

# training accuracy performed on the training set
print("The predicted result for training samples are\n", perceptron.predict(training_X))

# prediction a test sample [1, 10.7, -3.6] which has a false label = -1
print("The predicted result for test sample [1, 10.7, -3.6] is\n", perceptron.predict(np.array([1, 10.7, -3.6])))

q3_part3 = np.array([
    [1, 5.2, -6.8], [1, 8.3, 2.4], [1, -3.7, 5.6], [1, 1.3, 1.7], [1, 2.7, 7.2]
])

print("The predicted result for the generated set for the assignment is\n", perceptron.predict(q3_part3))