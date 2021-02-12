# NeuralNetwork
# Hidden activation function: tanh(x)
# # Output activation function: softmax(x)
# 1.Mini-batch SGD
# 2.SGD with Momentum
# 3.SGD with Nesterov Momentum
# 4.AdaGrad
# 5.RMSProp with Nesterov Momentum
# 6.Adam
import numpy as np
import matplotlib.pyplot as plt

# Function definition
def softmax(x):
    exp_x = np.exp(x-np.max(x,axis=0, keepdims=True))
    exp_x /= np.sum(exp_x, axis=0, keepdims=True)
    return exp_x
def ReLU(x):
    return np.maximum(0, x)
def tanh(x):
    y = np.tanh(x)
    return y

# 1.Mini-batch SGD
class NeuralNetwork_1:

    # Initialize NeuralNetwork
    def __init__(self, Inputnum, Hiddennum, Outputnum, learningRate):
        # Set the number of neurons in each layer
        self.Inputnum = Inputnum
        self.Hiddennum = Hiddennum
        self.Outputnum = Outputnum
        self.learningRate =learningRate

        # Initialization weight & bias
        self.W1 = np.random.normal(0.0, pow(2/(self.Hiddennum + self.Inputnum), -0.5),(self.Hiddennum, self.Inputnum))
        self.W2 = np.random.normal(0.0, pow(2/(self.Outputnum + self.Hiddennum), -0.5),(self.Outputnum, self.Hiddennum))
        self.b1 = np.zeros((self.Hiddennum,1))
        self.b2 = np.zeros((self.Outputnum, 1))

        # Set activation function
        self.activation_function_1 = lambda x: tanh(x)
        self.activation_function_2 = lambda x: softmax(x)

        pass

    # Train
    def train(self, dataset_lists, labelset_lists):
        # Forward propagation
        input = np.array(dataset_lists).T
        input_hidden = np.dot(self.W1, input) + self.b1
        hidden = self.activation_function_1(input_hidden)
        hidden_output = np.dot(self.W2, hidden) + self.b2
        output = self.activation_function_2(hidden_output)

        # Loss function
        N = output.shape[1]
        label = np.array(labelset_lists)
        target = np.array([label, 1 - label])
        loss = - np.trace(np.dot(target.T, np.log(output)))/N

        # Accuracy
        accuracy_label = 0
        for j in range(1, N):
            target_j = np.array(target.T[j - 1],ndmin=2)
            output_j = np.array(output.T[j - 1],ndmin=2).T
            correct = np.dot(target_j, output_j)
            if correct >= 0.5:
                accuracy_label += 1
        accuracy_rate = accuracy_label / N

        # Gradient
        output_gradient = output - target
        hidden_gradient = np.multiply(np.dot(self.W2.T, output_gradient), 1 - hidden * hidden)

        # Parameter update
        self.W2 += -self.learningRate * (np.dot(output_gradient, hidden.T)/N)
        self.b2 += -self.learningRate * (np.dot(output_gradient, np.ones((N, 1)))/N)
        self.W1 += -self.learningRate * (np.dot(hidden_gradient, input.T)/N)
        self.b1 += -self.learningRate * (np.dot(hidden_gradient, np.ones((N, 1)))/N)

        return loss

        pass

    # Test
    def test(self,dataset_lists, labelset_lists):
        # Forward propagation
        input = np.array(dataset_lists).T
        input_hidden = np.dot(self.W1, input) + self.b1
        hidden = self.activation_function_1(input_hidden)
        hidden_output = np.dot(self.W2, hidden) + self.b2
        output = self.activation_function_2(hidden_output)

        # Loss function
        N = output.shape[1]
        label = np.array(labelset_lists)
        target = np.array([label, 1 - label])
        loss = - np.trace(np.dot(target.T, np.log(output))) / N

        # Accuracy
        accuracy_label = 0
        for j in range(1, N):
            target_j = np.array(target.T[j - 1], ndmin=2)
            output_j = np.array(output.T[j - 1], ndmin=2).T
            correct = np.dot(target_j, output_j)
            if correct >= 0.5:
                accuracy_label += 1
        accuracy_rate = accuracy_label / N

        return accuracy_rate

    # Query
    def query(self,dataset_lists):
        # Forward propagation
        input = np.array(dataset_lists).T
        input_hidden = np.dot(self.W1, input) + self.b1
        hidden = self.activation_function_1(input_hidden)
        hidden_output = np.dot(self.W2, hidden) + self.b2
        output = self.activation_function_2(hidden_output)

        return output

    pass

# 2.SGD with Momentum
class NeuralNetwork_2:
    # Initialize NeuralNetwork
    def __init__(self, Inputnum, Hiddennum, Outputnum, learningRate, Momentum):
        # Set the number of neurons in each layer
        self.Inputnum = Inputnum
        self.Hiddennum = Hiddennum
        self.Outputnum = Outputnum
        self.Momentum = Momentum
        self.learningRate = learningRate

        # Initialization weight & bias
        self.W1 = np.random.normal(0.0, pow(2 / (self.Hiddennum + self.Inputnum), -0.5),
                                   (self.Hiddennum, self.Inputnum))
        self.W2 = np.random.normal(0.0, pow(2 / (self.Outputnum + self.Hiddennum), -0.5),
                                   (self.Outputnum, self.Hiddennum))
        self.b1 = np.zeros((self.Hiddennum, 1))
        self.b2 = np.zeros((self.Outputnum, 1))

        # Set activation function
        self.activation_function_1 = lambda x: tanh(x)
        self.activation_function_2 = lambda x: softmax(x)

        # Initial Speed
        self.V_W2 = 0
        self.V_b2 = 0
        self.V_W1 = 0
        self.V_b1 = 0

        pass

    # Train
    def train(self, dataset_lists, labelset_lists):

        # Forward propagation
        input = np.array(dataset_lists).T
        input_hidden = np.dot(self.W1, input) + self.b1
        hidden = self.activation_function_1(input_hidden)
        hidden_output = np.dot(self.W2, hidden) + self.b2
        output = self.activation_function_2(hidden_output)

        # Loss function
        N = output.shape[1]
        label = np.array(labelset_lists)
        target = np.array([label, 1 - label])
        loss = - np.trace(np.dot(target.T, np.log(output))) / N

        # Accuracy
        accuracy_label = 0
        for j in range(1, N):
            target_j = np.array(target.T[j - 1], ndmin=2)
            output_j = np.array(output.T[j - 1], ndmin=2).T
            correct = np.dot(target_j, output_j)
            if correct >= 0.5:
                accuracy_label += 1
        accuracy_rate = accuracy_label / N

        # Gradient
        output_gradient = output - target
        hidden_gradient = np.multiply(np.dot(self.W2.T, output_gradient), 1 - hidden * hidden)

        self.V_W2 = self.Momentum * self.V_W2 - self.learningRate * (np.dot(output_gradient, hidden.T) / N)
        self.V_b2 = self.Momentum * self.V_b2 - self.learningRate * (np.dot(output_gradient, np.ones((N, 1))) / N)
        self.V_W1 = self.Momentum * self.V_W1 - self.learningRate * (np.dot(hidden_gradient, input.T) / N)
        self.V_b1 = self.Momentum * self.V_b1 - self.learningRate * (np.dot(hidden_gradient, np.ones((N, 1))) / N)

        # Parameter update
        self.W2 += self.V_W2
        self.b2 += self.V_b2
        self.W1 += self.V_W1
        self.b1 += self.V_b1

        return loss

        pass

    # Test
    def test(self, dataset_lists, labelset_lists):
        # Forward propagation
        input = np.array(dataset_lists).T
        input_hidden = np.dot(self.W1, input) + self.b1
        hidden = self.activation_function_1(input_hidden)
        hidden_output = np.dot(self.W2, hidden) + self.b2
        output = self.activation_function_2(hidden_output)

        # Loss function
        N = output.shape[1]
        label = np.array(labelset_lists)
        target = np.array([label, 1 - label])
        loss = - np.trace(np.dot(target.T, np.log(output))) / N

        # Accuracy
        accuracy_label = 0
        for j in range(1, N):
            target_j = np.array(target.T[j - 1], ndmin=2)
            output_j = np.array(output.T[j - 1], ndmin=2).T
            correct = np.dot(target_j, output_j)
            if correct >= 0.5:
                accuracy_label += 1
        accuracy_rate = accuracy_label / N

        return accuracy_rate

        pass

    # Query
    def query(self, dataset_lists):
        # Forward propagation
        input = np.array(dataset_lists).T
        input_hidden = np.dot(self.W1, input) + self.b1
        hidden = self.activation_function_1(input_hidden)
        hidden_output = np.dot(self.W2, hidden) + self.b2
        output = self.activation_function_2(hidden_output)

        return output

# 3.SGD with Nesterov Momentum
class NeuralNetwork_3:
    # Initialize NeuralNetwork
    def __init__(self, Inputnum, Hiddennum, Outputnum, learningRate, Momentum):
        # Set the number of neurons in each layer
        self.Inputnum = Inputnum
        self.Hiddennum = Hiddennum
        self.Outputnum = Outputnum
        self.Momentum = Momentum

        self.learningRate = learningRate

        # Initialization weight & bias
        self.W1 = np.random.normal(0.0, pow(2 / (self.Hiddennum + self.Inputnum), -0.5),
                                   (self.Hiddennum, self.Inputnum))
        self.W2 = np.random.normal(0.0, pow(2 / (self.Outputnum + self.Hiddennum), -0.5),
                                   (self.Outputnum, self.Hiddennum))
        self.b1 = np.zeros((self.Hiddennum, 1))
        self.b2 = np.zeros((self.Outputnum, 1))

        # Set activation function
        self.activation_function_1 = lambda x: tanh(x)
        self.activation_function_2 = lambda x: softmax(x)

        # Initial Speed
        self.V_W2 = 0
        self.V_b2 = 0
        self.V_W1 = 0
        self.V_b1 = 0

        pass

    # Train
    def train(self, dataset_lists, labelset_lists):
        # Temporary update
        self.W2 = self.W2 + self.V_W2
        self.b2 = self.b2 + self.V_b2
        self.W1 = self.W1 + self.V_W1
        self.b1 = self.b1 + self.V_b1

        # Forward propagation
        input = np.array(dataset_lists).T
        input_hidden = np.dot(self.W1, input) + self.b1
        hidden = self.activation_function_1(input_hidden)
        hidden_output = np.dot(self.W2, hidden) + self.b2
        output = self.activation_function_2(hidden_output)

        # Loss function
        N = output.shape[1]
        label = np.array(labelset_lists)
        target = np.array([label, 1 - label])
        loss = - np.trace(np.dot(target.T, np.log(output))) / N

        # Accuracy
        accuracy_label = 0
        for j in range(1, N):
            target_j = np.array(target.T[j - 1], ndmin=2)
            output_j = np.array(output.T[j - 1], ndmin=2).T
            correct = np.dot(target_j, output_j)
            if correct >= 0.5:
                accuracy_label += 1
        accuracy_rate = accuracy_label / N

        # Gradient
        output_gradient = output - target
        hidden_gradient = np.multiply(np.dot(self.W2.T, output_gradient), 1 - hidden * hidden)

        # Speed update
        self.V_W2 = self.Momentum * self.V_W2 - self.learningRate * (np.dot(output_gradient, hidden.T) / N)
        self.V_b2 = self.Momentum * self.V_b2 - self.learningRate * (np.dot(output_gradient, np.ones((N, 1))) / N)
        self.V_W1 = self.Momentum * self.V_W1 - self.learningRate * (np.dot(hidden_gradient, input.T) / N)
        self.V_b1 = self.Momentum * self.V_b1 - self.learningRate * (np.dot(hidden_gradient, np.ones((N, 1))) / N)

        # Parameter update
        self.W2 += self.V_W2
        self.b2 += self.V_b2
        self.W1 += self.V_W1
        self.b1 += self.V_b1

        return loss

        pass

    # Test
    def test(self, dataset_lists, labelset_lists):
        # Forward propagation
        input = np.array(dataset_lists).T
        input_hidden = np.dot(self.W1, input) + self.b1
        hidden = self.activation_function_1(input_hidden)
        hidden_output = np.dot(self.W2, hidden) + self.b2
        output = self.activation_function_2(hidden_output)

        # Loss function
        N = output.shape[1]
        label = np.array(labelset_lists)
        target = np.array([label, 1 - label])
        loss = - np.trace(np.dot(target.T, np.log(output))) / N

        # Accuracy
        accuracy_label = 0
        for j in range(1, N):
            target_j = np.array(target.T[j - 1], ndmin=2)
            output_j = np.array(output.T[j - 1], ndmin=2).T
            correct = np.dot(target_j, output_j)
            if correct >= 0.5:
                accuracy_label += 1
        accuracy_rate = accuracy_label / N

        return accuracy_rate

        pass

    # Query
    def query(self, dataset_lists):
        # Forward propagation
        input = np.array(dataset_lists).T
        input_hidden = np.dot(self.W1, input) + self.b1
        hidden = self.activation_function_1(input_hidden)
        hidden_output = np.dot(self.W2, hidden) + self.b2
        output = self.activation_function_2(hidden_output)

        return output

# 4.AdaGrad
class NeuralNetwork_4:

    # Initialize NeuralNetwork
    def __init__(self, Inputnum, Hiddennum, Outputnum, learningRate, AdaGradtheta):
        # Set the number of neurons in each layer
        self.Inputnum = Inputnum
        self.Hiddennum = Hiddennum
        self.Outputnum = Outputnum
        self.learningRate = learningRate
        self.theta = AdaGradtheta

        # Initialization weight & bias
        self.W1 = np.random.normal(0.0, pow(2/(self.Hiddennum + self.Inputnum), -0.5),(self.Hiddennum, self.Inputnum))
        self.W2 = np.random.normal(0.0, pow(2/(self.Outputnum + self.Hiddennum), -0.5),(self.Outputnum, self.Hiddennum))
        self.b1 = np.zeros((self.Hiddennum,1))
        self.b2 = np.zeros((self.Outputnum, 1))

        # Set activation function
        self.activation_function_1 = lambda x: tanh(x)
        self.activation_function_2 = lambda x: softmax(x)

        # Cumulate the square of gradient
        self.r_W2 = 0
        self.r_b2 = 0
        self.r_W1 = 0
        self.r_b1 = 0

        pass

    # Train
    def train(self, dataset_lists, labelset_lists):
        # Forward propagation
        input = np.array(dataset_lists).T
        input_hidden = np.dot(self.W1, input) + self.b1
        hidden = self.activation_function_1(input_hidden)
        hidden_output = np.dot(self.W2, hidden) + self.b2
        output = self.activation_function_2(hidden_output)

        # Loss function
        N = output.shape[1]
        label = np.array(labelset_lists)
        target = np.array([label, 1 - label])
        loss = - np.trace(np.dot(target.T, np.log(output)))/N

        # Accuracy
        accuracy_label = 0
        for j in range(1, N):
            target_j = np.array(target.T[j - 1],ndmin=2)
            output_j = np.array(output.T[j - 1],ndmin=2).T
            correct = np.dot(target_j, output_j)
            if correct >= 0.5:
                accuracy_label += 1
        accuracy_rate = accuracy_label / N

        # Gradient
        output_gradient = output - target
        hidden_gradient = np.multiply(np.dot(self.W2.T, output_gradient), 1 - hidden * hidden)

        W2_gradient = np.dot(output_gradient, hidden.T)/N
        b2_gradient = np.dot(output_gradient, np.ones((N, 1)))/N
        W1_gradient = np.dot(hidden_gradient, input.T)/N
        b1_gradient = np.dot(hidden_gradient, np.ones((N, 1)))/N

        # Cumulate the square of gradient
        self.r_W2 += W2_gradient * W2_gradient
        self.r_b2 += b2_gradient * b2_gradient
        self.r_W1 += W1_gradient * W1_gradient
        self.r_b1 += b1_gradient * b1_gradient

        # Parameter update
        self.W2 += -(self.learningRate / (self.theta + np.power(self.r_W2,0.5)))*(W2_gradient)
        self.b2 += -(self.learningRate / (self.theta + np.power(self.r_b2,0.5)))*(b2_gradient)
        self.W1 += -(self.learningRate / (self.theta + np.power(self.r_W1,0.5)))*(W1_gradient)
        self.b1 += -(self.learningRate / (self.theta + np.power(self.r_b1,0.5)))*(b1_gradient)

        return loss

        pass

    # Test
    def test(self,dataset_lists, labelset_lists):
        # Forward propagation
        input = np.array(dataset_lists).T
        input_hidden = np.dot(self.W1, input) + self.b1
        hidden = self.activation_function_1(input_hidden)
        hidden_output = np.dot(self.W2, hidden) + self.b2
        output = self.activation_function_2(hidden_output)

        # Loss function
        N = output.shape[1]
        label = np.array(labelset_lists)
        target = np.array([label, 1 - label])
        loss = - np.trace(np.dot(target.T, np.log(output))) / N

        # Accuracy
        accuracy_label = 0
        for j in range(1, N):
            target_j = np.array(target.T[j - 1], ndmin=2)
            output_j = np.array(output.T[j - 1], ndmin=2).T
            correct = np.dot(target_j, output_j)
            if correct >= 0.5:
                accuracy_label += 1
        accuracy_rate = accuracy_label / N

        return accuracy_rate

    # Query
    def query(self,dataset_lists):
        # Forward propagation
        input = np.array(dataset_lists).T
        input_hidden = np.dot(self.W1, input) + self.b1
        hidden = self.activation_function_1(input_hidden)
        hidden_output = np.dot(self.W2, hidden) + self.b2
        output = self.activation_function_2(hidden_output)

        return output

    pass

# 5.RMSProp with Nesterov Momentum
class NeuralNetwork_5:

    # Initialize NeuralNetwork
    def __init__(self, Inputnum, Hiddennum, Outputnum, learningRate, Momentum, RMSPropRate, theta):
        # Set the number of neurons in each layer
        self.Inputnum = Inputnum
        self.Hiddennum = Hiddennum
        self.Outputnum = Outputnum
        self.learningRate = learningRate
        self.Momentum = Momentum
        self.RMSPropRate = RMSPropRate
        self.theta = theta

        # Initialization weight & bias
        self.W1 = np.random.normal(0.0, pow(2/(self.Hiddennum + self.Inputnum), -0.5),(self.Hiddennum, self.Inputnum))
        self.W2 = np.random.normal(0.0, pow(2/(self.Outputnum + self.Hiddennum), -0.5),(self.Outputnum, self.Hiddennum))
        self.b1 = np.zeros((self.Hiddennum,1))
        self.b2 = np.zeros((self.Outputnum, 1))

        # Set activation function
        self.activation_function_1 = lambda x: tanh(x)
        self.activation_function_2 = lambda x: softmax(x)

        # Initial Speed
        self.V_W2 = 0
        self.V_b2 = 0
        self.V_W1 = 0
        self.V_b1 = 0

        # Cumulate the square of gradient
        self.r_W2 = 0
        self.r_b2 = 0
        self.r_W1 = 0
        self.r_b1 = 0

        pass

    # Train
    def train(self, dataset_lists, labelset_lists):
        # Temporary update
        self.W2 = self.W2 + self.V_W2
        self.b2 = self.b2 + self.V_b2
        self.W1 = self.W1 + self.V_W1
        self.b1 = self.b1 + self.V_b1

        # Forward propagation
        input = np.array(dataset_lists).T
        input_hidden = np.dot(self.W1, input) + self.b1
        hidden = self.activation_function_1(input_hidden)
        hidden_output = np.dot(self.W2, hidden) + self.b2
        output = self.activation_function_2(hidden_output)

        # Loss function
        N = output.shape[1]
        label = np.array(labelset_lists)
        target = np.array([label, 1 - label])
        loss = - np.trace(np.dot(target.T, np.log(output)))/N

        # Accuracy
        accuracy_label = 0
        for j in range(1, N):
            target_j = np.array(target.T[j - 1],ndmin=2)
            output_j = np.array(output.T[j - 1],ndmin=2).T
            correct = np.dot(target_j, output_j)
            if correct >= 0.5:
                accuracy_label += 1
        accuracy_rate = accuracy_label / N

        # Gradient
        output_gradient = output - target
        hidden_gradient = np.multiply(np.dot(self.W2.T, output_gradient), 1 - hidden * hidden)

        W2_gradient = np.dot(output_gradient, hidden.T)/N
        b2_gradient = np.dot(output_gradient, np.ones((N, 1)))/N
        W1_gradient = np.dot(hidden_gradient, input.T)/N
        b1_gradient = np.dot(hidden_gradient, np.ones((N, 1)))/N

        # Cumulate the square of gradient
        self.r_W2 = self.r_W2 * self.RMSPropRate + (1-self.RMSPropRate ) * (W2_gradient * W2_gradient)
        self.r_b2 = self.r_b2 * self.RMSPropRate + (1-self.RMSPropRate ) * (b2_gradient * b2_gradient)
        self.r_W1 = self.r_W1 * self.RMSPropRate + (1-self.RMSPropRate ) * (W1_gradient * W1_gradient)
        self.r_b1 = self.r_b1 * self.RMSPropRate + (1-self.RMSPropRate ) * (b1_gradient * b1_gradient)

        # Speed update
        self.V_W2 = self.Momentum * self.V_W2 - (self.learningRate / (self.theta + np.power(self.r_W2,0.5))) * (W2_gradient)
        self.V_b2 = self.Momentum * self.V_b2 - (self.learningRate / (self.theta + np.power(self.r_b2,0.5))) * (b2_gradient)
        self.V_W1 = self.Momentum * self.V_W1 - (self.learningRate / (self.theta + np.power(self.r_W1,0.5))) * (W1_gradient)
        self.V_b1 = self.Momentum * self.V_b1 - (self.learningRate / (self.theta + np.power(self.r_b1,0.5))) * (b1_gradient)

        # Parameter update
        self.W2 += self.V_W2
        self.b2 += self.V_b2
        self.W1 += self.V_W1
        self.b1 += self.V_b1

        return loss

        pass

    # Test
    def test(self,dataset_lists, labelset_lists):
        # Forward propagation
        input = np.array(dataset_lists).T
        input_hidden = np.dot(self.W1, input) + self.b1
        hidden = self.activation_function_1(input_hidden)
        hidden_output = np.dot(self.W2, hidden) + self.b2
        output = self.activation_function_2(hidden_output)

        # Loss function
        N = output.shape[1]
        label = np.array(labelset_lists)
        target = np.array([label, 1 - label])
        loss = - np.trace(np.dot(target.T, np.log(output))) / N

        # Accuracy
        accuracy_label = 0
        for j in range(1, N):
            target_j = np.array(target.T[j - 1], ndmin=2)
            output_j = np.array(output.T[j - 1], ndmin=2).T
            correct = np.dot(target_j, output_j)
            if correct >= 0.5:
                accuracy_label += 1
        accuracy_rate = accuracy_label / N

        return accuracy_rate

    # Query
    def query(self,dataset_lists):
        # Forward propagation
        input = np.array(dataset_lists).T
        input_hidden = np.dot(self.W1, input) + self.b1
        hidden = self.activation_function_1(input_hidden)
        hidden_output = np.dot(self.W2, hidden) + self.b2
        output = self.activation_function_2(hidden_output)

        return output

    pass

# 6.Adam
class NeuralNetwork_6:

    # Initialize NeuralNetwork
    def __init__(self, Inputnum, Hiddennum, Outputnum, learningRate, Rate1, Rate2, theta):
        # Set the number of neurons in each layer
        self.Inputnum = Inputnum
        self.Hiddennum = Hiddennum
        self.Outputnum = Outputnum
        self.learningRate = learningRate
        self.Rate1 = Rate1
        self.Rate2 = Rate2
        self.theta = theta

        # Initialization First and Second order moment variable
        self.s_W2 = 0
        self.s_b2 = 0
        self.s_W1 = 0
        self.s_b1 = 0
        self.r_W2 = 0
        self.r_b2 = 0
        self.r_W1 = 0
        self.r_b1 = 0

        # Initialize Time step t=0
        self.t = 0

        # Initialization weight & bias
        self.W1 = np.random.normal(0.0, pow(2 / (self.Hiddennum + self.Inputnum), -0.5),
                                   (self.Hiddennum, self.Inputnum))
        self.W2 = np.random.normal(0.0, pow(2 / (self.Outputnum + self.Hiddennum), -0.5),
                                   (self.Outputnum, self.Hiddennum))
        self.b1 = np.zeros((self.Hiddennum, 1))
        self.b2 = np.zeros((self.Outputnum, 1))

        # Set activation function
        self.activation_function_1 = lambda x: tanh(x)
        self.activation_function_2 = lambda x: softmax(x)

        pass

    # Train
    def train(self, dataset_lists, labelset_lists):
        # Forward propagation
        input = np.array(dataset_lists).T
        input_hidden = np.dot(self.W1, input) + self.b1
        hidden = self.activation_function_1(input_hidden)
        hidden_output = np.dot(self.W2, hidden) + self.b2
        output = self.activation_function_2(hidden_output)

        # Loss function
        N = output.shape[1]
        label = np.array(labelset_lists)
        target = np.array([label, 1 - label])
        loss = - np.trace(np.dot(target.T, np.log(output))) / N

        # Accuracy
        accuracy_label = 0
        for j in range(1, N):
            target_j = np.array(target.T[j - 1], ndmin=2)
            output_j = np.array(output.T[j - 1], ndmin=2).T
            correct = np.dot(target_j, output_j)
            if correct >= 0.5:
                accuracy_label += 1
        accuracy_rate = accuracy_label / N

        # Gradient
        output_gradient = output - target
        hidden_gradient = np.multiply(np.dot(self.W2.T, output_gradient), 1 - hidden * hidden)

        W2_gradient = np.dot(output_gradient, hidden.T) / N
        b2_gradient = np.dot(output_gradient, np.ones((N, 1))) / N
        W1_gradient = np.dot(hidden_gradient, input.T) / N
        b1_gradient = np.dot(hidden_gradient, np.ones((N, 1))) / N

        self.t += 1

        # Update Biased first moment estimation
        self.s_W2 = self.Rate1 * self.s_W2 + (1-self.Rate1) * W2_gradient
        self.s_b2 = self.Rate1 * self.s_b2 + (1-self.Rate1) * b2_gradient
        self.s_W1 = self.Rate1 * self.s_W1 + (1-self.Rate1) * W1_gradient
        self.s_b1 = self.Rate1 * self.s_b1 + (1-self.Rate1) * b1_gradient
        self.r_W2 = self.Rate2 * self.r_W2 + (1-self.Rate2) * (W2_gradient * W2_gradient)
        self.r_b2 = self.Rate2 * self.r_b2 + (1-self.Rate2) * (b2_gradient * b2_gradient)
        self.r_W1 = self.Rate2 * self.r_W1 + (1-self.Rate2) * (W1_gradient * W1_gradient)
        self.r_b1 = self.Rate2 * self.r_b1 + (1-self.Rate2) * (b1_gradient * b1_gradient)

        # Correction of first moment deviation
        Correction_s_W2 = self.s_W2 / (1 - np.power(self.Rate1, self.t))
        Correction_s_b2 = self.s_b2 / (1 - np.power(self.Rate1, self.t))
        Correction_s_W1 = self.s_W1 / (1 - np.power(self.Rate1, self.t))
        Correction_s_b1 = self.s_b1 / (1 - np.power(self.Rate1, self.t))

        # Correction of second moment deviation
        Correction_r_W2 = self.r_W2 / (1 - np.power(self.Rate2, self.t))
        Correction_r_b2 = self.r_b2 / (1 - np.power(self.Rate2, self.t))
        Correction_r_W1 = self.r_W1 / (1 - np.power(self.Rate2, self.t))
        Correction_r_b1 = self.r_b1 / (1 - np.power(self.Rate2, self.t))

        # Parameter update
        self.W2 = self.W2 - self.learningRate * Correction_s_W2/(np.power(Correction_r_W2, 0.5)+self.theta)
        self.b2 = self.b2 - self.learningRate * Correction_s_b2/(np.power(Correction_r_b2, 0.5)+self.theta)
        self.W1 = self.W1 - self.learningRate * Correction_s_W1/(np.power(Correction_r_W1, 0.5)+self.theta)
        self.b1 = self.b1 - self.learningRate * Correction_s_b1/(np.power(Correction_r_b1, 0.5)+self.theta)

        return loss

        pass

    # Test
    def test(self, dataset_lists, labelset_lists):
        # Forward propagation
        input = np.array(dataset_lists).T
        input_hidden = np.dot(self.W1, input) + self.b1
        hidden = self.activation_function_1(input_hidden)
        hidden_output = np.dot(self.W2, hidden) + self.b2
        output = self.activation_function_2(hidden_output)

        # Loss function
        N = output.shape[1]
        label = np.array(labelset_lists)
        target = np.array([label, 1 - label])
        loss = - np.trace(np.dot(target.T, np.log(output))) / N

        # Accuracy
        accuracy_label = 0
        for j in range(1, N):
            target_j = np.array(target.T[j - 1], ndmin=2)
            output_j = np.array(output.T[j - 1], ndmin=2).T
            correct = np.dot(target_j, output_j)
            if correct >= 0.5:
                accuracy_label += 1
        accuracy_rate = accuracy_label / N

        return accuracy_rate

    # Query
    def query(self, dataset_lists):
        # Forward propagation
        input = np.array(dataset_lists).T
        input_hidden = np.dot(self.W1, input) + self.b1
        hidden = self.activation_function_1(input_hidden)
        hidden_output = np.dot(self.W2, hidden) + self.b2
        output = self.activation_function_2(hidden_output)

        return output

    pass



