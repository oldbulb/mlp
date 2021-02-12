import numpy as np
import matplotlib.pyplot as plt
import Data
import NeuralNetworkFrame

# Dataset
N = 200
Round = 2
Radius = 0.2
Bias = 0.02
Variation = 0.01

D = Data.DATA()
dataset = D.HelixWithNoise(N, Round, Radius, Bias, Variation)

# NeuralNetwork
Inputnum, Hiddennum, Outputnum = 2, 100, 2
learningRate = 0.03

#Momentum
Momentum = 0.9

#AdaGrad
AdaGradtheta = 0.00000001

# RMSProp
RMSPropRate = 0.8

#Adam
Rate1 = 0.9
Rate2 = 0.999
theta = 0.00000001

nn = NeuralNetworkFrame.NeuralNetwork_3(Inputnum, Hiddennum, Outputnum, learningRate, Momentum)

# Train
epochs = 900
batchs = 50
Loss = []
Accurancy = []

for epoch in range(1,epochs+1):
    # Batch dataset
    dataset_lists = []
    labelset_lists = []
    for batch in range(batchs):
        np.random.shuffle(dataset)
        dataset_lists.append([dataset[batch - 1][0], dataset[batch - 1][1]])
        labelset_lists.append(dataset[batch - 1][2])

    # Training
    nn.train(dataset_lists, labelset_lists)
    Loss.append(nn.train(dataset_lists, labelset_lists))
    Accurancy.append(nn.test(dataset_lists, labelset_lists))

print('Fianl Loss =', nn.train(dataset_lists, labelset_lists))


TrainEpoch = np.array(range(1,epochs+1))
LossFunction = np.array(Loss)
plt.plot(TrainEpoch, LossFunction, c='b',linewidth=1.0)
plt.show()

AccurancyRate = np.array(Accurancy)
plt.plot(TrainEpoch, AccurancyRate, c='orange',linewidth=1.0)
plt.show()

# Test
DataTest = D.Helix(N, Round, Radius, Bias)
test_data = []
test_label = []
for k in range(N+1):
    test_data.append([DataTest[k - 1][0], DataTest[k - 1][1]])
    test_label.append(DataTest[k - 1][2])
DataTestAccuracy = nn.test(test_data,test_label)
print('Test Accuracy = ', DataTestAccuracy)

# Decision boundary
x_min = -0.45
x_max = 0.45
y_min = -0.4
y_max = 0.4
h = 0.001
xx = np.arange(x_min, x_max, h)
yy = np.arange(y_min, y_max, h)

x_y =[]
for i in range(len(xx)):
    for j in range(len(yy)):
        x_y.append([xx[i],yy[j]])
X_Y = np.array(x_y,ndmin=2)
z = np.dot(np.array([0,1],ndmin=2),nn.query(X_Y))
Z = z.reshape(len(xx),len(yy)).T

X, Y = np.meshgrid(xx, yy)

contour = plt.contourf(X,Y,Z,10)
D.Scatter(DataTest, N)













