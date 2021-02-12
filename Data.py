import numpy as np
import matplotlib.pyplot as plt
import random

class DATA:

    def Helix(self, N, Round, Radius, Bias):

        self.AngleVariation = 4 * np.pi * Round/ N
        self.RadiusVariation = Radius / (2 * np.pi)
        X = []
        Y = []
        P = []
        Q = []
        data1 = []
        data2 = []

        for i in range(1,int(N/2+1)):
            a = self.AngleVariation*(i-1)
            b = Bias + self.RadiusVariation * a
            X.append(-b * np.cos(a))
            Y.append(b * np.sin(a))
            P.append(b * np.cos(a))
            Q.append(-b * np.sin(a))
            data1.append([X[i - 1], Y[i - 1], 1])
            data2.append([P[i - 1], Q[i - 1], 0])

        data = data1 + data2

        return data

    def HelixWithNoise(self, N, Round, Radius, Bias, Variation):

        self.AngleVariation = 4 * np.pi * Round/ N
        self.RadiusVariation = Radius / (2 * np.pi)
        X = []
        Y = []
        P = []
        Q = []
        data1 = []
        data2 = []

        for i in range(1,int(N/2+1)):
            a = self.AngleVariation*(i-1)
            b = Bias + self.RadiusVariation * a
            X.append(-b * np.cos(a))
            Y.append(b * np.sin(a))
            P.append(b * np.cos(a))
            Q.append(-b * np.sin(a))
            data1.append([X[i - 1] + random.gauss(0,Variation), Y[i - 1] + random.gauss(0,Variation), 1])
            data2.append([P[i - 1] + random.gauss(0,Variation), Q[i - 1] + random.gauss(0,Variation), 0])

        data = data1 + data2


        return data

    def Scatter(self,data, N):
        X = []
        Y = []
        P = []
        Q = []
        self.data = data

        for i in range(1, int(N/2+1)):
            X.append(self.data[i-1][0])
            Y.append(self.data[i-1][1])
        for j in range(int(N/2+1),N+1):
            P.append(self.data[j-1][0])
            Q.append(self.data[j-1][1])

        plt.scatter(X, Y, s=20, c='crimson', marker='+')
        plt.scatter(P, Q, s=20, c='dodgerblue', marker='.')
        plt.show()

        pass













