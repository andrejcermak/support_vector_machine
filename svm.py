"""
Python file that implements support vector machine as class, uses tkinter GUI and gives option of clicking on plane
to get classification of point where user clicked.
"""

import numpy as np
from scipy.optimize import minimize
from Tkinter import *
import math


class SupportVectorMachine:
    def __init__(self, size,x,y, math_funct, gamma):
        b = (0.0, 1.0 / (2 * size))
        self.bnds = [b]*size
        self.math_funct = math_funct
        self.gamma = gamma
        self.c = []
        self.x = x
        self.y = y
        self.size = size

        self.root = Tk()
        self.canvas = Canvas(self.root, width=800, height=800)

        self.canvas.pack()
        self.root.update()

    def __constraint__(self, c):
        """Function for optimizer in scipy"""
        con_sum = 0
        for i in range(len(c)):
            con_sum += c[i] * self.y[i]
        return con_sum

    def __kernel__(self, u,v):
        """Kernel trick, math function is given on init, default returns linear"""
        x = float((np.dot(u, v.T)))/self.gamma

        # hyperbolic tangent
        if self.math_funct == "tanh":
            return math.tanh(x)

        # exponential
        if self.math_funct == "exp":
            return math.e**(np.linalg.norm(u-v)/10)

        # squared
        if self.math_funct == "cube":
            return x ** 2
        return x

    def __objective__(self, c):
        """Primary function for minimizer, but it maximizes the f(c1...cn), as explained on wiki"""
        first = 0
        for alpha in c:
            first += alpha

        second = 0
        for i in range(len(self.x)):
            for j in range(len(self.x)):
                # print y[i]*y[j]*c[i]*c[j]*(x[i]*x[j])
                second += self.y[i] * self.y[j] * c[i] * c[j] * self.__kernel__(self.x[i], self.x[j])

        return -(first - 0.5 * second)

    def learn(self):
        """
        Main function that handles whole svm learning, creates GUI
        """
        c0 = [10]*self.size

        con = {'type': 'eq', 'fun': self.__constraint__}

        sol = minimize(self.__objective__, c0, method='SLSQP', bounds=self.bnds, constraints=con)
        self.c = sol.x
        max = -1
        max_pos = -1
        for i in range(self.size):
            if self.c[i] > max:
                max = self.c[i]
                max_pos = i

        self.b = 0
        for i in range(self.size):
            self.b += self.c[i] * self.y[i] * self.__kernel__(self.x[i], self.x[max_pos])
        self.b -= self.y[max_pos]
        self.__parse_canvas__()
        self.__draw_points__()
        self.canvas.bind("<Button-1>", self.__click__)
        self.canvas.pack()
        self.root.mainloop()


    def classify(self, z):
        """
        Function that classifies given point on plane
        """

        result = 0
        for i in range(len(self.x)):
            result += self.c[i]*self.y[i]*self.__kernel__(self.x[i],z)
        result -= self.b
        return result

    def __draw_points__(self):
        """
        Draws points on plane with respect to their sign
        """
        for i in range(len(self.x)):
            if self.y[i] == 1:
                self.canvas.create_oval(self.x[i][0] - 5,self.x[i][1] - 5, self.x[i][0] + 5, self.x[i][1] + 5,
                                        fill='#000000')
            else:
                self.canvas.create_oval(self.x[i][0] - 5, self.x[i][1] - 5, self.x[i][0] + 5, self.x[i][1] + 5)
            self.canvas.create_text(self.x[i][0], self.x[i][1] + 30, text=str(i), tags="text")

    def __parse_canvas__(self):
        """
        Colors the plane with respect to built math model
        """
        i = 0
        while i <= 300:
            j = 0
            while j <= 300:
                z = np.array([i, j])
                if self.classify(z) > 0.0:
                    self.canvas.create_oval(i, j, i + 4, j + 4, outline='SeaGreen1', fill='SeaGreen1')
                else:
                    self.canvas.create_oval(i, j, i + 4, j + 4, outline='tomato2', fill='tomato2')
                self.canvas.pack()
                self.root.update()
                j += 4
            i += 4

    def __click__(self, event):
        """Function that handles user point input as clicking on the plane,
        draws filled or empty with respect to classification"""
        x = float(event.x)
        y = float(event.y)
        z = np.array([x, y])

        print z, self.classify(z)
        if self.classify(z) > 0:
            self.canvas.create_oval(x, y, x + 5, y + 5, fill='#000000')
            self.canvas.create_text(x, y, text="+", tags="text")
        else:
            self.canvas.create_oval(x, y, x + 5, y + 5)
            self.canvas.create_text(x, y, text="-", tags="text")
        self.canvas.pack()
        self.root.update()
        return 0



