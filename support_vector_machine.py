import numpy as np
from scipy.optimize import minimize
from Tkinter import *
import math


root = Tk()
canvas = Canvas(root, width=800, height=800)

canvas.pack()
root.update()


# kernel, when data not linearly separable
def kernel(u,v):
    x = float((np.dot(u, v.T)))/80

    # hyperbolic tangent
    # return math.tanh(x)

    # exponential
    #  return math.e**(np.linalg.norm(u-v)/10)

    # squared
    return x**2


# optimizing function for calculating optimal alphas
def objective(c):
    global x, y
    first = 0
    for alpha in c:
        first += alpha

    second = 0

    for i in range(len(x)):   
        for j in range(len(x)):
            # print y[i]*y[j]*c[i]*c[j]*(x[i]*x[j])
            second += y[i]*y[j]*c[i]*c[j]*kernel(x[i], x[j])

    return -(first - 0.5*second)


# constraint for sum of alpha*y
def constraint(c):
    global y
    con_sum = 0
    for i in range(len(c)):
        con_sum += c[i]*y[i]
    return con_sum


# function to do calculation based on SVM machine learning
def learn():
    global x, y
    n = 5
    c0 = (50, 50, 500, 1, 1)

    b = (0.0, 1.0 / (2 * n))
    bnds = (b, b, b, b, b)

    con = {'type': 'eq', 'fun': constraint}

    sol = minimize(objective, c0, method='SLSQP', bounds=bnds, constraints=con)
    c = sol.x

    max = -1
    max_pos = -1
    for i in range(n):
        if c[i] > max:
            max = c[i]
            max_pos = i

    b = 0
    for i in range(n):
        b += c[i] * y[i] * kernel(x[i], x[max_pos])
    b -= y[max_pos]
    print "final c", c
    return b, c

# final function to return, the final value of a point on plane
def classify(z):
    global c, y, x, b
    s = 0
    for i in range(len(x)):
        s += c[i]*y[i]*kernel(x[i],z)
    s -= b
    return s


# function that returns point (plus or minus), based on the position where user clicked
def click(event):
    global canvas, root
    x = float(event.x)
    y = float(event.y)
    z = np.array([x, y])
    print z, classify(z)
    if classify(z) > 0:
        canvas.create_oval(x, y, x + 5, y + 5, fill='#000000')
        canvas.create_text(x, y, text="+", tags="text")
    else:
        canvas.create_oval(x, y, x + 5, y + 5)
        canvas.create_text(x, y, text="-", tags="text")
    canvas.pack()
    root.update()
    return 0


# function to color the part of the plane, based on classifier
def parse_canvas():
    global canvas, root
    i=0
    while (i <= 300):
        j=0
        while(j<=300):
            z = np.array([i, j])
            if classify(z) > 0.0:
                canvas.create_oval(i, j, i + 4, j + 4,outline = 'SeaGreen1' ,fill='SeaGreen1')
            else:
                canvas.create_oval(i, j, i + 4, j + 4, outline = 'tomato2',fill='tomato2')
            canvas.pack()
            root.update()
            j += 4
        i+=4

y = (1, -1, 1, 1, -1)
x = np.array([[0+40,10+100],[30+100,0+100],[100+100,50+100],[90+100,10+100], [0+100,70+100]])

b,c = learn()
parse_canvas()

for i in range(len(x)):
    if y[i] ==1:
        canvas.create_oval(x[i][0]-5, x[i][1]-5, x[i][0]+5, x[i][1]+5, fill = '#000000')
    else:
        canvas.create_oval(x[i][0]-5, x[i][1]-5, x[i][0]+5, x[i][1]+5)
    canvas.create_text(x[i][0], x[i][1]+30, text=str(i), tags="text")

canvas.pack()
root.update()

canvas.bind("<Button-1>", click)
canvas.pack()
root.update()
canvas.pack()


root.mainloop()

