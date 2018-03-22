#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:06:55 2017
@author: preranasingh
"""



import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
import matplotlib.lines as mlines
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder

def sigmoid (x):
    return 1/(1 + np.exp(-x)) 

def derive_sigmoid(x):
    return x * (1 - x)

#input sample
x=np.array([[0.05,0.1]])
x

#output sample
y=np.array([[0.01,0.99]])
y


m=np.mean(x)

#input layer
ilayer_unit = x.shape[1] 

#hidden layer
hlayer_unit = 2 

#output layer
output_unit = 2 
t_iter=70000
learning_rate=0.1

np.random.seed(2)

#uniformly distributed value of theta in each layer
theta_1=numpy.random.uniform(0.0,1.0,size=(ilayer_unit,hlayer_unit))
theta_2=numpy.random.uniform(0.0,1.0,size=(hlayer_unit,output_unit))
bias_1=np.ones((1,hlayer_unit))
bias_2=np.ones((1,output_unit))
totalcost=[]
iteration_values=range(1,t_iter+1)
total_theta1=np.zeros((t_iter,4))
total_theta2=np.zeros((t_iter,4))
plot_theta1=np.zeros(t_iter)
plot_theta2=np.zeros(t_iter)


#iterations to calculate the value cost and update value of theta
for i in range(t_iter):
    
    #forward propgation
    hlayer_temp=np.dot(x,theta_1)
    hlayer_input=hlayer_temp + bias_1
    hlayer_activations = sigmoid(hlayer_input)
    olayer_temp=np.dot(hlayer_activations,theta_2)
    olayer_input= olayer_temp+ bias_2
    output = sigmoid(olayer_input)
    
    #calculating the error
    error=y-output
    
    #mean squared error for cost function
    cost=(np.mean((math.pow((error[:,0]),2)+math.pow((error[:,1]),2))))*0.5
    
    totalcost.append(cost)
    #backward propogation
    slope_olayer = derive_sigmoid(output)
    d_output = error * slope_olayer
    error_hidden_layer = d_output.dot(theta_2.T)
    slope_hlayer = derive_sigmoid(hlayer_activations)
    d_hiddenlayer = error_hidden_layer * slope_hlayer
    #updating theta2
    theta_2 += hlayer_activations.T.dot(d_output)*learning_rate
    #print theta_2
    total_theta2[i,0]=theta_2[0][0]
    total_theta2[i,1]=theta_2[1][0]
    total_theta2[i,2]=theta_2[0][1]
    total_theta2[i,3]=theta_2[1][1]
    
    #updateing theta 1
    theta_1 += x.T.dot(d_hiddenlayer)*learning_rate
    #print theta_1
    total_theta1[i,0]=theta_1[0][0]
    total_theta1[i,1]=theta_1[1][0]
    total_theta1[i,2]=theta_1[0][1]
    total_theta1[i,3]=theta_1[1][1]
    
    #taking mean values of theta at each iteration
    plot_theta1[i]=np.mean(total_theta1[i], axis=0)
    plot_theta2[i]=np.mean(total_theta2[i], axis=0)

print 'output value:',output

#Plot for cost vs no of iterations
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Total Cost vs Iterations Plot')
ax.plot(totalcost,iteration_values,color='blue')
ax.set_xlabel('Total cost')
ax.set_ylabel('Number of Iterationa')
fig.show()




#plot for theta1 paramater vs no of iterations
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Mean value of Theta for hidden layer vs Iterations Plot')
ax.plot(plot_theta1,iteration_values,color='red')
ax.set_xlabel('theta 1')
ax.set_ylabel('Number of Iterationa')
fig.show()

#plot for theta2 parameter vs no of iterations
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Mean Theta for output layer vs Iterations Plot')
ax.plot(plot_theta2,iteration_values,color='green')
ax.set_xlabel('Theta 2')
ax.set_ylabel('Number of Iterationa')
fig.show()

#plot for theta1 paramater vs no of iterations
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Theta for hidden layer vs Iterations Plot')
ax.plot(total_theta1,iteration_values,color='red')
ax.set_xlabel('theta 1')
ax.set_ylabel('Number of Iterationa')
fig.show()

#plot for theta2 parameter vs no of iterations
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Theta for output layer vs Iterations Plot')
ax.plot(total_theta2,iteration_values,color='c')
ax.set_xlabel('Theta 2')
ax.set_ylabel('Number of Iterationa')
fig.show()



###As seen from the output values after 70000 iterations the value of output comes out 
###to be similar as y

#