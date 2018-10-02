# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 11:39:47 2018

@author: Mohak
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ds = pd.read_csv('G:\Credit_Card_Applications.csv')

x = ds.iloc[:,:-1].values
y = ds.iloc[:,-1].values

# X is the input, it contains customer data
# Y is a vector which contains the binary value whether the application got accepted. It will be used to see how many fraud applications got accepted.
#it is not used as a output or comparision vector

#feature scaling to get all the features bw 0-1 (normalization)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
x = sc.fit_transform(x)

#training the som
#change the working dir to where the minisom.py is located 
#import minisom to make use of the SOM library
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1, learning_rate=0.5)
#x,y corrospond to diensions so here we create a map of 10x10. Dimensions of map are arbitiary so we can chose any dimensions we want
#input_len is the number of input params. here we take in consideration customerID because we will map the result with how many applications got accepted.
#sigma is the radius of neighbourhood (affected area)

#training the network
#we will initialize the weights of the map close to 0.
som.random_weights_init(x)
som.train_random(x,100)
#this function takes arguments data and epochs.

#visualizing the results
#making the maps using pylab because we are not using general representation style like bar graph, pie chart, etc
from pylab import bone, pcolor,colorbar, plot, show
#step1: Initialize the window or figure that will contain the map
bone()
pcolor(som.distance_map().T)
#distance_map function will return the matrix of all interneuron distances, we will take transpose of this matrix. pcolor will apply color range accordingly
#now we will classify the colors by providiong a color bar
colorbar()
#colorbar corrosponds to normaliesd interneuron distances
#the nodes with high mean-interneuron-distances are outliers, meaning that they are far from other neuron clusters(dark) and hence are the outliers
#now we'll add markers to see if the outliers got approval or not.
markers = ['o', 's']
colors = ['r', 'g']
#markers are circle(o) and square(s)
#colors are used to distince the approved applications and rejected applications
#we'll loop over all the customers and we'll color accordingly
for i, j in enumerate(x):
    #j is the row and i is the indexes of the database
    #we will take the winning node for all the customers by a .winner() method
    w = som.winner(j)
    #we will place the marker based on its approval status
    #the coordinated of center point of square are (0.5, 0.5) as each side of square has a dim of (0,1)
    plot(w[0]+0.5, w[1]+0.5, markers[y[i]], markeredgecolor=colors[y[i]], markerfacecolor='None',markersize=10, markeredgewidth=2)
    #markers[y[i]] : y[i] has binary values so if it is 0, it would correspond to markers[0] which is circle
    #if y[i] is 1, it would correspond to markers[1] which is a square
    #same logic for colours, depending on the values of y[i] the appropriate coloring would be done.
    #we will color only the edge of the marker because of there are 2 markers in a place, we would be able to see both of them
    #hence markerfacecolor is set to None
show()
#green square indicates that the cusstomers got approval
#red square indicates that the customers didnt get approval
#both the shapes : indicate that some customers got approval while some didnt, which is the case with out outliers.
#now we'll have to catch the potential cheaters(which are the outliers, squares which are white)
#we'll do it by using a reverse mapping function and identifying the customer ids of potential fraud customers.

#finding the frauds
#although we dont have a direct inverse mapping funciton, we can use a dictionary for reverse mapping
mappings = som.win_map(x)
#keys in the dictionary 'mappings' corresponds to coordinates of winning node
#size corresponds to the no of customers (list) which are there in that key
#we will use the coordinated of potential fraws customers from thr graph and get the list of customers.
#for that we will take use of the map and we'll use those coordinates
frauds = np.concatenate((mappings[(5,2)], mappings[(5,7)], mappings[(5,5)]), axis=0)
#concatenate is a numpy function and it takes the arguments as ((lists to be joint), way of joining) axis=0 means vertical join
#inverse scaling:
#we used the sc object for scaling hence well use it for inverse scaling too
frauds = sc.inverse_transform(frauds)
