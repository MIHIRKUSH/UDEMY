import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

dataset=pd.read_csv("Credit_Card_Applications.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

#Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
Mc=MinMaxScaler()
X=Mc.fit_transform(X)

from minisom import MiniSom#A Library used for traning the SOM
som=MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)
som.random_weights_init(X)#Randomly Inizializing the weights 
som.train_random(data=X,num_iteration=100)#Training the Som with 100 iterations 

 #Visualize the results
from pylab import bone,pcolor,colorbar,plot,show
bone()# white window 
pcolor(som.distance_map().T )#will return all the mean inter nueron distances for the winning nodes 
colorbar()#the colorbar on the right 
markers=['o','s']#the circle shows normal people and the s is for square which shows fraud
colors=['r','g']#red and green color 
for i, x in enumerate(X):#i is the index of all the customes and x is the detail of each customer who will be assigned a red or a green color based on the MID
   w=som.winner(x)#will fetch the winning node for the customer x 
   plot(w[0]+0.5,
        w[1]+0.5,markers[Y[i]],markeredgecolor=colors[Y[i]],markerfacecolor='None',markersize=10,
        markeredgewidth=2)#in this for loop we plot the winner in the window
   #which comes from bone plot will fetch the placement of the marker in the middle of the square
   #markeredgecolor will give the color to the edges of the circle of square which is given by y and
   #y[i] will loop through the winners who were granted so if y[i]=0 then a circle and 1 then square 
   show()
   
   
#Finding the frauds
mappings=som.win_map(X)#a dict which will map the winning nodes to the customers   
frauds=mappings[(7,3)]#the co ordinates of the outliers 
frauds=Mc.inverse_transform(frauds)#inverse mapping to the list of customerID who have cheated 