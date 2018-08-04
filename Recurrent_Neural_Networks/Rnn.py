import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
training_set=dataset_train.iloc[:,1:2].values#Converting into numpy arrays 

#Feature scaling applying normalization
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
training_set_scaled=sc.fit_transform(training_set) 

#look at 60 previous stock prices to predict the the 1 next 
X_train=[]
Y_train=[]
for i in range(60, 1258):#1257 rows 
    X_train.append(training_set_scaled[i-60:i,0])#Selecting 0 to 59 stock prices for learning ie (60-60 to 60)
    Y_train.append(training_set_scaled[i, 0])#will contain the 60th stock price which will learnt from X_train 
X_train,Y_train=np.array(X_train),np.array(Y_train)#converting onto numpy arrays

#Reshaping 
X_train=np.reshape(X_train,(1198,60,1))
#Creating a indicator which will be the 3rd dimension based on the(observation,total time steps) 

from keras.models import Sequential#Iniliatize 
from keras.layers import Dense#Add the output layer
from keras.layers import LSTM#Add LSTM layer
from keras.layers import Dropout#Add some droput regulariztion to avoid overfitting 

regressor=Sequential()

regressor.add(LSTM(units= 50 ,return_sequences=True,input_shape=(60,1)))
#(number of nuerons,return the current output sequence to the next layer,
#input shape which will be accepted by the layer)
regressor.add(Dropout(0.2))#20% nuerosn will be dropped during eating iteration 

regressor.add(LSTM(units= 50 ,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units= 50 ,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units= 50))#last LSTM layer default parameter is false 
regressor.add(Dropout(0.2))

#Output Layer
regressor.add(Dense(units=1))#number of dimensions in the output layer 

#Compiling the RNN 
regressor.compile(optimizer='adam',loss='mean_squared_error')

#Fitting the nueral network to the dataset 
regressor.fit(X_train,Y_train,epochs=50,batch_size=20)

#Choosing the test data for comparing 
dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
real_price=dataset_test.iloc[:,1:2].values#Choosing the Open coloum of the dataset 


dataset_all=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
#Concatinating both the open coloums of the  datasets along the vertical axis 

inputs=dataset_all[len(dataset_all)-len(dataset_test)-60:].values

inputs=inputs.reshape(-1,+1)

inputs=sc.transform(inputs)


X_test=[]
for i in range(60, 80):#test set contains only 20 finacial days   
    X_test.append(inputs[i-60:i,0])#Selecting 0 to 59 stock prices for learning ie (60-60 to 60)
X_test=np.array(X_test)
X_test=np.reshape(X_test,(20,60,1))
predicted_price=regressor.predict(X_test)
predicted_price=sc.inverse_transform(predicted_price)

plt.plot(real_price,color='red',label='The Original')
plt.plot(predicted_price,color='blue',label='The Predicted')
plt.title('Google Stock price prediction')
plt.xlabel('Time')
plt.ylabel('Stock value')
plt.legend()
plt.show()

