from keras.models import Sequential#initializong the network
from keras.layers import Convolution2D#adding the convolutional layers,and bacuse images are in 2d
from keras.layers import MaxPooling2D#for Pooling 
from keras.layers import Flatten#for flattening 
from keras.layers import Dense#for adding the fully connected layers

#Initializing the CNN
classifier=Sequential()#Initilaizing the network as a sequence of layers

classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation="relu"))
#Addiding a Convulutional layer with 32 features detectors of size 3*3
#input_shape is the shape of your image which you are expecting
#Relu is used just so that we dont have negative feature values in our NN and for having non-linearity 

classifier.add(MaxPooling2D(pool_size=(2,2)))
#2*2 is the stride which the feature detector will pass to take out the main features 

#Adding one more layer of  Convolutional layer for better accuracy
classifier.add(Convolution2D(32,3,3,activation="relu"))
#input shape parameter is not required because it will take the pooled features maps from
#the 2 previous steps 
classifier.add(MaxPooling2D(pool_size=(2,2)))



classifier.add(Flatten())

#Making the neural network
classifier.add(Dense(output_dim=128,activation='relu')) #A number around 
#100 is a good choice for output_dimensions but not less than that and should be of the power of 2 
classifier.add(Dense(output_dim=1,activation='sigmoid')) 



#Compile
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#Fitting the model to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
#Rescaling images of the test set so that they have the values between 0 and 1 


#Creating the training set 
training_set = train_datagen.flow_from_directory('training_set',
                                                     target_size=(64, 64),#dimensions expected by the model
                                                     batch_size=32,
                                                     class_mode='binary')#dependent variable is binary 


#Creating the test set 
test_set= test_datagen.flow_from_directory( 'test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

#Fitting our CNN model to the trainig set as well as evaluting its
#performance on the test set 

classifier.fit_generator(training_set,
                          steps_per_epoch=8000,
                          epochs=50,
                          validation_data=test_set,
                          validation_steps=2000) 