#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries we need

# In[1]:


import numpy as np
import pickle
from sklearn.utils import shuffle
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import BatchNormalization , Conv2D , LeakyReLU,Conv2DTranspose
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


# ## Handling input and output Data for Training

# In[ ]:


# Load training images
train_images = pickle.load(open("inputs.p", "rb" ))

# Load image labels
labels = pickle.load(open("outputs.p", "rb" ))

# Make into arrays as the neural network wants these
train_images = np.array(train_images)
labels = np.array(labels)

# Normalize labels - training images get normalized to start in the network
labels = labels / 255

# Shuffle images along with their labels, then split into training/validation sets
train_images, labels = shuffle(train_images, labels)
# Validation set size 10% and remainder for training
X_train, X_val, y_train, y_val 
    = train_test_split(train_images,labels, test_size=0.1)    
#setting some parameters
batch_size = 200
epochs = 20


# ## Building our model

# In[43]:


model = Sequential()
model.add(BatchNormalization(input_shape=X_train.shape[1:]))
#encoder part
model.add(Conv2D(64,(4,4),padding='same',strides=(1,1),name='conv1'))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(64,(4,4),padding='same',strides=(2,2),name='conv2'))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(128,(4,4),padding='same',strides=(2,2),name='conv3'))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(256,(4,4),padding='same',strides=(2,2),name='conv4'))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(512,(4,4),padding='same',strides=(2,2),name='conv5'))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(512,(4,4),padding='same',strides=(2,2),name='conv6'))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(512,(4,4),padding='same',strides=(2,2),name='conv7'))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(512,(4,4),padding='same',strides=(2,2),name='conv8'))
model.add(LeakyReLU(alpha=0.2))
#decoder part
model.add(Conv2DTranspose(512,(4,4),padding='same',strides=(2,2),activation='relu',name='deconv1'))
model.add(Conv2DTranspose(512,(4,4),padding='same',strides=(2,2),activation='relu',name='deconv2'))
model.add(Conv2DTranspose(512,(4,4),padding='same',strides=(2,2),activation='relu',name='deconv3'))
model.add(Conv2DTranspose(256,(4,4),padding='same',strides=(2,2),activation='relu',name='deconv4'))
model.add(Conv2DTranspose(128,(4,4),padding='same',strides=(2,2),activation='relu',name='deconv5'))
model.add(Conv2DTranspose(64,(4,4),padding='same',strides=(2,2),activation='relu',name='deconv6'))
model.add(Conv2DTranspose(64,(4,4),padding='same',strides=(2,2),activation='relu',name='deconv7'))
#last layer
model.add(Conv2D(3,(4,4),padding='same',strides=(1,1),activation='tanh',name='last_layer'))
model.summary()


# ## Compiling model and train it

# In[ ]:


# compile our model
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
# training our model
model.fit_generator(ImageDataGenerator().flow(X_train, y_train, batch_size=batch_size),steps_per_epoch = len(X_train)/batch_size,validation_data = ImageDataGenerator().flow(X_val, y_val, batch_size=50) 
                    ,validation_steps = len(X_val)/50 ,epochs = 20,verbose=2)
#save our model
model.save('movieColor.h5')

