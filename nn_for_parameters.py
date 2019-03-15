#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


# In[2]:


def func2learn(x,y, p1, p2):
    return p1*np.exp(x) + y*x*p2

def create_traning_data(Ntest, npts=3, seed = 0, add_3rd_dim = False):
    X = np.zeros((Ntest,npts, npts, 1)) if add_3rd_dim else  np.zeros((Ntest,npts, npts))
    # random numbers in (0,1) to represnt p1, p2
    np.random.seed(seed)
    Y = np.random.rand(Ntest, 2)
    
    x = np.linspace(start=0.0, stop = 1.0, num=npts)
    y = x
    
    for n in np.arange(Ntest):
        for i in np.arange(npts):
            for j in np.arange(npts):
                if not add_3rd_dim:
                    X[n,i,j] = func2learn(x[i], y[j], Y[n,0], Y[n,1])
                else:
                    X[n,i,j, 0] = func2learn(x[i], y[j], Y[n,0], Y[n,1])
        
    return X,Y


# In[3]:


X, Y = create_traning_data(5,3)
X2, Y2 = create_traning_data(5,3, add_3rd_dim=True)
print(X[0], X2[0])
#print(Y)


# In[4]:


def create_model1(npts, nhnodes = 10):
    X_input = Input((npts, npts))
    X = Flatten()(X_input)
    #X = Dense(nhnodes, activation='tanh', name='Hidden', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Dense(2, name='Final_1', kernel_initializer = glorot_uniform(seed=0))(X)
    # Create model
    model = Model(inputs = X_input, outputs = X, name='Model1')
    return model


# In[5]:


model = create_model1(3)
model.summary()
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


# In[6]:


X_train, Y_train = create_traning_data(10000,3)
model.fit(X_train, Y_train, epochs = 100, batch_size = None)


# In[7]:


pred = model.predict(X)
pred[0:10,:], Y[0:10,:]


# In[8]:


model.get_weights()


# In[10]:


def create_model2(npts, nhnodes = 6):
    X_input = Input((npts, npts, 1))
    X = Conv2D(filters = 20, kernel_size = (3, 3), strides = (1,1), padding = 'valid', name = 'conv1_22_1', kernel_initializer = glorot_uniform(seed=0))(X_input)
    #X = Conv2D(filters = 6, kernel_size = (2, 2), strides = (1,1), padding = 'valid', name = 'conv1_22_2', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Flatten()(X)
    #X = Dense(nhnodes, activation='tanh', name='Hidden', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Dense(2, activation='sigmoid', name='Final_1', kernel_initializer = glorot_uniform(seed=0))(X)
    # Create model
    model = Model(inputs = X_input, outputs = X, name='Model2')
    return model


# In[11]:


model2 = create_model2(3)
model2.summary()
model2.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


# In[13]:


X_train2, Y_train2 = create_traning_data(10000,3, add_3rd_dim=True)
model2.fit(X_train2, Y_train2, epochs = 100, batch_size = None)


# In[15]:


pred = model2.predict(X_train2)
pred[0:10,:], Y_train2[0:10,:]

