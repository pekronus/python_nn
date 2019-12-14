#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


def create_model(dims):
    model = Sequential()
    i = 0
    for d in dims:
        print(d)
        model.add(Dense(d, input_dim=1, activation='relu')) if i == 0 else model.add(Dense(d, input_dim=1, activation='relu'))
        i+=1
    model.add(Dense(1, activation='linear'))
    return model


# In[19]:


def ifunc(x):
    return 0.25*(np.sin(2*np.pi*x*x)+2.0)

np.random.seed(0)
X = np.random.sample([2048])
Y = inputfunct(X) + 0.2*np.random.normal(0,0.2,len(X))

Xreal = np.arange(0.0, 1.0, 0.01)
Yreal = ifunc(Xreal)


# In[56]:


model1 = create_model([8, 64])
model1.compile(optimizer='adam', loss='mse', metrics=['mse'])
model1.summary()
#training
nepoch = 128
nbatch = 16
model1.fit(X, Y, epochs=nepoch, batch_size=nbatch)
Ylearn = model1.predict(Xreal)


# In[57]:



#plot
#plt.plot(X,Y,'.', label='Raw noisy input data')
plt.plot(Xreal,Yreal - Ylearn.flatten(), label='diff', linewidth=4.0, c='black')
#plt.plot(Xreal, Ylearn, label='Output of the Neural Net', linewidth=4.0, c='red')
plt.legend()


# In[45]:


model1.predict([0, 0.5, 1])


# In[53]:


Xreal[-1],Yreal[-1]

