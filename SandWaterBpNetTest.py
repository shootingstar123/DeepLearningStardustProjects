#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import time
# %matplotlib inline


# In[ ]:


import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--epoch_size',type=int,default=10)
parser.add_argument('--batch_size',type=int,default=360)
args = parser.parse_args()
epoch_size = args.epoch_size
batch_size = args.batch_size
print("epoch=%s,batch_size=%s"%(epoch_size,batch_size))


# In[2]:


months=13
DIR_ROOT = "/Users/stardust/StardustJavaProjects/JavaTest";


# In[3]:


data_bed = []
for i in range(1,months+1):
    data_ = pd.read_csv(DIR_ROOT + "/{}bed_level.txt.csv".format(i),header=None)
    data_.columns = ['x','y','bed_level']
    data_bed.append(data_)


# In[4]:


len(data_bed)


# In[5]:


data_bed[12]


# In[6]:


data_ = []
data_.append(data_bed[4].iloc[30100:30103])
data_.append(data_bed[4].iloc[30103:30106])
data_


# In[7]:


data_bed.count


# In[8]:


data_sedmiment = []
for i in range(1,months):
    data_ = pd.read_csv(DIR_ROOT + "/{}sediment_concentration.xyz.csv".format(i),header=None)
    data_.columns = ['x','y','sand_z','concentration']
    data_sedmiment.append(data_)
data_sedmiment[4].iloc[30100:30200,:]


# In[9]:


data_sedmiment[4].count


# In[10]:


data_velocity = []
for i in range(1,months):
    data_ = pd.read_csv(DIR_ROOT + "/{}velocity.xyz.csv".format(i),header=None)
    data_.columns = ['x','y','x_velocity','y_velocity','xy_velocity']
    data_velocity.append(data_)
data_velocity[4].iloc[30100:30200,:]


# In[11]:


data_water_depth = []
for i in range(1,months):
    data_ = pd.read_csv(DIR_ROOT + "/{}water_depth.xyz.csv".format(i),header=None)
    data_.columns = ['x','y','water_depth']
    data_water_depth.append(data_)
data_water_depth[4].iloc[30100:30200,:]


# In[12]:


plt.scatter(data_bed[4].iloc[:,2],data_sedmiment[4].iloc[:,3])


# In[13]:


plt.scatter(data_bed[4].iloc[:,2],data_velocity[4].iloc[:,4])


# In[14]:


data_sedmiment[4].iloc[0:3,3]


# In[15]:


data_bed[4].iloc[0:3,2]


# In[16]:


from itertools import chain
data_all = pd.DataFrame()
for i in range(0,months-1):
    print(i)
    data_ = pd.concat([data_bed[i],data_sedmiment[i].iloc[:,2:4],data_velocity[i].iloc[:,2:5],data_water_depth[i].iloc[:,2],data_bed[i+1].iloc[:,2]],axis=1)
    data_all = pd.concat([data_all,data_],axis=0)
data_all.head()
data_all.columns = ['x','y','bed_level','sand_z','concentration','x_velocity','y_velocity','xy_velocity','water_depth','bed_level_next']


# In[17]:


len(data_all)


# In[18]:


def function(a, b):
    if a == b:
        return 1
    else:
        return 0


# In[19]:


data_all['bool'] = data_all.apply(lambda x : function(x['bed_level'],x['bed_level_next']),axis = 1)


# In[20]:


data_all


# In[21]:


data_all[data_all['bool']==0]


# In[ ]:





# In[22]:


data_all.iloc[30100:30200,:]


# In[23]:


model = tf.keras.Sequential()
model.add(layers.Dense(32, input_shape=(9,), activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.summary()
model.add(layers.Dense(1))


# In[24]:


model.compile(optimizer='adam',
              loss='mse'
)


# In[25]:


split=396000


# In[26]:


train_x = data_all.iloc[:split,0: -2]
train_y = data_all.iloc[:split,-2]


# In[27]:


train_x


# In[28]:


train_y


# In[29]:


mean = train_x.mean(axis=0)
std = train_x.std(axis=0)
train_max = train_x.max(axis=0)
train_min = train_x.min(axis=0)
mean.shape


# In[30]:


test_x = data_all.iloc[split:, 0: -2]
test_y = data_all.iloc[split:,-2]


# In[31]:


train_x = (train_x - mean)/(train_max - train_min)
test_x = (test_x - mean)/(train_max - train_min)
train_x.head()


# In[32]:


time_begin = time.time()
history = model.fit(train_x, train_y, batch_size = batch_size, epochs=epoch_size, verbose=1)
time_end = time.time()
print("fit cost time: %.2f s"%(time_end - time_begin))


# In[33]:


history.history.get('loss')[-5:]


# In[34]:


plt.plot(history.epoch, history.history.get('loss'))


# In[35]:


test_x


# In[36]:


result = model.predict(test_x, batch_size=36000)
result = pd.DataFrame(result)
result_y = pd.DataFrame(test_y)
result = pd.concat([result,result_y],axis=1)
result


# In[37]:


test_y


# In[38]:


evaluate_result = model.evaluate(test_x,test_y)
print(evaluate_result)


# In[ ]:





# In[39]:


result_save = pd.DataFrame(result)
result_save.to_csv("~/SandWaterBpNetTest.csv")
print(result_save[8630:8670])


# In[ ]:




