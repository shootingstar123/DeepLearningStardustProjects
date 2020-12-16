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


# In[2]:


import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--epoch_size',type=int,default=10)
parser.add_argument('--batch_size',type=int,default=360)
args = parser.parse_args()
epoch_size = args.epoch_size
batch_size = args.batch_size
print("epoch=%s,batch_size=%s"%(epoch_size,batch_size))


# In[3]:


months=13
DIR_ROOT = "/Users/stardust/StardustJavaProjects/JavaTest";


# In[4]:


data_bed = []
for i in range(1,months+1):
    data_ = pd.read_csv(DIR_ROOT + "/{}bed_level.txt.csv".format(i),header=None)
    data_.columns = ['x','y','bed_level']
    data_bed.append(data_)


# In[5]:


len(data_bed)


# In[6]:


data_bed[12]


# In[7]:


data_ = []
data_.append(data_bed[4].iloc[30100:30103])
data_.append(data_bed[4].iloc[30103:30106])
data_


# In[8]:


data_bed.count


# In[9]:


data_sedmiment = []
for i in range(1,months+1):
    data_ = pd.read_csv(DIR_ROOT + "/{}sediment_concentration.xyz.csv".format(i),header=None)
    data_.columns = ['x','y','sand_z','concentration']
    data_sedmiment.append(data_)
data_sedmiment[4].iloc[30100:30200,:]


# In[10]:


data_sedmiment[4].count


# In[11]:


data_velocity = []
for i in range(1,months+1):
    data_ = pd.read_csv(DIR_ROOT + "/{}velocity.xyz.csv".format(i),header=None)
    data_.columns = ['x','y','x_velocity','y_velocity','xy_velocity']
    data_velocity.append(data_)
data_velocity[4].iloc[30100:30200,:]


# In[12]:


data_water_depth = []
for i in range(1,months+1):
    data_ = pd.read_csv(DIR_ROOT + "/{}water_depth.xyz.csv".format(i),header=None)
    data_.columns = ['x','y','water_depth']
    data_water_depth.append(data_)
data_water_depth[4].iloc[30100:30200,:]


# In[13]:


plt.scatter(data_bed[4].iloc[:,2],data_sedmiment[4].iloc[:,3])


# In[14]:


plt.scatter(data_bed[4].iloc[:,2],data_velocity[4].iloc[:,4])


# In[15]:


data_sedmiment[4].iloc[0:3,3]


# In[16]:


data_bed[4].iloc[0:3,2]


# In[17]:


data_all = []
for i in range(0,months):
    print(i)
    data_ = pd.concat([data_bed[i],data_sedmiment[i].iloc[:,2:4],data_velocity[i].iloc[:,2:5],data_water_depth[i].iloc[:,2]],axis=1)
    data_all.append(data_)


# In[18]:


data_all[:100]


# In[ ]:





# In[19]:


len(data_all)


# In[20]:


step_len = 3
predict_len = 1
data_len = 36000


# In[21]:


data_all[1].iloc[8640]


# In[22]:


data_step = []
for i in range(0,months-step_len-predict_len+1):
    print(i)
    for index, row in data_all[i].iterrows():
        data_ = []
        for j in range(0,step_len+predict_len):
            data_.append(data_all[i+j].iloc[index])
        data_step.append(data_)


# In[ ]:


data_step[8640:8643]


# In[ ]:


data_step = np.array(data_step)


# In[ ]:


data_step.shape


# In[ ]:


x = data_step[:, :-predict_len, :]
y = data_step[:, -predict_len, 2]


# In[ ]:


x.shape,y.shape


# In[ ]:


x[0:3],y[0:3],months,len(data_all)


# In[ ]:


split=324000


# In[ ]:


train_x = x[:split]
train_y = y[:split]
test_x = x[split:]
test_y = y[split:]


# In[ ]:


train_x.shape,train_y.shape,test_x.shape,test_y.shape


# In[ ]:


train_x[8640:8643],train_y[8640:8643]


# In[ ]:


mean = train_x.mean(axis=0)
std = train_x.std(axis=0)
train_max = train_x.max(axis=0)
train_min = train_x.min(axis=0)
mean.shape


# In[ ]:


train_x = (train_x - mean)/(train_max - train_min)
test_x = (test_x - mean)/(train_max - train_min)
train_x[:1],train_y[:1]


# In[ ]:


model = tf.keras.Sequential()
model.add(layers.Flatten(input_shape=train_x.shape[1:]))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1))
model.summary()


# In[ ]:


model.compile(optimizer='adam',
              loss='mae'
)


# In[ ]:


time_begin = time.time()
history = model.fit(train_x, train_y, batch_size = batch_size, epochs=epoch_size, verbose=1)
time_end = time.time()
print("fit cost time: %.2f s"%(time_end - time_begin))


# In[ ]:


history.history.get('loss')[-5:]


# In[ ]:


plt.plot(history.epoch, history.history.get('loss'))


# In[ ]:


test_x


# In[ ]:


result = model.predict(test_x, batch_size=36000)
result = pd.DataFrame(result)
result_y = pd.DataFrame(test_y)
result = pd.concat([result,result_y],axis=1)
result


# In[ ]:


test_y


# In[ ]:


evaluate_result = model.evaluate(test_x,test_y)
print(evaluate_result)


# In[ ]:





# In[ ]:


result_save = pd.DataFrame(result)
result_save.to_csv(DIR_ROOT + "/SandWaterBpPlusNet.csv")
print(result_save[8630:8670])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




