#!/usr/bin/env python
# coding: utf-8

# In[19]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import time
# %matplotlib inline


# In[20]:


# import argparse
# parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument('--epoch_size',type=int,default=10)
# parser.add_argument('--batch_size',type=int,default=360)
# args = parser.parse_args()
# epoch_size = args.epoch_size
# batch_size = args.batch_size
# print("epoch=%s,batch_size=%s"%(epoch_size,batch_size))


# In[63]:


epoch_size = 100
batch_size = 360
print("epoch=%s,batch_size=%s"%(epoch_size,batch_size))


# In[22]:


months=13
DIR_ROOT = "/Users/stardust/StardustJavaProjects/JavaTest";


# In[23]:


data_bed = []
for i in range(1,months+1):
    data_ = pd.read_csv(DIR_ROOT + "/{}bed_level.txt.csv".format(i),header=None)
    data_.columns = ['x','y','bed_level']
    data_bed.append(data_)


# In[24]:


len(data_bed)


# In[25]:


data_bed[12]


# In[26]:


data_ = []
data_.append(data_bed[4].iloc[30100:30103])
data_.append(data_bed[4].iloc[30103:30106])
data_


# In[27]:


data_bed.count


# In[28]:


data_sedmiment = []
for i in range(1,months+1):
    data_ = pd.read_csv(DIR_ROOT + "/{}sediment_concentration.xyz.csv".format(i),header=None)
    data_.columns = ['x','y','sand_z','concentration']
    data_sedmiment.append(data_)
data_sedmiment[4].iloc[30100:30200,:]


# In[29]:


data_sedmiment[4].count


# In[30]:


data_velocity = []
for i in range(1,months+1):
    data_ = pd.read_csv(DIR_ROOT + "/{}velocity.xyz.csv".format(i),header=None)
    data_.columns = ['x','y','x_velocity','y_velocity','xy_velocity']
    data_velocity.append(data_)
data_velocity[4].iloc[30100:30200,:]


# In[31]:


data_water_depth = []
for i in range(1,months+1):
    data_ = pd.read_csv(DIR_ROOT + "/{}water_depth.xyz.csv".format(i),header=None)
    data_.columns = ['x','y','water_depth']
    data_water_depth.append(data_)
data_water_depth[4].iloc[30100:30200,:]


# In[32]:


plt.scatter(data_bed[4].iloc[:,2],data_sedmiment[4].iloc[:,3])


# In[33]:


plt.scatter(data_bed[4].iloc[:,2],data_velocity[4].iloc[:,4])


# In[34]:


data_sedmiment[4].iloc[0:3,3]


# In[35]:


data_bed[4].iloc[0:3,2]


# In[36]:


data_all = []
for i in range(0,months):
    print(i)
    data_ = pd.concat([data_bed[i],data_sedmiment[i].iloc[:,2:4],data_velocity[i].iloc[:,2:5],data_water_depth[i].iloc[:,2]],axis=1)
    data_all.append(data_)


# In[37]:


data_all[:100]


# In[ ]:





# In[38]:


len(data_all)


# In[84]:


step_len = 11
predict_len = 1
data_len = 36000


# In[85]:


data_all[1].iloc[8640]


# In[86]:


data_step = []
for i in range(0,months-step_len-predict_len+1):
    print(i)
    for index, row in data_all[i].iterrows():
        data_ = []
        for j in range(0,step_len+predict_len):
            data_.append(data_all[i+j].iloc[index])
        data_step.append(data_)


# In[87]:


data_step[8640:8643]


# In[88]:


data_step = np.array(data_step)


# In[89]:


data_step.shape


# In[90]:


x = data_step[:, :-predict_len, :]
y = data_step[:, -predict_len, 2]


# In[91]:


x.shape,y.shape


# In[92]:


x[0:3],y[0:3],months,len(data_all)


# In[93]:


split=36000 * (12 - step_len)


# In[94]:


train_x = x[:split]
train_y = y[:split]
test_x = x[split:]
test_y = y[split:]


# In[95]:


train_x.shape,train_y.shape,test_x.shape,test_y.shape


# In[96]:


train_x[8640:8643],train_y[8640:8643]


# In[97]:


mean = train_x.mean(axis=0)
std = train_x.std(axis=0)
train_max = train_x.max(axis=0)
train_min = train_x.min(axis=0)
mean.shape


# In[98]:


train_x = (train_x - mean)/(train_max - train_min)
test_x = (test_x - mean)/(train_max - train_min)
train_x[:1],train_y[:1]


# In[99]:


model = tf.keras.Sequential()
model.add(layers.LSTM(32,return_sequences=True, input_shape=(train_x.shape[1:]), ))
model.add(layers.LSTM(16,return_sequences=True))
model.add(layers.LSTM(8))
model.add(layers.Dense(1))
model.summary()


# In[100]:


model.compile(optimizer='adam',
              loss='mae'
)


# In[101]:


time_begin = time.time()
history = model.fit(train_x, train_y, batch_size = batch_size, epochs=epoch_size, verbose=1)
time_end = time.time()
print("fit cost time: %.2f s"%(time_end - time_begin))


# In[57]:


history.history.get('loss')[-5:]


# In[58]:


plt.plot(history.epoch, history.history.get('loss'))


# In[59]:


test_x


# In[60]:


result = model.predict(test_x, batch_size=36000)
result = pd.DataFrame(result)
result_y = pd.DataFrame(test_y)
result = pd.concat([result,result_y],axis=1)
result


# In[61]:


test_y


# In[62]:


evaluate_result = model.evaluate(test_x,test_y)
print(evaluate_result)


# In[ ]:





# In[ ]:


result_save = pd.DataFrame(result)
result_save.to_csv(DIR_ROOT + "/SandWaterLSTMTest.csv")
print(result_save[8630:8670])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




