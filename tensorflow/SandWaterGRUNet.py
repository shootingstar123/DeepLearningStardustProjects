#!/usr/bin/env python
# coding: utf-8

# In[41]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import time
# %matplotlib inline


# In[43]:


# import argparse
# parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument('--epoch_size',type=int,default=10)
# parser.add_argument('--batch_size',type=int,default=360)
# args = parser.parse_args()
# epoch_size = args.epoch_size
# batch_size = args.batch_size
# print("epoch=%s,batch_size=%s"%(epoch_size,batch_size))


# In[44]:


months=13
DIR_ROOT = "/Users/stardust/StardustJavaProjects/JavaTest";


# In[45]:


data_bed = []
for i in range(1,months+1):
    data_ = pd.read_csv(DIR_ROOT + "/{}bed_level.txt.csv".format(i),header=None)
    data_.columns = ['x','y','bed_level']
    data_bed.append(data_)


# In[46]:


len(data_bed)


# In[47]:


data_bed[12]


# In[48]:


data_ = []
data_.append(data_bed[4].iloc[30100:30103])
data_.append(data_bed[4].iloc[30103:30106])
data_


# In[49]:


data_bed.count


# In[50]:


data_sedmiment = []
for i in range(1,months+1):
    data_ = pd.read_csv(DIR_ROOT + "/{}sediment_concentration.xyz.csv".format(i),header=None)
    data_.columns = ['x','y','sand_z','concentration']
    data_sedmiment.append(data_)
data_sedmiment[4].iloc[30100:30200,:]


# In[51]:


data_sedmiment[4].count


# In[52]:


data_velocity = []
for i in range(1,months+1):
    data_ = pd.read_csv(DIR_ROOT + "/{}velocity.xyz.csv".format(i),header=None)
    data_.columns = ['x','y','x_velocity','y_velocity','xy_velocity']
    data_velocity.append(data_)
data_velocity[4].iloc[30100:30200,:]


# In[53]:


data_water_depth = []
for i in range(1,months+1):
    data_ = pd.read_csv(DIR_ROOT + "/{}water_depth.xyz.csv".format(i),header=None)
    data_.columns = ['x','y','water_depth']
    data_water_depth.append(data_)
data_water_depth[4].iloc[30100:30200,:]


# In[54]:


plt.scatter(data_bed[4].iloc[:,2],data_sedmiment[4].iloc[:,3])


# In[55]:


plt.scatter(data_bed[4].iloc[:,2],data_velocity[4].iloc[:,4])


# In[56]:


data_sedmiment[4].iloc[0:3,3]


# In[57]:


data_bed[4].iloc[0:3,2]


# In[58]:


data_all = []
for i in range(0,months):
    print(i)
    data_ = pd.concat([data_bed[i],data_sedmiment[i].iloc[:,2:4],data_velocity[i].iloc[:,2:5],data_water_depth[i].iloc[:,2]],axis=1)
    data_all.append(data_)


# In[59]:


data_all[:100]


# In[ ]:





# In[60]:


len(data_all)


# In[61]:


step_len = 3
predict_len = 1
data_len = 36000


# In[62]:


data_all[1].iloc[8640]


# In[63]:


data_step = []
for i in range(0,months-step_len-predict_len+1):
    print(i)
    for index, row in data_all[i].iterrows():
        data_ = []
        for j in range(0,step_len+predict_len):
            data_.append(data_all[i+j].iloc[index])
        data_step.append(data_)


# In[64]:


data_step[8640:8643]


# In[65]:


data_step = np.array(data_step)


# In[66]:


data_step.shape


# In[67]:


x = data_step[:, :-predict_len, :]
y = data_step[:, -predict_len, 2]


# In[68]:


x.shape,y.shape


# In[69]:


x[0:3],y[0:3],months,len(data_all)


# In[70]:


split=36000 * (12 - step_len)


# In[71]:


train_x = x[:split]
train_y = y[:split]
test_x = x[split:]
test_y = y[split:]


# In[72]:


train_x.shape,train_y.shape,test_x.shape,test_y.shape


# In[73]:


train_x[8640:8643],train_y[8640:8643]


# In[74]:


mean = train_x.mean(axis=0)
std = train_x.std(axis=0)
train_max = train_x.max(axis=0)
train_min = train_x.min(axis=0)
mean.shape


# In[75]:


train_x = (train_x - mean)/(train_max - train_min)
test_x = (test_x - mean)/(train_max - train_min)
train_x[:1],train_y[:1]


# In[83]:


model = tf.keras.Sequential()
model.add(layers.GRU(32,return_sequences=True, input_shape=(train_x.shape[1:]), ))
model.add(layers.GRU(16,return_sequences=True))
model.add(layers.GRU(8))
model.add(layers.Dense(1))
model.summary()


# In[84]:


model.compile(optimizer='adam',
              loss='mae'
)


# In[85]:


epoch_size = 10
batch_size = 360
print("epoch=%s,batch_size=%s"%(epoch_size,batch_size))


# In[86]:


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
result_save.to_csv(DIR_ROOT + "/SandWaterLSTMTest.csv")
print(result_save[8630:8670])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




