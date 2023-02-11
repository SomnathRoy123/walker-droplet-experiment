#!/usr/bin/env python
# coding: utf-8

# # Tracking particle
# 

# Library

# In[12]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp
#import cv2
import os


# In[13]:


#Read the images
@pims.pipeline
def gray(image):
    return image[:,:, 1]  # Take just the green channel
frames = gray(pims.PyAVVideoReader('0.mp4'))

"""
# In[39]:


f1 = tp.locate(frames[15],27 ,minmass=9500)
tp.annotate(f1, frames[14])

fig, ax = plt.subplots()
ax.hist(f1['mass'], bins=20)

# Optionally, label the axes.
ax.set(xlabel='mass', ylabel='count');


tp.subpx_bias(f1)
"""

# In[41]:


f = tp.batch(frames[50:200],19, minmass=9500, processes=1)


# In[42]:


t = tp.link(f, 15, memory=5)
#tp.quiet()
t1 = tp.filter_stubs(t, 5)
#t.head()
t.head()
plt.figure()
tp.plot_traj(t1)
plt.show()

"""
# In[44]:


plt.figure()
tp.mass_size(t1.groupby('particle').mean()); # convenience function -- just plots size vs. mass


# In[47]:


d = tp.compute_drift(t1)
d.plot()
plt.show()


# In[48]:





# In[59]:


im = tp.imsd(tm, 100/285., 30)  # microns per pixel = 100/285., frames per second = 24


# In[74]:



#im.index

fig, ax = plt.subplots()
ax.plot(im.index ,im.iloc[:,0], 'b-', alpha=0.5)

ax.plot(im.index ,im.iloc[:,1], 'k-', alpha=0.5)


# In[68]:


im.head


# In[ ]:


"""

