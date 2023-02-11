#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


# In[48]:


df=np.loadtxt("Faraday_Arrival/39.8_80_trial2/trail_2.csv",delimiter=',',unpack=True)
#Read datas from acc first accelerpometeri
time1=df[0];
ax1=df[1];
ay1=df[2]; 
az1=df[3]; 

"""
#Data from second accelerometer
time2=df[7]
ax2=df[4]/2048.;
ay2=df[5]/2048.; 
az2=df[6]/2048.; 
"""

# In[49]:


#Interpolation
time_fft=time1*10**(-6)
acc1=interp1d(time_fft, az1, kind='cubic')
time_1=np.linspace(time_fft[0],time_fft[20],100000)
plt.plot(time_1,acc1(time_1))
#Plot before interpolation
plt.plot(time_fft[0:20],az1[0:20],'o')
plt.show()


# In[50]:


#FFT
df=time_fft[1]-time_fft[0]
time2_=np.arange(time_fft[0],time_fft[len(time_fft)-1],df/1000)
y=acc1(time2_)
L1=len(y)
y=y/L1
dt=time2_[1]-time2_[0]
a1=np.fft.fft(y)
freq=np.fft.fftfreq(time2_.shape[-1],dt)

plt.plot(freq,abs(a1))
plt.xlim(0,125)
plt.show()


# In[ ]:





# In[59]:


index, _ = find_peaks(abs(a1), height=0.1)
print(freq[index[0]],freq[index[1]])
# Get magnitude and phase
magnitude1 = np.abs(a1[index[0]])
phase = [np.angle(a1[index[0]]),np.angle(a1[index[1]])]
magnitude2 = np.abs(a1[index[1]])
for i in range(len(phase)):
    
    if phase[i]<0:
        phase[i]=phase[i]+np.pi
    
#freq1=freq[index[0]]
#freq2=freq[index[1]]
freq1=40.
freq2=80.
#print(max(y),np.mean(y),min(y))

print("Magnitude:", magnitude1,magnitude2 ,", phase:", phase)


# In[40]:

"""
a=magnitude1
b=magnitude2
phase1=phase[0]
phase2=phase[1]
"""


# In[41]:

n=20000
def func(t,a,b,phase1,phase2):
    y1 = a*np.sin(2*np.pi*freq1*t+phase1)+b*np.sin(2*np.pi*freq2*t+phase2)+np.mean(acc1(time2_))
    return y1
params_bound=[[0,0,0,0],[4.0,4.0,2*np.pi,2*np.pi]]
a,b,phase1,phase2 = curve_fit(func,time2_[0:n],acc1(time2_)[0:n],bounds=params_bound)[0]
print(a,b,phase1,phase2)
#y2 = np.mean(y) + magnitude1 * np.sin(time1 * int(freq[index[0]]) * 2 * np.pi + phase1) + magnitude2 * np.sin(time1 * int(freq[index[1]]) * 2*np.pi + phase2)
az1_fit=func(time2_,a,b,phase1,phase2)

plt.plot(time2_[0:n],acc1(time2_)[0:n],'k')
plt.plot(time2_[0:n],az1_fit[0:n],'b.')
#plt.xlim(5,5.05)
plt.show()

"""
# In[ ]:

# 2nd accelerometer


#Data from second accelerometer
time2=df[7]
ax2=df[4];
ay2=df[5]; 
az2=df[6]; 

# In[49]:


#Interpolation
time_fft2=time2*10**(-3)
acc1=interp1d(time_fft2, az2, kind='cubic')
time_2=np.linspace(time_fft1[0],time_fft2[20],100000)
plt.plot(time_2,acc1(time_2))
#Plot before interpolation
plt.plot(time_fft[0:20],az1[0:20],'o')
plt.show()


# In[50]:


#FFT
df=time_fft[1]-time_fft[0]
time2_=np.arange(time_fft[0],time_fft[len(time_fft)-1],df/1000)
y=acc1(time2_)
L1=len(y)
y=y/L1
dt=time2_[1]-time2_[0]
a1=np.fft.fft(y)
freq=np.fft.fftfreq(time2_.shape[-1],dt)

plt.plot(freq,abs(a1))
plt.xlim(0,100)
plt.show()


# In[ ]:





# In[59]:


index, _ = find_peaks(abs(a1), height=0.1)
# Get magnitude and phase
magnitude1 = np.abs(a1[index[0]])
phase = [np.angle(a1[index[0]]),np.angle(a1[index[1]])]
magnitude2 = np.abs(a1[index[1]])
for i in range(len(phase)):
    
    if phase[i]<0:
        phase[i]=phase[i]+np.pi
    
freq1=40.
freq2=80.

#print(max(y),np.mean(y),min(y))

print("Magnitude:", magnitude1,magnitude2 ,", phase:", phase)


# In[41]:

n=50000
def func(t,a,b,phase1,phase2):
    y1 = a*np.sin(2*np.pi*freq1*t+phase1)+b*np.sin(2*np.pi*freq2*t+phase2)+np.mean(acc1(time2_))
    return y1
params_bound=[[0,0,0,0],[4.0,4.0,2*np.pi,2*np.pi]]
a,b,phase1,phase2 = curve_fit(func,time2_[0:n],acc1(time2_)[0:n],bounds=params_bound)[0]
print(a,b,phase1,phase2)
#y2 = np.mean(y) + magnitude1 * np.sin(time1 * int(freq[index[0]]) * 2 * np.pi + phase1) + magnitude2 * np.sin(time1 * int(freq[index[1]]) * 2*np.pi + phase2)
az1_fit=func(time2_,a,b,phase1,phase2)

plt.plot(time2_[0:n],acc1(time2_)[0:n],'k')
plt.plot(time2_[0:n],az1_fit[0:n],'b.')
#plt.xlim(5,5.05)
plt.show()
"""
