# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error,r2_score

# In[2]:

calcsv= pd.read_csv(r'D:\uni\4th year\1st sem\RESEARCH\details\Data\2-Manel\14,15,16,17 & 18_12_2023.csv')

# In[3]:

Day = calcsv['Day'].to_numpy()
time = calcsv['time'].to_numpy()
ambient_temp = calcsv['ambient'].to_numpy()
surface_temp = calcsv['surface'].to_numpy()
inside_temp = calcsv['inside'].to_numpy()
inside_err = calcsv['err_inside'].to_numpy()

# In[4]:

day_3 = calcsv.drop(index = calcsv[Day!='day 3'].index)
day_4 = calcsv.drop(index = calcsv[Day!='day 4'].index)
day_5 = calcsv.drop(index = calcsv[Day!='day 5'].index)
day_6 = calcsv.drop(index = calcsv[Day!='day 6'].index)
day_7 = calcsv.drop(index = calcsv[Day!='day 7'].index)

# In[5]:

x = np.arange(len(Day))
plt.figure(figsize=[15,7])

plt.plot(x,ambient_temp,color='red')
plt.plot(x,surface_temp,color='green')
plt.plot(x,inside_temp,color='blue')

y=[34,322,610,898,1135]
xl=['day 3 - 12:00PM\n(14/12/2023)','day 4 - 12:00PM\n(15/12/2023)','day 5 - 12:00PM\n(16/12/2023)','day 6 - 12:00PM\n(17/12/2023)','day 7 - 7:45AM\n(18/12/2023)']
plt.xticks(y,xl,rotation = 0,fontsize='15')
plt.yticks(fontsize='15')

x2=[178,466,754,1042]
for j in range (4):
    plt.axvline(x=x2[j], color = 'black', label = 'axvline',linewidth='0.8') 
    
plt.ylim([24,36])

plt.xlabel('Number of days since the flower started blooming',fontsize='20')
plt.ylabel('Temperature in $^\circ$C',fontsize='20')
plt.title('Temperature of ambient,surface and inside of the bud vs time',fontsize='25')
plt.legend(["Ambient temperature", "Surface temperature","Inside temperature"],
           loc = 'upper left',fontsize='15',markerscale=6,
           shadow = True, facecolor = 'yellow')
plt.grid(axis='y')

plt.show()

# In[6]:

g = [day_3,day_4,day_5,day_6,day_7]
for j in range (5):
    index=[]
    i=0
    for i in range (len(g[j]['time'].to_numpy())):
        index.append(i)
    g[j]['index']= index
    g[j]= g[j].set_index('index')

# In[7]:

d = [g[0],g[1],g[2],g[3],g[4]]
c = ['red','orange','violet','green','blue']
xl= []

plt.figure(figsize=[10,5])
for j in range(4):
    x=[]
    for k in range (len(d[1]['inside'].to_numpy())-len(d[j]['inside'].to_numpy()),len(d[1]['inside'].to_numpy())):
        x.append(k)
        
    plt.plot(x,d[j]['inside'].to_numpy(),color=c[j])

x=[]
for k in range (0,len(d[4]['inside'].to_numpy())):
    x.append(k)
plt.plot(x,d[4]['inside'].to_numpy(),color=c[4])  

y=[0,12,24,36,48,60,72,84,96,108,120,132,144,156,168,180,192,204,216,228,240,252,264,276]

for k in range(len(y)):
    xl.append(g[1]['time'].to_numpy()[y[k]])

plt.xticks(y,xl,rotation = 90,fontsize='15')
plt.yticks(fontsize='15')

plt.xticks(fontsize='15')
plt.yticks(fontsize='15')
#plt.xlim([24,31])
plt.xlabel ('Time',fontsize='15')
plt.ylabel ('Inside temperature in $^\circ$C',fontsize='15')
plt.title ('Inside temperature vs time',fontsize='20')
plt.legend(["day 3(14/12/2023)", "day 4(15/12/2023)","day 5(16/12/2023)", "day 6(17/12/2023)","day 7(18/12/2023)"],
           loc = 'upper left',fontsize='8',markerscale=2,
           shadow = True, facecolor = 'cyan')

plt.grid()
plt.show()

# In[8]:

d = [g[0],g[1],g[2],g[3],g[4]]
c = ['red','orange','violet','green','blue']
xl= []

plt.figure(figsize=[10,5])
for j in range(4):
    x=[]
    for k in range (len(d[1]['inside'].to_numpy())-len(d[j]['inside'].to_numpy()),len(d[1]['inside'].to_numpy())):
        x.append(k)
        
    plt.plot(x,d[j]['surface'].to_numpy(),color=c[j])

x=[]
for k in range (0,len(d[4]['inside'].to_numpy())):
    x.append(k)
plt.plot(x,d[4]['surface'].to_numpy(),color=c[4])  

y=[0,12,24,36,48,60,72,84,96,108,120,132,144,156,168,180,192,204,216,228,240,252,264,276]

for k in range(len(y)):
    xl.append(g[1]['time'].to_numpy()[y[k]])

plt.xticks(y,xl,rotation = 90,fontsize='15')
plt.yticks(fontsize='15')

plt.xticks(fontsize='15')
plt.yticks(fontsize='15')
#plt.xlim([24,31])
plt.xlabel ('Time',fontsize='15')
plt.ylabel ('Surface temperature in $^\circ$C',fontsize='15')
plt.title ('Surface temperature vs time',fontsize='20')
plt.legend(["day 3(14/12/2023)", "day 4(15/12/2023)","day 5(16/12/2023)", "day 6(17/12/2023)","day 7(18/12/2023)"],
           loc = 'upper left',fontsize='10',markerscale=2,
           shadow = True, facecolor = 'cyan')

plt.grid()
plt.show()

# In[11]:

d = [day_3,day_4,day_5,day_6,day_7]
c = ['red','orange','yellow','green','blue']

plt.figure(figsize=[10,5])
for i in range(5):    
    plt.scatter (d[i]['surface'].to_numpy(),d[i]['inside'].to_numpy(),s=12,color=c[i])
for j in range(5):    
    mean = np.mean(d[j]['inside'].to_numpy())
    plt.axhline(y=mean, color = c[j], label = 'axhline',linewidth='2')
    
plt.xticks(fontsize='15')
plt.yticks(fontsize='15')
plt.xlim([24,31])
plt.xlabel ('Surface temperature in $^\circ$C',fontsize='15')
plt.ylabel ('Inside temperature in $^\circ$C',fontsize='15')
plt.title ('Inside temperature vs surface temperature',fontsize='20')
plt.legend(["day 3(14/12/2023)", "day 4(15/12/2023)","day 5(16/12/2023)", "day 6(17/12/2023)",
            "day 7(18/12/2023)","mean of inside temperature for day 3",
            "mean of inside temperature for day 4","mean of inside temperature for day 5",
            "mean of inside temperature for day 6","mean of inside temperature for day 7"],
           loc = 'lower right',fontsize='5',markerscale=2,shadow = True, facecolor = 'cyan')

plt.grid()
plt.show()