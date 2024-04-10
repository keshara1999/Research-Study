# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error,r2_score

# In[3]:

calcsv= pd.read_csv(r'D:\uni\4th year\1st sem\RESEARCH\details\Data\1_Pulasthi lotus\2 flower\12,13,14,15 & 16_10_2023.csv')

# In[4]:

Day = calcsv['Day'].to_numpy()
time = calcsv['time'].to_numpy()
ambient_temp = calcsv['ambient'].to_numpy()
surface_temp = calcsv['surface'].to_numpy()
inside_temp = calcsv['inside'].to_numpy()
inside_err = calcsv['err_inside'].to_numpy()

# In[13]:

x = np.arange(len(Day))
plt.figure(figsize=[15,7])

plt.plot(x,ambient_temp,color='red',linewidth='2')
plt.plot(x,surface_temp,color='green',linewidth='2')
plt.plot(x,inside_temp,color='blue',linewidth='2')

y=[56,344,632,920,1202]
xl=['day 5 - 12:00PM\n(12/10/2023)','day 6 - 12:00PM\n(13/10/2023)','day 7 - 12:00PM\n(14/10/2023)','day 8 - 12:00PM\n(15/10/2023)','day 9 - 12:00PM\n(16/10/2023)']
plt.xticks(y,xl,rotation = 0,fontsize='15')
plt.yticks(fontsize='15')

x2=[200,488,776,1064]
for j in range (4):
    plt.axvline(x=x2[j], color = 'black', label = 'axvline',linewidth='0.8') 

plt.ylim([24,36])

plt.xlabel('Number of days since the flower started blooming',fontsize='20')
plt.ylabel('Temperature in $^\circ$C',fontsize='20')
plt.title('Temperature of ambient,surface and inside of the bud vs Days',fontsize='25')
plt.legend(["Ambient temperature", "Surface temperature","Inside temperature"],
           loc = 'upper left',fontsize='15',markerscale=6,
           shadow = True, facecolor = 'yellow')
plt.grid(axis='y')
plt.show()

# In[14]:

day_5 = calcsv.drop(index = calcsv[Day!='day 5'].index)
day_6 = calcsv.drop(index = calcsv[Day!='day 6'].index)
day_7 = calcsv.drop(index = calcsv[Day!='day 7'].index)
day_8 = calcsv.drop(index = calcsv[Day!='day 8'].index)
day_9 = calcsv.drop(index = calcsv[Day!='day 9'].index)

# In[15]:

g = [day_5,day_6,day_7,day_8,day_9]
for j in range (5):
    index=[]
    i=0
    for i in range (len(g[j]['time'].to_numpy())):
        index.append(i)
    g[j]['index']= index
    g[j]= g[j].set_index('index')

# In[16]:

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
plt.legend(["day 5(12/10/2023)", "day 6(13/10/2023)","day 7(14/10/2023)", "day 8(15/10/2023)",
            "day 9(16/10/2023)"],loc = 'upper left',fontsize='8',markerscale=2,
           shadow = True, facecolor = 'cyan')

plt.grid()
plt.show()

# In[17]:

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
plt.legend(["day 5(12/10/2023)", "day 6(13/10/2023)","day 7(14/10/2023)", "day 8(15/10/2023)","day 9(16/10/2023)"],
           loc = 'upper left',fontsize='10',markerscale=2,
           shadow = True, facecolor = 'cyan')

plt.grid()
plt.show()

# In[18]:

d = [g[0],g[1],g[2],g[3],g[4]]
c = ['red','orange','yellow','green','blue','cyan']
x1=[]
x2=[]
x3=[]

for j in range(0,5): 
    plt.figure(figsize=[10,5])
    plt.plot(d[j]['time'].to_numpy(),d[j]['inside'].to_numpy(),color=c[0])
    plt.plot(d[j]['time'].to_numpy(),d[j]['surface'].to_numpy(),color=c[4])
    
    y1=[3+3,15+3,27+3,39+3,51+3,63+3,75+3,87+3,99+3,111+3,123+3,135+3,147+3,159+3,171+3,183+3]
    y=[0,12,24,36,48,60,72,84,96,108,120,132,144,156,168,180,192,204,216,228,240,252,264,276]
    y2=[0,12,24,36,48,60,72,84,96,108,120,132]
    
    if j==0:
        for l in range(len(y1)):
            x1.append(g[0]['time'].to_numpy()[y1[l]])
        plt.xticks(x1,x1,rotation = 90,fontsize='15')
        
    elif j==4:
        for m in range(len(y2)):
            x3.append(g[4]['time'].to_numpy()[y2[m]])
        plt.xticks(x3,x3,rotation = 90,fontsize='15')
        
    else:        
        for k in range(len(y)):
            x2.append(g[1]['time'].to_numpy()[y[k]])
        plt.xticks(x2,x2,rotation = 90,fontsize='15')
        
    plt.yticks(fontsize='15')
    
    plt.ylim([24,34])
    plt.xlabel ('time',fontsize='15')
    plt.ylabel ('Inside temperature in $^\circ$C',fontsize='15')
    plt.title ('Inside temperature vs surface temperature',fontsize='20')
    plt.legend(["day"+str(j+5)],
               loc = 'lower right',fontsize='10',markerscale=2,
               shadow = True, facecolor = 'cyan')

    plt.grid()
    plt.show()

# In[20]:

d = [day_5,day_6,day_7,day_8,day_9]
c = ['red','orange','yellow','green','blue']

plt.figure(figsize=[10,5])
for i in range(5):    
    plt.scatter (d[i]['surface'].to_numpy(),d[i]['inside'].to_numpy(),s=100,color=c[i])
for j in range(5):    
    mean = np.mean(d[j]['inside'].to_numpy())
    plt.axhline(y=mean, color = c[j], label = 'axhline',linewidth='2')    

plt.xticks(fontsize='15')
plt.yticks(fontsize='15')
plt.xlim([24,33])
plt.xlabel ('Surface temperature in $^\circ$C',fontsize='15')
plt.ylabel ('Inside temperature in $^\circ$C',fontsize='15')
plt.title ('Inside temperature vs surface temperature',fontsize='20')
plt.legend(["day 5(12/10/2023)", "day 6(13/10/2023)","day 7(14/10/2023)", "day 8(15/10/2023)",
            "day 9(16/10/2023)","mean of inside temperature for day 5",
            "mean of inside temperature for day 6","mean of inside temperature for day 7",
            "mean of inside temperature for day 8","mean of inside temperature for day 9"],
           loc = 'lower right',fontsize='8',markerscale=0.4,
           shadow = True, facecolor = 'cyan')

plt.grid()
plt.show()

# In[22]:

d = [day_5,day_6,day_7,day_8,day_9]
c = ['red','orange','black','green','blue']

for i in range(5):
    plt.figure()
    plt.scatter (d[i]['surface'].to_numpy(),d[i]['inside'].to_numpy(),s=4,color=c[i])
    plt.errorbar(d[i]['surface'].to_numpy(),d[i]['inside'].to_numpy(),d[i]['err_inside'],fmt =".",color=c[i])
    
    b=np.polyfit(d[i]['surface'].to_numpy(),d[i]['inside'].to_numpy(),1)
    y2 = np.poly1d(b)
    x2 = np.arange(min(d[i]['surface'].to_numpy()),max(d[i]['surface'].to_numpy())+0.1,0.1)
    
    plt . plot(x2,y2(x2),color ="violet",linewidth ='2')
    print('coeficient of polynomial for day'+str(i+1),b)
    
    RMSE = sqrt(mean_squared_error(d[i]['inside'].to_numpy(),y2(d[i]['surface'].to_numpy())))
    R2 = r2_score(d[i]['inside'].to_numpy(),y2(d[i]['surface'].to_numpy()))
    
    y_error = y2(d[i]['surface'].to_numpy())-d[i]['inside'].to_numpy()
    
    print ('RMSE value = ',RMSE)
    print ('R2 value = ',R2)
    
    #plt.xlim([24,31])
    plt.ylim([28,34])
    plt.xlabel ('Surface temperature in $^\circ$C')
    plt.ylabel ('Inside temperature in $^\circ$C')
    plt.title ('Inside temperature vs surface temperature of flower bud on day '+ str(i+5)+" ("+str(i+12)+"/10/2023)")
    plt.legend(["Inside temperature", "linear fit curve"],
    loc = 'lower right',fontsize='10',markerscale=3,
    shadow = True, facecolor = 'cyan')  
    plt.grid()
           
    plt.figure(figsize=[20,5])
    plt . scatter(d[i]['surface'].to_numpy(),y_error,s=50,color=c[i])
    plt . plot(d[i]['surface'].to_numpy(),0*d[i]['surface'].to_numpy(),color='brown',linewidth='4') 
    #plt.xlim([24,31])

    plt.xticks(fontsize='25')
    plt.yticks(fontsize='25')
    plt.xlabel('Prob value',fontsize='25')
    plt.ylabel('Difference between predicted and\n actual temperatures in $^\circ$C',fontsize='20')
    plt.title('Residual plot for air region temperature vs prob value',fontsize='40')
    plt.legend(["Residuals", "Reference axis"],loc = 'upper left',fontsize='25',
               markerscale=2,shadow = True, facecolor = 'yellow')  

    plt.grid()
    
    plt.show()