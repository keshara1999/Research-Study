# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error,r2_score

# In[ ]:

calcsv= pd.read_csv(r'D:\uni\4th year\1st sem\RESEARCH\details\Data\1_Pulasthi lotus\1 flower\08 ,09,10 & 11_09_2023.csv')

# In[ ]:

Day = calcsv['Day'].to_numpy()
time = calcsv['time'].to_numpy()
ambient_temp = calcsv['ambient'].to_numpy()
surface_temp = calcsv['surface'].to_numpy()
inside_temp = calcsv['inside'].to_numpy()
inside_err = calcsv['err_inside'].to_numpy()

# In[ ]:

x = np.arange(len(Day))
plt.figure(figsize=[15,7])

plt.plot(x,ambient_temp,color='red')
plt.plot(x,surface_temp,color='green')
plt.plot(x,inside_temp,color='blue')

y=[1,271,559,774]
xl=["day 7 - 1:35PM\n(08/09/2023)","day 8 - 12:00PM\n(09/09/2023)","day 9 - 12:00PM\n(10/09/2023)","day 10 - 5:55AM\n(11/09/2023)"]
plt.xticks(y,xl,fontsize='15')
plt.yticks(fontsize='15')

x2=[127,415,703]
for j in range (3):
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

# In[ ]:

day_7 = calcsv.drop(index = calcsv[Day!='day 7'].index)
day_8 = calcsv.drop(index = calcsv[Day!='day 8'].index)
day_9 = calcsv.drop(index = calcsv[Day!='day 9'].index)
day_10 = calcsv.drop(index = calcsv[Day!='day 10'].index)

# In[ ]:

g = [day_7,day_8,day_9,day_10]
for j in range (4):
    index=[]
    i=0
    for i in range (len(g[j]['time'].to_numpy())):
        index.append(i)
    g[j]['index']= index
    g[j]= g[j].set_index('index')

# In[ ]:

d = [g[0],g[1],g[2],g[3]]
c = ['red','orange','green','blue']
xl= []

plt.figure(figsize=[40,10])
for j in range(3):
    x=[]
    for k in range (len(d[1]['inside'].to_numpy())-len(d[j]['inside'].to_numpy()),len(d[1]['inside'].to_numpy())):
        x.append(k)
        
    plt.plot(x,d[j]['inside'].to_numpy(),color=c[j],linewidth='6')

x=[]
for k in range (0,len(d[3]['inside'].to_numpy())):
    x.append(k)
plt.plot(x,d[3]['inside'].to_numpy(),color=c[3],linewidth='6')  

y=[0,12,24,36,48,60,72,84,96,108,120,132,144,156,168,180,192,204,216,228,240,252,264,276]

for k in range(len(y)):
    xl.append(g[1]['time'].to_numpy()[y[k]])

plt.xticks(y,xl,rotation = 90,fontsize='40')
plt.yticks(fontsize='40')

#plt.xlim([24,31])
plt.xlabel ('Time',fontsize='45')
plt.ylabel ('Inside temperature in $^\circ$C',fontsize='45')
plt.title ('Inside temperature vs time',fontsize='60')
plt.legend(["day 7(08/09/2023)", "day 8(09/09/2023)","day 9(10/09/2023)", "day 10(11/09/2023)"],
           loc = 'lower right',fontsize='20',markerscale=2,
           shadow = True, facecolor = 'cyan')

plt.grid()
plt.show()

# In[ ]:

d = [g[0],g[1],g[2],g[3]]
c = ['red','orange','green','blue']
xl= []

plt.figure(figsize=[10,5])
for j in range(3):
    x=[]
    for k in range (len(d[1]['inside'].to_numpy())-len(d[j]['inside'].to_numpy()),len(d[1]['inside'].to_numpy())):
        x.append(k)
        
    plt.plot(x,d[j]['surface'].to_numpy(),color=c[j])

x=[]
for k in range (0,len(d[3]['inside'].to_numpy())):
    x.append(k)
plt.plot(x,d[3]['surface'].to_numpy(),color=c[3])  

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
plt.legend(["day 7(08/09/2023)", "day 8(09/09/2023)","day 9(10/09/2023)", "day 10(11/09/2023)"],
           loc = 'upper left',fontsize='8',markerscale=2,
           shadow = True, facecolor = 'cyan')

plt.grid()
plt.show()

# In[ ]:

d = [g[0],g[1],g[2],g[3]]
c = ['red','orange','yellow','green','blue','cyan']
x1=[]
x2=[]
x3=[]

for j in range(0,4): 
    plt.figure(figsize=[10,5])
    plt.plot(d[j]['time'].to_numpy(),d[j]['inside'].to_numpy(),color=c[0])
    plt.plot(d[j]['time'].to_numpy(),d[j]['surface'].to_numpy(),color=c[3])
    
    y1=[3+2,15+2,27+2,39+2,51+2,63+2,75+2,87+2,99+2,111+2]
    y=[0,12,24,36,48,60,72,84,96,108,120,132,144,156,168,180,192,204,216,228,240,252,264,276]
    y2=[0,12,24,36,48,60]
    
    if j==0:
        for l in range(len(y1)):
            x1.append(g[0]['time'].to_numpy()[y1[l]])
        plt.xticks(x1,x1,rotation = 90,fontsize='15')
        
    elif j==3:
        for m in range(len(y2)):
            x3.append(g[3]['time'].to_numpy()[y2[m]])
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
    plt.legend(["day"+str(j+7)+" ("+str(j+8)+"/09/2023)"],
               loc = 'lower right',fontsize='10',markerscale=2,
               shadow = True, facecolor = 'cyan')

    plt.grid()
    plt.show()

# In[ ]:

d = [day_7,day_8,day_9,day_10]
c = ['red','blue','orange','green','yellow']

plt.figure(figsize=[10,5])
for i in range(4):    
    plt.scatter (d[i]['surface'].to_numpy(),d[i]['inside'].to_numpy(),s=100,color=c[i])
for j in range(4):    
    mean = np.mean(d[j]['inside'].to_numpy())
    plt.axhline(y=mean, color = c[j], label = 'axhline',linewidth='2')

plt.xticks(fontsize='15')
plt.yticks(fontsize='15')
plt.xlim([24,33])
plt.xlabel ('Surface temperature in $^\circ$C',fontsize='15')
plt.ylabel ('Inside temperature in $^\circ$C',fontsize='15')
plt.title ('Inside temperature vs surface temperature',fontsize='20')
plt.legend(["day 7(08/09/2023)", "day 8(09/09/2023)","day 9(10/09/2023)", "day 10(11/09/2023)",
            "mean of inside temperature for day 7","mean of inside temperature for day 8",
            "mean of inside temperature for day 9",
            "mean of inside temperature for day 10"],loc = 'lower right',fontsize='8',
           markerscale=0.4,shadow = True, facecolor = 'cyan')

plt.grid()
plt.show()

# In[ ]:

d = [day_7,day_8,day_9,day_10]
c = ['red','orange','black','green','blue']

for i in range(4):
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
    plt.title ('Inside temperature vs surface temperature of flower bud on day '+ str(i+7))
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

# In[ ]: