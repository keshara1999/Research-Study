import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

# In[20]:

calcsv= pd.read_csv(r'D:\uni\4th year\1st sem\RESEARCH\details\Data\2-Manel\14,15,16,17 & 18_12_2023.csv')

# In[21]:

surface_train1,surface_test1,inside_train1,inside_test1 = train_test_split(calcsv[['surface']],calcsv.inside,test_size=0.04,random_state=0)

# In[22]:

plt.figure(figsize=[12,8])
plt.scatter(surface_train1,inside_train1,s=15,color='blue')

b=np.polyfit(surface_train1['surface'].to_numpy(),inside_train1.to_numpy(),2)
print('poly coef =',b)

y = np.poly1d(b)
x = np.arange(min(surface_train1['surface'].to_numpy()),max(surface_train1['surface'].to_numpy()),0.01)

plt.plot(x,y(x),color ="red",linewidth ='4')

MSE_of_inside_new_temp = mean_squared_error(inside_train1.to_numpy(),y(surface_train1['surface'].to_numpy()))
R2_of_inside_new_temp = r2_score(inside_train1.to_numpy(),y(surface_train1['surface'].to_numpy()))

print ('MSE value of prdicted inside temp value vs actual inside temp for training dataset = ',MSE_of_inside_new_temp)
print ('R2 value of prdicted inside temp value vs actual inside temp for training dataset = ',R2_of_inside_new_temp)

plt.xlim([min(surface_train1['surface'].to_numpy())-1,max(surface_train1['surface'].to_numpy())+1])

plt.xticks(fontsize='15')
plt.yticks(fontsize='15')

plt.xlabel('Surface temperature values in $^\circ$C',fontsize='20')
plt.ylabel('Inside temperature values in $^\circ$C',fontsize='20')
plt.title('Temperature values of inside vs outer surface for \ntraining data set',fontsize='25')
plt.legend([ "Actual inside temperature values","Predicted inside temperature value"],
           loc = 'upper left',fontsize='15',markerscale=4,
           shadow = True, facecolor = 'yellow')  
          
plt.grid()
plt.show()

# In[23]:

plt.figure(figsize=[12,8])

plt.scatter(surface_test1['surface'].to_numpy(),inside_test1.to_numpy(),s=50,color='green')

x2 = np.arange(min(surface_test1['surface'].to_numpy()),max(surface_test1['surface'].to_numpy()),0.01)
plt.plot(x2,y(x2),color ="orange",linewidth ='4')

MSE_of_inside_test = mean_squared_error(inside_test1.to_numpy(),y(surface_test1['surface'].to_numpy()))
R2_of_inside_test = r2_score(inside_test1.to_numpy(),y(surface_test1['surface'].to_numpy()))

print ('MSE value of prdicted inside temp value vs actual inside temp for testing dataset = ',MSE_of_inside_test)
print ('R2 value of prdicted inside temp value vs actual inside temp for testing dataset = ',R2_of_inside_test)

plt.xlim([min(surface_test1['surface'].to_numpy())-1,max(surface_test1['surface'].to_numpy())+1])

plt.xticks(fontsize='15')
plt.yticks(fontsize='15')

plt.xlabel('Surface temperature values in $^\circ$C',fontsize='20')
plt.ylabel('Inside temperature values in $^\circ$C',fontsize='20')
plt.title('Temperature values of inside vs outer surface for\n testing data set',fontsize='25')
plt.legend(["Actual inside temperature values","Predicted inside temperature value"],
           loc = 'upper left',fontsize='15',markerscale=2,
           shadow = True, facecolor = 'yellow')  
          
plt.grid()
plt.show()

# In[24]:

x=[]
y_train_e=[]
y_test_e=[]
dif=[]

plt.figure()
for i in range (1,50): 
    surface_train1,surface_test1,inside_train1,inside_test1 = train_test_split(calcsv[['surface']],calcsv.inside,test_size=i*0.02,random_state=0)
    model=LinearRegression()
    model.fit(surface_train1,inside_train1)
    
    b=np.polyfit(surface_train1['surface'].to_numpy(),inside_train1.to_numpy(),2)
    y = np.poly1d(b)
    
    MSE_training = mean_squared_error(inside_train1.to_numpy(),y(surface_train1['surface'].to_numpy()))
    MSE_testing = mean_squared_error(inside_test1.to_numpy(),y(surface_test1['surface'].to_numpy()))
    
    x.append(i*0.02)
    y_train_e.append(MSE_training)
    y_test_e.append(MSE_testing)
    
plt.plot(x,y_train_e,c='red')
plt.plot(x,y_test_e,c='blue')

plt.scatter(x,y_train_e,c='red',s=20)
plt.scatter(x,y_test_e,c='blue',s=20)

plt.xlabel('Test size',fontsize='15')
plt.ylabel('MSE value',fontsize='15')
plt.title('MSE value vs Test size',fontsize='20')
plt.legend(["Train dataset", "Test dataset"],
           loc = 'lower left',fontsize='10',markerscale=4,
           shadow = True, facecolor = 'yellow')  
plt.grid()
plt.show()