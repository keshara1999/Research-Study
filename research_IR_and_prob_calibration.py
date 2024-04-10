# In[49]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error,r2_score

# In[50]:

probcalcsv= pd.read_csv(r'D:\uni\4th year\1st sem\RESEARCH\details\Data\Calibration\Prob calibration\prob_data.csv')
IRcalcsv= pd.read_csv(r'D:\uni\4th year\1st sem\RESEARCH\details\Data\Calibration\IR calibration\IR_data.csv')

# In[51]:

Ambient_temp = probcalcsv['amb'].to_numpy()
water_temp = probcalcsv['water'].to_numpy()
prob_value = probcalcsv['prob'].to_numpy()

# In[52]:

Alcohol = IRcalcsv['Hg therm'].to_numpy()
IR = IRcalcsv['IR_object'].to_numpy()

# In[53]:

IR_new = np.poly1d(np.polyfit(Alcohol,IR,1))

plt.figure(figsize=[15,10])
plt.scatter(Alcohol,IR,color='green',s=100)
plt.plot(Alcohol,IR_new(Alcohol),color='red',linewidth ='4')

plt.xticks(fontsize='12')
plt.yticks(fontsize='15')

plt.ylim([23,45])

RMSE_of_IR_Alcohol_temp = sqrt(mean_squared_error(Alcohol,IR))
R2_of_IR_Alcohol_temp = r2_score(Alcohol,IR)

RMSE_of_IR_new_temp = sqrt(mean_squared_error(Alcohol,IR_new(Alcohol)))
R2_of_IR_new_temp = r2_score(Alcohol,IR_new(Alcohol))

print ('RMSE value of IR temp vs Alcohol temp = ',RMSE_of_IR_Alcohol_temp)
print ('R2 value of IR temp vs Alcohol temp = ',R2_of_IR_Alcohol_temp)

print ('RMSE value of new temp vs IR temp = ',RMSE_of_IR_new_temp)
print ('R2 value of new temp vs IR temp = ',R2_of_IR_new_temp)
    
plt.xlabel('Alcohol thermometer value for water surface temperature ($^\circ$C)',fontsize='20')
plt.ylabel('IR sensor value for water surface temperature ($^\circ$C)',fontsize='18')
plt.title('Graph of IR sensor value vs Alcohol thermometer value for water\nsurface temperature',fontsize='25')
plt.legend(["Alcohol thermometer value", "IR sensor value"],
           loc = 'upper left',fontsize='15',markerscale=1,
           shadow = True, facecolor = 'yellow')
plt.grid(axis='y')
plt.show()

# In[54]:

IRcal_40 = IRcalcsv.drop(index = IRcalcsv[Alcohol>40].index)

# In[55]:

Alcohol_40 = IRcal_40['Hg therm'].to_numpy()
IR_40 = IRcal_40['IR_object'].to_numpy()

# In[56]:

IR_new_40 = np.poly1d(np.polyfit(Alcohol_40,IR_40,1))

plt.figure(figsize=[15,10])
plt.scatter(Alcohol_40,IR_40,color='blue',s=100)
plt.plot(Alcohol_40,IR_new_40(Alcohol_40),color='orange',linewidth ='4')

plt.xticks(fontsize='12')
plt.yticks(fontsize='15')

plt.ylim([24,41])

RMSE_of_IR_Alcohol_temp_40 = sqrt(mean_squared_error(Alcohol_40,IR_40))
R2_of_IR_Alcohol_temp_40 = r2_score(Alcohol_40,IR_40)

RMSE_of_IR_new_temp_40 = sqrt(mean_squared_error(Alcohol_40,IR_new_40(Alcohol_40)))
R2_of_IR_new_temp_40 = r2_score(Alcohol_40,IR_new_40(Alcohol_40))

print ('RMSE value of IR temp vs Alcohol temp less than 40 = ',RMSE_of_IR_Alcohol_temp_40)
print ('R2 value of IR temp vs Alcohol temp less than 40 = ',R2_of_IR_Alcohol_temp_40)

print ('RMSE value of new temp vs IR temp = ',RMSE_of_IR_new_temp_40)
print ('R2 value of new temp vs IR temp = ',R2_of_IR_new_temp_40)
    
plt.xlabel('Alcohol thermometer value for water surface temperature ($^\circ$C)',fontsize='20')
plt.ylabel('IR sensor value for water surface temperature ($^\circ$C)',fontsize='18')
plt.title('Graph of IR sensor value vs Alcohol thermometer value less than 40 Celsius for water\nsurface temperature',fontsize='25')
plt.legend(["Alcohol thermometer value", "IR sensor value"],
           loc = 'upper left',fontsize='15',markerscale=1,
           shadow = True, facecolor = 'yellow')
plt.grid(axis='y')
plt.show()

# In[57]:

plt.figure(figsize=[15,10])
plt.scatter(prob_value,Ambient_temp,s=40,color='red')

b=np.polyfit(prob_value,Ambient_temp,2)
print('ambient =',b)

y2 = np.poly1d(b)
x2 = np.arange(410,660,1)

plt . plot(x2,y2(x2),color ="green",linewidth ='4')

RMSE_of_pred_actual = sqrt(mean_squared_error(Ambient_temp,y2(prob_value)))
R2_of_pred_actual = r2_score(Ambient_temp,y2(prob_value))

print ('RMSE value of pred val vs Actual val = ',RMSE_of_pred_actual)
print ('R2 value of pred val vs Actual val = ',R2_of_pred_actual)

plt.xlim([410,660])

plt.xticks(fontsize='15')
plt.yticks(fontsize='15')
plt.xlabel('Thermistor value',fontsize='20')
plt.ylabel('Temperature in $^\circ$C',fontsize='20')
plt.title('Graph of temperature of Air region vs Thermistor value',fontsize='25')
plt.legend(["Air region temperature","2$^n$$^d$ order best fit line"],
           loc = 'upper left',fontsize='15',markerscale=1,
           shadow = True, facecolor = 'yellow')  
          
plt.grid()
plt.show()

# In[58]:

prob_new = probcalcsv.drop(index = probcalcsv[prob_value>550].index)

# In[59]:

Ambient_new = prob_new['amb'].to_numpy()
water_new = prob_new['water'].to_numpy()
prob_new = prob_new['prob'].to_numpy()

# In[60]:

plt.figure(figsize=[15,10])
plt.scatter(prob_new,Ambient_new,s=40,color='blue')

b=np.polyfit(prob_new,Ambient_new,2)
print('ambient =',b)

y3 = np.poly1d(b)
x3 = np.arange(410,555,1)

plt . plot(x3,y3(x3),color ="orange",linewidth ='4')

RMSE_of_pred_actual = sqrt(mean_squared_error(Ambient_new,y3(prob_new)))
R2_of_pred_actual = r2_score(Ambient_new,y3(prob_new))

print ('RMSE value of pred val vs Actual val = ',RMSE_of_pred_actual)
print ('R2 value of pred val vs Actual val = ',R2_of_pred_actual)

plt.xlim([410,555])

plt.xticks(fontsize='15')
plt.yticks(fontsize='15')
plt.xlabel('Thermistor value',fontsize='20')
plt.ylabel('Temperature in $^\circ$C',fontsize='20')
plt.title('Graph of temperture of air region values vs thermister value less than 550',fontsize='25')
plt.legend(["Air region temperature","2$^n$$^d$ order best fit line"],
           loc = 'upper left',fontsize='15',markerscale=1,
           shadow = True, facecolor = 'yellow')  
          
plt.grid()
plt.show()

# In[ ]: