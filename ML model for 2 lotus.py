# In[171]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder,LabelBinarizer

# In[172]:

obj_en = LabelEncoder()
obj_bin = LabelBinarizer()

# In[173]:

calcsv2= pd.read_csv(r'E:\uni\4th year\1st sem\RESEARCH\details\Data\1_Pulasthi lotus\2 flower\12,13,14,15 & 16_10_2023.csv')
calcsv1= pd.read_csv(r'E:\uni\4th year\1st sem\RESEARCH\details\Data\1_Pulasthi lotus\1 flower\08 ,09,10 & 11_09_2023.csv')

# In[174]:

calcsv=pd.concat([calcsv1,calcsv2])

# In[175]:

ind=[]
for i in range (0,1974):
    ind.append(i)    
calcsv['index']= ind
calcsv= calcsv.set_index('index')

# In[176]:

day_cat = calcsv['Day']
day=obj_en.fit_transform(day_cat)

# In[177]:

day_bin=obj_bin.fit_transform(day_cat)

# In[178]:

calcsv['Day5']=day_bin[:,0]
calcsv['Day6']=day_bin[:,1]
calcsv['Day7']=day_bin[:,2]
calcsv['Day8']=day_bin[:,3]
calcsv['Day9']=day_bin[:,4]
calcsv['Day10']=day_bin[:,5]

# In[179]:

surface_train1,surface_test1,inside_train1,inside_test1 = train_test_split(calcsv[['Day5','Day6','Day7','Day8','Day9','Day10','surface']],calcsv.inside,test_size=0.06,random_state=5)

# In[180]:

model=LinearRegression()
model.fit(surface_train1,inside_train1)

# In[181]:

polinomial=[]
polinomial=np.append(model.coef_,model.intercept_)
print(polinomial)

# In[182]:

pred_train=model.predict(surface_train1)
real_train=inside_train1.to_numpy()

# In[183]:

MSE_training = mean_squared_error(real_train,pred_train)
print ('MSE value of training dataset = ',MSE_training)

# In[184]:

pred_test=model.predict(surface_test1)
real_test=inside_test1.to_numpy()

# In[185]:

MSE_testing = mean_squared_error(real_test,pred_test)
print ('MSE value of testing dataset = ',MSE_testing)

# In[186]:

model.score(surface_train1,inside_train1)

# In[187]:

model.score(surface_test1,inside_test1)

# In[188]:

x=[]
y_train_e=[]
y_test_e=[]
dif=[]

plt.figure()
for i in range (1,50): 
    surface_train1,surface_test1,inside_train1,inside_test1 = train_test_split(calcsv[['Day5','Day6','Day7','Day8','Day9','Day10',
    'surface','ambient']],calcsv.inside,test_size=i*0.02,random_state=5)
   
    model=LinearRegression()
    model.fit(surface_train1,inside_train1)
    
    pred_train=model.predict(surface_train1)
    real_train=inside_train1.to_numpy()
    MSE_training = mean_squared_error(real_train,pred_train)
    
    pred_test=model.predict(surface_test1)
    real_test=inside_test1.to_numpy()
    MSE_testing = mean_squared_error(real_test,pred_test)
    
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