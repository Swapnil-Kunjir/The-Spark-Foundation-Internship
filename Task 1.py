#!/usr/bin/env python
# coding: utf-8

# **By Swapnil Kunjir**
# # The Sparks Foundation
# ## Data Science and Business Analytics Internship
# ### Task 1: Prediction Using Supervised ML (Predicting percentage of students based on number of study hours using linear regression)
# 
# * **Problem Statement - Predict Score of a student if he studies for 9.25 hours/day.**

# In[1]:


# All libraries required for this Analysis
import pandas as pd
import numpy as np
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


# In[2]:


# Reading data directly from URL
data=pd.read_csv('http://bit.ly/w-data')
data.head()


# In[3]:


data.shape


# In[4]:


data.describe().round(decimals=2)


# In[5]:


correlation=data.corr('pearson')
print('correlation between no. of study hours/day and Marks scored is',correlation.iloc[0,1])


# **WE can clearly see No. of study hours and Marks scored in exam are highly Positively correlated.**

# ### Scatter Plot for finding Nature of the relationship 

# In[6]:


import matplotlib.pyplot as plt
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# **From Scatter Plot we can see that there is strong positive linear relationship between No. of Study Hours And Marks Scored in Exam**

# ## Data Preparation
# Here we divide data into Features(independent variable(s)) and Label (dependent variable).  
# After That We divide data into 80% for Traing the model and 20% for Testing that model.

# In[7]:


X1=data.iloc[:,:-1].values
Y1=data.iloc[:,-1].values


# In[8]:


X=X1.reshape(-1,1)
Y=Y1.reshape(-1,1)


# In[9]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=42)


# ## Training Model
# Now we will use training data to train the linear regression model.

# In[21]:


LR=LinearRegression()
LR.fit(x_train,y_train)
print('Fitted Linear Regression Equation is: y=',LR.coef_,'*x+',LR.intercept_)


# In[13]:


plt.scatter(x_train,y_train,color='k')
plt.plot(x_train,(LR.coef_*x_train+LR.intercept_),color='r')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ## Testing model 
# Using test data we will see that how accurately our model can predict scores.

# In[14]:


y_predicted=LR.predict(x_test)
y_predicted


# In[15]:


p=pd.DataFrame(list(zip(x_test,y_test,y_predicted)),columns=['x','target_y','predicted_y'])
p.head()


# ## Model Evaluation
# Here we used mean absolute error instead of mean squared error because it gives equal weightage to all observations.    Unlike mean squared error which gives more weightage to the outliers.

# In[20]:


mean_absolute_error(y_test,y_predicted)


# **It can be seen that value of mean absolute error is low. So we caan say that our model fits data well.**  
# 

# In[16]:


plt.scatter(x_test,y_test,color='k')
plt.plot(x_test,y_predicted,color='r')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ## Prediction 
# Now we will predict score for our problem statement that is how much a student will score if he will study for 9.25 hours/day.

# In[17]:


hours = np.array([9.25])
hours=hours.reshape(-1,1)
own_pred = LR.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# **From this we can say that He will Score 92 marks if he studies for 9.25 hours/day.**

# In[ ]:




