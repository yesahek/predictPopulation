#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report

import pandas as pd
import matplotlib.pyplot as plt


# ## load the DataSet

# In[2]:


data = pd.read_csv('population.csv')


# ## Split the dataset into features (x) and target (y)

# In[3]:


x = data.iloc[:, :-1].values


# In[4]:


y = data.iloc[:, -1].values


# ## Split the dataset into training and testing sets

# In[79]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# ## Let's Train the model

# In[80]:


model = LinearRegression()


# In[81]:


model.fit(x_train, y_train)


# ## Let's Evaluate the model

# In[82]:


predictions = model.predict(x_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


# In[83]:


print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[84]:


# Output the predicted values


# In[85]:


predictions = [int(i) for i in predictions]
x_test = [int(i) for i in x_test]


# In[86]:


pd.DataFrame({'year':x_test ,'test': y_test, 'predictions':predictions})


# In[90]:


plt.scatter(y_test, predictions)


# In[91]:


plt.hist(y_test - predictions)


# In[92]:


year = 2023


# In[93]:


population = model.predict([[year]])


# In[94]:


print('The predicted population in', year, 'is', float(population))


# In[ ]:




