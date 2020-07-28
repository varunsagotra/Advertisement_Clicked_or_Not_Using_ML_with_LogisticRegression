#!/usr/bin/env python
# coding: utf-8

# # Advertisement_Clicked_or_Not_Using_ML_with_Logistic Regression 
# 
# `ML micro-project using Logistic Regression` :
# 
# In this project we will work with a fake advertising data set, indicating whether or not a particular internet user clicked on an advertisement. 
# 
# `We will create a logistic regression model that will predict whether or not a user will click on an ad, based on his/her features. As this is a binary classification problem, a logistic regression model is well suited here.`

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# `## Data`
# 
# The data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad

# In[2]:


ad_data = pd.read_csv('/Users/ceo/Desktop/advertising.csv')


# In[3]:


ad_data.head()


# In[4]:


ad_data.info()


# In[5]:


ad_data.describe()


# ## Exploratory Analysis
# 
# Checking out the distribution of user age.

# In[6]:


plt.hist(ad_data['Age'],bins=30)
plt


# `Checking out the relationship between age and daily time spent on site.`

# In[7]:


sns.jointplot('Age','Daily Time Spent on Site',ad_data)


# `And the relationship between daily time spent on site and daily internet usage.`

# In[8]:


sns.jointplot('Daily Time Spent on Site','Daily Internet Usage',ad_data)


# `Finally, a pairplot to visualise everything else, colored on the basis of whether they clicked the ad or not.`

# In[9]:


sns.pairplot(ad_data,hue='Clicked on Ad')


# # Model Building

# `We'll split the data into training set and testing set using train_test_split, but first, let's convert the 'Country' feature to an acceptable form for the model.`

# In[10]:


ad_data.columns


# As we can't directly use the 'Country' feature (because it's a categorical string), we have to find another way to feed it into the model.
# 
# One way to go about this is to drop the feature, but we risk losing useful information. 
# 
# So, what we can do is, `convert the categorical feature into [dummy variables] using pandas.`

# In[11]:


countries = pd.get_dummies(ad_data['Country'],drop_first=True)


# `Concatenating dummy variables with the original dataset, and dropping other features.`

# In[12]:


ad_data = pd.concat([ad_data,countries],axis=1)


# In[13]:


ad_data.drop(['Country','Ad Topic Line','City','Timestamp'],axis=1,inplace=True)


# `Splitting the dataset`

# In[14]:


X = ad_data.drop('Clicked on Ad',axis=1)
y = ad_data['Clicked on Ad']


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# `Training the model`

# In[17]:


from sklearn.linear_model import LogisticRegression


# In[18]:


logclf = LogisticRegression()


# In[19]:


logclf.fit(X,y)


# ## Predictions and Evaluations

# In[20]:


predictions = logclf.predict(X_test)


# `Classification report for the model`

# In[30]:


from sklearn.metrics import classification_report
print("\n Classification Report \n *********************\n\n",classification_report(y_test,predictions))


# In[ ]:




