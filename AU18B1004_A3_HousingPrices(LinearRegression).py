#!/usr/bin/env python
# coding: utf-8

# ### Housing Price Predection

# ### Importing Required Libraries

# In[83]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import seaborn as sns
#import warnings
#warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Importing & Reading Data
# 
# Data is being imported using pandas .read_csv() function

# In[2]:


df = pd.read_csv(r'C:\Users\utkar\OneDrive\Desktop\Machine Learning\London.csv')
df.head()


# ### Understanding The Data
# 
# We have a dataset which contain data regarding the house prices in London. 
# 
# -   **Property Name** e.g. Queen ROad
# -   **Price** e.g. 735000
# -   **House Type** e.g. House
# -   **Area in sq ft** e.g. 814
# -   **No. of Bedrooms** e.g. 5
# -   **No. of Bathrooms** e.g 4
# -   **No. of Receptions** e.g. 5
# -   **Location** e.g. Wimbledon
# -   **City/County** e.g. London
# -   **Postal Code** e.g. SW19 8NY

# ### Data Exploration
# 
# First we will check how many rows and columns we have in our dataset and to do so we will use .shape 

# In[3]:


df.shape


# Now to get a descreptive analysis of out dataset we will use .describe() function

# In[4]:


df.describe()


# To check whether we have null values or not we will use .isnull().sum() method

# We will drop the Column which we don't require

# In[7]:


df.info()


# In[5]:


df.isnull().sum()


# Now to treat null values in Location column we will replace the null values by its mode. We do so by using .fillna() for filling null values and .mode() for finding mode of the column.

# In[8]:


df['Location'].fillna(df['Location'].mode()[0], inplace = True)


# In[6]:


df.drop(['Unnamed: 0'],axis=1,inplace=True)


# After filling the values we will again check the null values

# In[9]:


df.isnull().sum()


# Correlation Matrix to check the correlation between Price(Dependent Variable) and other Independent Variable

# In[10]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True)


# In[12]:


cdf = df[['Property Name','Price','House Type','Area in sq ft','No. of Bedrooms','No. of Bathrooms', 'No. of Receptions', 'Location', 'City/County', 'Postal Code']]
cdf.head()


# Now we will create a new column using lambda function

# In[13]:


cdf['City'] = cdf['City/County'].apply(lambda x: 1 if x == "London" else 0)
print(cdf['City'])


# In[14]:


sns.scatterplot(x = cdf['City'], y = cdf['Price'])


# In[15]:


cdf['City'].unique()


# Now we will plot the dependent variable graph to check if it is rightly skewed 

# In[84]:


sns.distplot(cdf['Price'])


# Now to check the relation between different columns we will plot a series of graph

# In[77]:


sns.scatterplot(x = cdf['Area in sq ft'], y = cdf['Price'], hue = df['House Type']) 


# In[18]:


sns.scatterplot(x = cdf['No. of Receptions'], y = cdf['Price'], hue = df['House Type']) 


# Since the values of Price is a large value so we will apply a log transformation

# In[19]:


cdf['LogPrice'] = np.log(cdf['Price'])


# In[20]:


cdf['Price_per_area'] = (cdf['LogPrice']*cdf['Area in sq ft'])


# In[82]:


sns.distplot(cdf['LogPrice'])


# To convert the categorical values into numerical value we will use Label Encoder 

# In[25]:


#Import library:
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
#New variable for outlet
cdf['Housetype'] = le.fit_transform(cdf['House Type'])
var_mod = ['Price', 'Area in sq ft', 'Housetype', 'No. of Bedrooms', 'City/County']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])


# In[85]:


cdf['PropertyName'] = le.fit_transform(cdf['Property Name'])
var_model = ['Price', 'Area in sq ft', 'Housetype', 'No. of Bedrooms', 'City/County', 'PropertyName']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])


# In[86]:


cdf.head()


# In[26]:


cdf['Area_sq'] = cdf['Area in sq ft']
cdf['Bedrooms'] = cdf['No. of Bedrooms']


# In[42]:


data = cdf[['Area_sq', 'Bedrooms', 'City', 'Housetype','Price_per_area']]


# In[43]:


X = data.drop('Price_per_area', axis=1) #independent variables

Y = data['Price_per_area'] #dependent variable


# We will now split the data into test and train part and now we will use linear regression model to predict the dependent variable

# In[44]:


from sklearn.model_selection import train_test_split #train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100) 


# In[35]:


#test_train = np.random.rand(len(df)) < 0.8
#train = cdf[test_train]
#test = cdf[~test_train]


# Using linear model from sklearn library and fit our model and find regression coefficent and regression intercept

# In[45]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['Area_sq', 'Bedrooms', 'City', 'Housetype']])
y = np.asanyarray(train[['Price_per_area']])
regr.fit(X_train,Y_train)
print(regr.coef_)


# To predict we will use regression predict function

# In[47]:


pred = regr.predict(X_test)


# In[54]:


#ypredict = regr.predict(test[['Area_sq', 'Bedrooms', 'City', 'Housetype']])
#x = np.asanyarray(test[['Area_sq', 'Bedrooms', 'City', 'Housetype']])
#y = np.asanyarray(test[['Price_per_area']])
#print("Residual sum of squares: %.2f"
 #     % np.mean((ypredict - y) ** 2))
#print('Variance score: %.2f' % regr.score(x, y))


# ### Evaluation
# 
# ```
# - Mean absolute error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just average error.
# - Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean absolute error because the focus is geared more towards large errors. This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.
# - Root Mean Squared Error (RMSE).
# - R-squared is not error, but is a popular metric for accuracy of your model. It represents how close the data are to the fitted regression line. The higher the R-squared, the better the model fits your data. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
# ```

# In[59]:


from sklearn.metrics import mean_squared_error
print("Mean absolute error: %.2f" % np.mean(np.absolute(Y_test - pred)))
rmse = np.sqrt(mean_squared_error(Y_test, pred)) #RMSE for the predicted value
rmse


# In[53]:


regr.score(X_test, Y_test)

