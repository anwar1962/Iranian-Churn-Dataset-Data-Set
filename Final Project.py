#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
#Loading the case1churn.csv dataset
datainput = pd.read_csv('Case1Churn.csv')
datainput.head()


# In[7]:


#checking total values of the case1churn.csv dataset
datainput['Churn'].value_counts()


# In[8]:


#Exploring dataset/checking customer churn by age
print(datainput.groupby('Age')['Churn'].value_counts())


# In[9]:


# Import matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
  
# Visualize the distribution of 'Total day minutes'
plt.hist(datainput['Seconds of Use'], bins = 100)
  
# Display the plot
plt.show()


# In[10]:


# Creating a box plot to understand the binary variables
sns.boxplot(x = 'Churn',
            y = 'Tariff Plan',
            data = datainput,
            sym = "",                  
            hue = "Complains") 
# Display the plot
plt.show()


# In[11]:


# Displaying information about the dataset
datainput.info()


# In[12]:


# Diplaying summary of the dataset
datainput.describe()


# In[13]:


# Preprocessing the dataset/dropping unwanted variables

datainput.drop(['FN','FP'], axis=1, inplace=True)
datainput


# In[14]:


# Visualizing summary of churn in the dataset

plt.figure(figsize=(10,10))
plt.pie(x=[495, 2655], labels=['1','0'], autopct='%1.0f%%', pctdistance=0.5,labeldistance=0.7,colors=['b','r'])
plt.title('Distribution of Churn Customers')


# In[15]:


# Creating a correlation matrix plot
corr = datainput.corr()
plt.figure(figsize=(18,8))
sns.heatmap(corr, annot = True, cmap='twilight')


# In[16]:


# Creating a correlation matrix plot in order

fig, ax = plt.subplots(figsize=(10,14))
churn_data_corr = datainput.corr()[['Churn']].sort_values(
  by='Churn', ascending=False)
sns.heatmap(churn_data_corr, annot=True, ax=ax)


# In[17]:


#Identifying response variable:
    
response = datainput['Churn']
dataset = datainput.drop(columns = 'Churn')


# In[18]:


# Creating a training and testing data 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(datainput, response,stratify = response, test_size = 0.2)


# In[19]:


print("Number transactions X_train datainput: ", X_train.shape)
print("Number transactions y_train datainput: ", y_train.shape)
print("Number transactions X_test datainput: ", X_test.shape)
print("Number transactions y_test datainput: ", y_test.shape)


# In[62]:


# Creating Support Vector Machine and Predicting 

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear', probability= True)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)


# In[64]:


# Creating Evaluation Metrics of the Support Vector Machine 
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:




