#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':[10,10]},font_scale=1.2)


# In[2]:


get_ipython().system('pip install imblearn')


# In[3]:


df_train=pd.read_excel("C:/Users/hicha/Desktop/train.xlsx")
df_test=pd.read_excel("C:/Users/hicha/Desktop/test.xlsx")


# In[4]:


#df_train.head(5)
df_train.tail()


# In[5]:


get_ipython().system('pip install xgboost')


# In[6]:


df_train.describe()


# In[7]:


df_train.info()


# In[8]:


df_train.isnull().sum()


# In[9]:


df_train.nunique()


# In[10]:


nu=df_train.isnull().sum()
nu[nu>0]


# In[11]:


plt.figure(figsize=(4,4)) 
sns.heatmap(df_train.isnull())


# ## Clean Data

# In[12]:


def clean(d):
    d.drop(['Cabin','Name','Ticket','Embarked','Fare'],axis=1,inplace=True)
    d.Age=d.Age.fillna(d.Age.median())
    d.dropna()
    return d


# In[13]:


clean(df_train)


# In[14]:


clean(df_test)


# In[15]:


plt.figure(figsize=(4,4))  
sns.heatmap(df_train.isnull())


# ## Data Analysis

# In[16]:


co=df_train.corr()


# In[17]:


plt.figure(figsize=(5,5))   
sns.heatmap(co,annot=True,fmt='.1f',linewidth=.3)


# In[18]:


plt.figure(figsize=(7,6))   
df_train.Survived.value_counts()


# In[19]:


df_train.Sex.value_counts()


# In[20]:


plt.figure(figsize=(4,4))   
df_train.Sex.value_counts().plot.pie(autopct='%0.02f%%')


# In[21]:


plt.figure(figsize=(4,4))   
sns.countplot(df_train.Sex,hue=df_train.Survived)


# In[22]:


plt.figure(figsize=(4,4))   
sns.countplot(df_train.Pclass,hue=df_train.Survived)


# In[23]:


plt.figure(figsize=(4,4))   
sns.histplot(df_train.Age)


# ## Transform Data

# In[24]:


df_train.Sex=df_train.Sex.map({'male':0,'female':1})
df_test.Sex=df_test.Sex.map({'male':0,'female':1})
#df_test.Sex=pd.get_dummies(df_test.Sex)


# In[25]:


df_train.info()


# In[26]:


df_test


# In[27]:


df_train


# ## 5-Creat Model

# In[28]:


X=df_train.drop(['Survived'],axis=1)
y=df_train.Survived


# In[29]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=.8)


# In[30]:


model1=DecisionTreeClassifier()


# In[31]:


model1.fit(X_train,y_train)


# In[32]:


#pre=model1.predict(X_train)


# In[33]:


#accuracy_score(pre,y_train)


# In[34]:


accuracies=[]


# In[35]:


def all(model):
    model.fit(X_train,y_train)
    pre=model.predict(X_train)
    accuracy=accuracy_score(pre,y_train)
    print('Accuracy =',accuracy)
    accuracies.append(accuracy)


# In[36]:


model1=LogisticRegression()
all(model1)


# In[37]:


model2=RandomForestClassifier()
all(model2)


# In[38]:


model3=GradientBoostingClassifier()
all(model3)


# In[39]:


model4=DecisionTreeClassifier()
all(model4)


# In[40]:


model5=KNeighborsClassifier()
all(model5)


# In[41]:


model6=GaussianNB()
all(model6)


# In[42]:


model7=SVC()
all(model7)


# In[43]:


Algorithms=['LogisticRegression','RandomForestClassifier','GradientBoostingClassifier','DecisionTreeClassifier','KNeighborsClassifier','GaussianNB','SVC',]


# In[44]:


new=pd.DataFrame({'Algorithms':Algorithms,'accuracies':accuracies})


# In[81]:


new


# In[82]:


modelx=SVC()
modelx.fit(X_train,y_train)


# In[83]:


predt=modelx.predict(df_test)


# In[84]:


finl=df_test.PassengerId


# In[85]:


new_dataframe=pd.DataFrame({'PassengerId':finl,'Survived':predt})


# In[86]:


new_dataframe.to_csv('submission7.csv',index=False)


# In[ ]:




