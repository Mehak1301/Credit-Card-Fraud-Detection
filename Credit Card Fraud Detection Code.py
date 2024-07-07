#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data= pd.read_csv("creditcard.csv")


# In[3]:


data.head()


# In[7]:


pd.options.display.max_columns= None


# In[8]:


data.head()


# In[9]:


data.tail()


# In[11]:


data.shape


# In[13]:


print("Number of columns: {}".format(data.shape[1]))
print("Number of rows: {}".format(data.shape[1]))


# In[14]:


data.info()


# In[16]:


data.isnull().sum()


# In[17]:


from sklearn.preprocessing import StandardScaler


# In[18]:


sc= StandardScaler()
data['Amount']= sc.fit_transform(pd.DataFrame(data['Amount']))


# In[19]:


data.head()


# In[20]:


data= data.drop(['Time'], axis=1)


# In[21]:


data.head()


# In[22]:


data.duplicated().any()


# In[23]:


data= data.drop_duplicates()


# In[25]:


data.shape


# In[26]:


data['Class'].value_counts()


# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[29]:


sns.countplot(data['Class'])


# In[30]:


x= data.drop('Class', axis=1)
y= data['Class']


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


x_train, x_test, y_train, y_test= train_test_split(x,y, test_size= 0.2, random_state=42)


# In[34]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# In[38]:


classifier={
    "LogisticRegression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}
for name, clf in classifier.items():
    print(f"\n=========={name}============")
    clf.fit(x_train, y_train)
    y_pred= clf.predict(x_test)
    print(f"\n Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"\n Precision: {precision_score(y_test, y_pred)}")
    print(f"\n Recall: {recall_score(y_test, y_pred)}")
    print(f"\n F1 Score: {f1_score(y_test, y_pred)}")
    
    


# In[39]:


# Undersampling


# In[40]:


normal= data[data['Class']==0]
fraud= data[data['Class']== 1]


# In[41]:


normal.shape


# In[43]:


fraud.shape


# In[44]:


normal_sample = normal. sample(n=473)


# In[45]:


normal_sample.shape


# In[47]:


new_data= pd.concat([normal_sample, fraud], ignore_index= True)


# In[48]:


new_data.head()


# In[50]:


new_data['Class'].value_counts()


# In[51]:


x= new_data.drop('Class', axis=1)
y= new_data['Class']


# In[52]:


x_train, x_test, y_train, y_test= train_test_split(x,y, test_size= 0.2, random_state=42)


# In[54]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# In[55]:


classifier={
    "LogisticRegression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}
for name, clf in classifier.items():
    print(f"\n=========={name}============")
    clf.fit(x_train, y_train)
    y_pred= clf.predict(x_test)
    print(f"\n Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"\n Precision: {precision_score(y_test, y_pred)}")
    print(f"\n Recall: {recall_score(y_test, y_pred)}")
    print(f"\n F1 Score: {f1_score(y_test, y_pred)}")


# In[56]:


#Oversampling


# In[58]:


x= data.drop('Class', axis= 1)
y= data['Class']


# In[59]:


x.shape


# In[60]:


y.shape


# In[62]:


get_ipython().system('pip install imbalanced-learn')


# In[63]:


from imblearn.over_sampling import SMOTE


# In[64]:


x_res, y_res= SMOTE().fit_resample(x,y)


# In[65]:


y_res.value_counts()


# In[66]:


classifier={
    "LogisticRegression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}
for name, clf in classifier.items():
    print(f"\n=========={name}============")
    clf.fit(x_train, y_train)
    y_pred= clf.predict(x_test)
    print(f"\n Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"\n Precision: {precision_score(y_test, y_pred)}")
    print(f"\n Recall: {recall_score(y_test, y_pred)}")
    print(f"\n F1 Score: {f1_score(y_test, y_pred)}")


# In[68]:


dtc= DecisionTreeClassifier()
dtc.fit(x_res, y_res)


# In[69]:


import joblib


# In[73]:


joblib.dump(dtc, "credit_card_model.pkl")


# In[74]:


model= joblib.load("credit_card_model.pkl")


# In[76]:


pred= model.predict([[-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62]])


# In[77]:


pred[0]


# In[78]:


if pred[0]== 0:
    print("Normal Transaction")
else:
    print("Fraud Transaction")

