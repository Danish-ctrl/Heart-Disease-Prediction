
# coding: utf-8

# In[72]:


import os


# In[73]:


os.chdir("C:/Users/User/Desktop/Pallavi_20Mar/Pallavi")


# In[74]:


import pandas as pd


# In[75]:


df=pd.read_csv("pallavi.csv")


# In[76]:


df


# In[77]:


dummies_cp=pd.get_dummies(df["cp"])


# In[78]:


dummies_cp


# In[79]:


dummies_cp.columns=["Dummy_cp_1","Dummy_cp_2","Dummy_cp_3","Dummy_cp_4"]





# In[80]:


dummies_cp


# In[81]:


dummies_restecg=pd.get_dummies(df["restecg"])


# In[11]:


dummies_restecg


# In[82]:


dummies_restecg.columns=["Dummy_restecg_0","Dummy_restecg_1","Dummy_restecg_2"]


# In[83]:


dummies_restecg


# In[84]:


dummies_slope=pd.get_dummies(df["slope"])


# In[85]:


dummies_slope


# In[86]:


dummies_slope.columns=["Dummy_slope_1","Dummy_slope_2","Dummy_slope_3"]



# In[87]:


dummies_slope


# In[88]:


dummies_ca=pd.get_dummies(df["ca"])


# In[19]:


dummies_ca


# In[89]:


dummies_ca.columns=["Dummy_ca_0","Dummy_ca_1","Dummy_ca_2","Dummy_ca_3"]


# In[21]:


dummies_ca


# In[90]:


dummies_thal=pd.get_dummies(df["thal"])


# In[23]:


dummies_thal


# In[91]:


dummies_thal.columns=["Dummy_thal_3","Dummy_thal_6","Dummy_thal_7"]


# In[25]:


dummies_thal


# In[92]:


df = pd.concat([df, dummies_ca,dummies_cp,dummies_restecg,dummies_slope,dummies_thal], axis=1)
df


# In[93]:


df=df.drop(["ca","cp","restecg","slope","thal"], axis=1)


# In[94]:


df.info()


# In[95]:


Y=df["num"]


# In[96]:


X= df.drop(["num"], axis=1)


# In[97]:


Y


# In[98]:


X


# In[99]:


X.info()


# In[100]:


#importing the modules
#import numpy n-dimensional array
import numpy as np
#import sklearn python machine learning  modules
import sklearn as sk
#import matplotlib for plotting
import matplotlib.pyplot as plt
#import datasets  and linear_model from sklearn module
from sklearn import datasets, linear_model
#import Polynomial features from sklearn module
from sklearn.preprocessing import PolynomialFeatures
#import train_test_split data classification
from sklearn.model_selection import train_test_split
#import ConfusionMatrix from pandas_ml
from sklearn.metrics import confusion_matrix


# In[101]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=500)


# In[102]:


#Calculating the data
print("Train and test sizes, respectively:", len(X_train), len(Y_train), "|", len(X_test), len(Y_test))


# In[103]:


logistic = linear_model.LogisticRegression(C=1e5)


# In[104]:



from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)


# In[105]:


#Fitting the Algorithm for X_train and y_train
logistic.fit(X_train, Y_train)



# In[106]:


clf.fit(X_train, Y_train)


# In[107]:


#Scoring
print("Using logistic Regresion Algorithm Acuracy Score: ", logistic.score(X_test, Y_test))


# In[108]:


print(" Using Decision Tree Algorithm  Accuracy Score: ", clf.score(X_test, Y_test))


# In[109]:


Y_predicted_logistic = np.array(logistic.predict(X_test))
Y_right = np.array(Y_test)
#print y_test
#The confusion matrix (or error matrix) is one way to summarize the performance of a classifier


# In[110]:


Y_predicted_logistic


# In[111]:


Y_right


# In[112]:


LogReg=confusion_matrix(Y_predicted_logistic,Y_right)


# In[113]:


LogReg.view()


# In[114]:


Y_predicted_Desision = np.array(clf.predict(X_test))


# In[115]:


Y_predicted_Desision


# In[116]:


DecTree=confusion_matrix(Y_predicted_Desision,Y_right)


# In[117]:


DecTree.view()


# In[118]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[119]:


Rand=RandomForestClassifier()


# In[120]:


Rand.fit(X_train,Y_train)


# In[121]:


print("Using RandomForest Algorithm Acuracy Score: ", Rand.score(X_test, Y_test))


# In[133]:


#validation part

Validation=pd.read_csv("validation.csv")
Validation


# In[134]:


dummies_cp=pd.get_dummies(Validation["cp"])
dummies_cp
dummies_cp.columns=["Dummy_cp_1","Dummy_cp_2","Dummy_cp_3","Dummy_cp_4"]
dummies_cp
dummies_restecg=pd.get_dummies(Validation["restecg"])
dummies_restecg
dummies_restecg.columns=["Dummy_restecg_0","Dummy_restecg_1","Dummy_restecg_2"]
dummies_restecg
dummies_slope=pd.get_dummies(Validation["slope"])
dummies_slope
dummies_slope.columns=["Dummy_slope_1","Dummy_slope_2","Dummy_slope_3"]
dummies_slope
dummies_ca=pd.get_dummies(Validation["ca"])
dummies_ca
dummies_ca.columns=["Dummy_ca_0","Dummy_ca_1","Dummy_ca_2","Dummy_ca_3"]
dummies_ca
dummies_thal=pd.get_dummies(Validation["thal"])
dummies_thal
dummies_thal.columns=["Dummy_thal_3","Dummy_thal_6","Dummy_thal_7"]
dummies_thal
Validation1 = pd.concat([Validation, dummies_ca,dummies_cp,dummies_restecg,dummies_slope,dummies_thal], axis=1)
Validation1
Validation1=Validation1.drop(["ca","cp","restecg","slope","thal"], axis=1)
Validation1.info()


# In[135]:


prediction=Rand.predict(Validation1)


# In[136]:


prediction


# In[131]:


print("The final prediction is  :",prediction)


# In[137]:


Validation["Prediction"]=prediction
Validation

