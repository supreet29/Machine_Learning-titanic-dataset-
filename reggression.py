# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('titanic dataset.csv')
real_data = pd.read_csv('titanic dataset.csv')
data.describe()


data['Age'].fillna(data['Age'].median(), inplace=True)

survived_sex = data[data['Survived']==1]['Sex'].value_counts()
dead_sex = data[data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=False, figsize=(15,8))

survived_port_of_entry = data[data['Survived']==1]['Parch'].value_counts()
dead_port_of_entry = data[data['Survived']==0]['Parch'].value_counts()
df = pd.DataFrame([survived_port_of_entry,dead_port_of_entry])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=False, figsize=(15,8))

survived_port_of_entry = data[data['Survived']==1]['SibSp'].value_counts()
dead_port_of_entry = data[data['Survived']==0]['SibSp'].value_counts()
df = pd.DataFrame([survived_port_of_entry,dead_port_of_entry])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=False, figsize=(15,8))

survived_port_of_entry = data[data['Survived']==1]['Embarked'].value_counts()
dead_port_of_entry = data[data['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_port_of_entry,dead_port_of_entry])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=False, figsize=(15,8))

figure = plt.figure(figsize=(15,8))
plt.hist([data[data['Survived']==1]['Age'], data[data['Survived']==0]['Age']], stacked=False, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
figure = plt.figure(figsize=(15,8))
plt.hist([data[data['Survived']==1]['Fare'],data[data['Survived']==0]['Fare']], stacked=False, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()

plt.figure(figsize=(15,8))
ax = plt.subplot() #scatter plot 
ax.scatter(data[data['Survived']==1]['Age'],data[data['Survived']==1]['Fare'],c='green',s=40)
ax.scatter(data[data['Survived']==0]['Age'],data[data['Survived']==0]['Fare'],c='red',s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)


X = data.drop('Survived',1, inplace= True)
X = data.drop('Name',1, inplace= True)
X = data.drop('Ticket',1, inplace= True)
X = data.drop('Cabin',1, inplace= True)
X = data.drop('Embarked',1, inplace= True)
X = data.drop('SibSp',1, inplace= True)
X = data.drop('Parch',1, inplace= True)
X = data.iloc[:,0:12].values
Y = real_data.iloc[:,1 ].values    



# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()

# avoiding the dummy variable trap
X=X[:, 1:]

#ploynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
X = poly.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)



# Feature Scalz
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# fitting the multiple linear regression to the training set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the test set results

y_pred=regressor.predict(X_test)
for i in range (0,len(y_pred)):
    if y_pred[i] > 0.5 :
        y_pred[i]= 1
    else:
        y_pred[i]= 0 
 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
 







