# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:46:37 2020

@author: Maneshwar
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

test_df= pd.read_csv('test.csv')
train_df= pd.read_csv('train.csv')

train_df.info()
train_df.describe()
train_df.head(8)

total= train_df.isnull().sum().sort_values(ascending= False)
percent_1= train_df.isnull().sum()/train_df.isnull().count()*100
percent_2= (round(percent_1, 1)).sort_values(ascending= False)
missing_data= pd.concat([total, percent_2], axis= 1, keys= ['Total', '%'])

train_df.columns.values

sns.barplot(x= 'Pclass', y= 'Survived', data= train_df)

train_df= train_df.drop(['PassengerId'], axis= 1)

data= [train_df, test_df]
for dataset in data:
    mean= train_df['Age'].mean()
    std= test_df['Age'].std()
    is_null= dataset['Age'].isnull().sum()
    rand_age= np.random.randint(mean-std, mean+std, size= is_null)
    age_slice= dataset['Age'].copy()
    age_slice[np.isnan(age_slice)]= rand_age
    dataset['Age']= age_slice
    dataset['Age']= train_df['Age'].astype(int)
    
train_df['Age'].isnull().sum()
train_df['Embarked'].describe()

common_value= 'S'
data= [train_df, test_df]
for dataset in data:
    dataset['Embarked']= dataset['Embarked'].fillna(common_value)
    
train_df.info()
test_df.isnull().sum()

data= [train_df, test_df]
for dataset in data:
    dataset['Fare']= dataset['Fare'].fillna(0)
    dataset['Fare']= dataset['Fare'].astype(int)
    
data= [train_df, test_df]
titles= {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}
for dataset in data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr','Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title']= dataset['Title'].map(titles)
    dataset['Title']= dataset['Title'].fillna(0)
    
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)

data= [train_df, test_df]
genders= {'male':0, 'female':1}
for dataset in data:
    dataset['Sex']= dataset['Sex'].map(genders)
    
train_df= train_df.drop(['Ticket'], axis= 1)
test_df= test_df.drop(['Ticket'], axis= 1)

ports = {"S": 0, "C": 1, "Q": 2}
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)
    
data= [train_df, test_df]

for dataset in data:
    dataset['Age']= dataset['Age'].astype(int)
    dataset.loc[dataset['Age']<=11, 'Age']= 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
        
train_df['Age'].value_counts()
train_df.head(10)

pd.qcut(train_df['Fare'], q=5)

data= [train_df, test_df]
for dataset in data:
    dataset.loc[dataset['Fare']<=7, 'Fare']= 0
    dataset.loc[(dataset['Fare']>7) & (dataset['Fare']<=10), 'Fare']= 1
    dataset.loc[(dataset['Fare']>10) & (dataset['Fare']<=21), 'Fare']= 2
    dataset.loc[(dataset['Fare']>21) & (dataset['Fare']<=39), 'Fare']= 3
    dataset.loc[(dataset['Fare']>39) & (dataset['Fare']<=512), 'Fare']= 4
    dataset.loc[dataset['Fare']>512, 'Fare']= 5
    dataset['Fare']= dataset['Fare'].astype(int)
    
test_df.isnull().sum() 

data= [train_df, test_df]
for dataset in data:
    dataset['Relatives']= dataset['SibSp']+ dataset['Parch']
    dataset.loc[dataset['Relatives']>0, 'Not_alone']= 1
    dataset.loc[dataset['Relatives']==0, 'Not_alone']= 0
    dataset['Not_alone']= dataset['Not_alone'].astype(int)

train_df['Not_alone'].value_counts()

data= [train_df, test_df]
for dataset in data:
    dataset['Fare_Per_Person']= dataset['Fare']/(dataset['Relatives']+1)
    dataset['Fare_Per_Person']= dataset['Fare_Per_Person'].astype(int)
    
train_df.head(10)

train_df= train_df.drop(['Cabin'], axis= 1)
test_df= test_df.drop(['Cabin'], axis= 1)

train_df.isnull().sum()

X_train= train_df.drop('Survived', axis= 1)
Y_train= train_df['Survived']
X_test= test_df.drop('PassengerId', axis= 1).copy()

from sklearn.linear_model import SGDClassifier
sgd= linear_model.SGDClassifier(max_iter= 5, tol= None)
sgd.fit(X_train, Y_train)
Y_pred= sgd.predict(X_test)
sgd.score(X_train, Y_train)

random_forest= RandomForestClassifier(n_estimators= 100)
random_forest.fit(X_train, Y_train)
Y_prediction= random_forest.predict(X_test)
random_forest.score(X_train, Y_train)

output= pd.DataFrame({'PassengerId':test_df.PassengerId, 'Survived':Y_prediction})
output.to_csv('RandomForest_Titanic.csv', index= False)

logreg= LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred= logreg.predict(X_test)
logreg.score(X_train, Y_train)

linear_svc= LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred= linear_svc.predict(X_test)
linear_svc.score(X_train, Y_train)

output= pd.DataFrame({'PassengerId':test_df.PassengerId, 'Survived':Y_pred})
output.to_csv('LogisticRegressios_Titanic.csv', index= False)

knn= KNeighborsClassifier(n_neighbors= 3)
knn.fit(X_train, Y_train)
Y_pred= knn.predict(X_test)
knn.score(X_train, Y_train)

output= pd.DataFrame({'PassengerId':test_df.PassengerId, 'Survived':Y_PREDN})
output.to_csv('RDitanic.csv', index= False)

decision_tree= DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred= decision_tree.predict(X_test)
decision_tree.score(X_train, Y_train)


from sklearn.model_selection import cross_val_score
rf= RandomForestClassifier(n_estimators= 100)
scores= cross_val_score(rf, X_train, Y_train, cv= 10, scoring= 'accuracy')

scores.mean()
scores.std()

importances= pd.DataFrame({'Feature':X_train.columns, 'Importance':np.round(random_forest.feature_importances_, 3)})
importances= importances.sort_values('Importance', ascending= False).set_index('Feature')
importances.plot.bar()

train_df= train_df.drop('Not_alone', axis= 1)
test_df= test_df.drop('Not_alone', axis= 1)

train_df= train_df.drop('Parch',axis= 1)
test_df= test_df.drop('Parch', axis=1)

from sklearn.model_selection import RandomizedSearchCV
n_estimators= [int(x) for x in np.linspace(100, 2000, num= 20)]
max_features= ['auto', 'sqrt']
max_depth= [int(x) for x in np.linspace(10, 110, num= 11)]
max_depth.append(None)
min_samples_split= [2, 4, 10, 12, 16, 18, 25, 30, 35]
min_samples_leaf= [1, 5, 10, 15, 20, 25]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'criterion': ['gini', 'entropy']}

rf= RandomForestClassifier(oob_score= True, n_jobs= -1, random_state= 1)
rf_random= RandomizedSearchCV(estimator= rf, param_distributions= random_grid, n_iter= 100, cv= 3, verbose= 2, random_state= 1, n_jobs= -1)
rf_random.fit(X_train, Y_train)
Y_prediction= rf_random.predict(X_test)


random_forest= RandomForestClassifier(n_estimators= 100, oob_score= True)
random_forest.fit(X_train, Y_train)
Y_prediction= random_forest.predict(X_test)

param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1,2,3,4,5,25, 50, 70], "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [100,120, 140,160,200,1500]} 
from sklearn.model_selection import GridSearchCV, cross_val_score
rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)
clf= GridSearchCV(estimator= rf, param_grid= param_grid, n_jobs= -1)
clf.fit(X_train, Y_train)

clf.best_params_

random_forest= RandomForestClassifier(criterion= 'entropy', min_samples_leaf= 3, min_samples_split=35,n_estimators=120,max_features= 'auto', oob_score= True, random_state= 1, n_jobs= -1)
random_forest.fit(X_train, Y_train)
Y_PREDN= random_forest.predict(X_test)


random_forest.score(X_train, Y_train)

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions= cross_val_predict(random_forest, X_train, Y_train, cv= 3)
confusion_matrix(Y_train, predictions)

from sklearn.metrics import f1_score
f1_score(Y_train, predictions)

from sklearn.metrics import precision_recall_curve
y_scores= random_forest.predict_proba(X_train)
y_scores= y_scores[:, 1]
















