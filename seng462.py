#Efe Özdemir 202028011
#Nazlı Hilal Özer 201928025

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data_passengerIds = test_data["PassengerId"]
test_data.head()

women = train_data.loc[train_data.Sex  == "female"]["Survived"]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == "male"]["Survived"]
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men)

train_data.info()

#Ticket and cabin values are not neccessary and unwanted
#Because there are a lot of missing cabin values and ticket values are unique to each person
#Name values is unique to each person
columnsToDrop = ['Ticket', 'Cabin', 'Name']
train_data = train_data.drop(columnsToDrop, axis=1)
test_data = test_data.drop(columnsToDrop, axis=1)

train_data.head()

test_data.head()

#PassengerId is an unneccessary colum for our model
train_data = train_data.drop(['PassengerId'], axis=1)
test_data = test_data.drop(['PassengerId'], axis=1)


all_data = pd.concat([train_data, test_data])
all_data.info()


#Filling missing age values
train_data["Age"].fillna(train_data['Age'].median(), inplace=True)
test_data["Age"].fillna(test_data['Age'].median(), inplace=True)

#Filling missing embarked values
train_data['Embarked'].fillna(train_data["Embarked"].dropna().mode()[0], inplace=True)

#Filling missing fare values
test_data["Fare"].fillna(test_data["Fare"].dropna().median(), inplace=True)


train_data.info()


test_data.info()


#Convert string values to numerical values
#It is requires by some ml algorithms

age_mapper = {
    "male": 0,
    "female": 1
}

for df in [test_data, train_data]:
    df["Sex"] = df["Sex"].map(age_mapper).astype(int)
    
train_data.head()


embarked_mapper = {
    'S': 0,
    'C': 1,
    'Q': 2
}

for df in [test_data, train_data]:
    df["Embarked"] = df["Embarked"].map(embarked_mapper).astype(int)

train_data.head()


#Seperate the fare values to bins
#Binning this large scale values is good for our algorithm
#It reduces the distinct value count by categorizing numerical values
#It improves the performance of the models
train_data['FareBin'] = pd.qcut(train_data["Fare"], 4)
train_data[['FareBin', "Survived"]].groupby(['FareBin'], as_index=False).mean().sort_values(by='FareBin', ascending=True)

for df in [train_data, test_data]:
    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
    df.loc[ df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)

train_data.head()


train_data = train_data.drop("FareBin", axis=1)


#Produce new feature with existing ones that will provide better prediction
for df in [test_data, train_data]:
    df["familySize"] = df["SibSp"] + df["Parch"] + 1
    
for df in [test_data, train_data]:
    df["isAlone"] = 0
    df.loc[df["familySize"] == 1, "isAlone"] = 1
    
train_data[["isAlone", "Survived"]].groupby(["isAlone"], as_index = False).mean()


columnsToDrop = ["SibSp", "Parch", "familySize"]
test_data = test_data.drop(columnsToDrop, axis=1)
train_data = train_data.drop(columnsToDrop, axis=1)


train_data.head()


test_data.head()


#Modelling
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

X = train_data.drop("Survived", axis = 1)
y = train_data["Survived"]

logisticRegression = LogisticRegression()
logisticRegression.fit(X, y)

#Confidence on training data
logisticRegression.score(X, y)


kNN = KNeighborsClassifier(n_neighbors = 3)
kNN.fit(X, y)
kNN.score(X, y)


randomForest = RandomForestClassifier(n_estimators=100, max_depth=6)
randomForest.fit(X, y)
randomForest.score(X, y)


predictions = kNN.predict(test_data)


output = pd.DataFrame({'PassengerId': test_data_passengerIds, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")