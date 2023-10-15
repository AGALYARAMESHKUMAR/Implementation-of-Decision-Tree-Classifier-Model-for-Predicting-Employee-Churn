# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
# AIM:

To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
# Equipments Required:
    1. Hardware – PCs
    2. Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm

1.Import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics. 10.Find the accuracy of our model and predict the require values.
# Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:Agalya.R 
RegisterNumber: 212222040003

import pandas as pd
data=pd.read_csv("/content/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

*/
```
# Output:
# data.head()
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119394395/070e66e8-72d9-48bc-af11-0f942de75556)

# data.info()
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119394395/7b9ac1c2-439f-44a7-a7d5-7170253c443f)

# isnull() and sum ()
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119394395/ff9e4a13-bf02-4871-ad62-bf90032d5b9f)

# data value counts()
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119394395/ba4f9094-c29e-44cb-bec1-f1f9f77d0842)

# data.head() for salary
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119394395/ab612b4c-21ff-4bd7-bdaf-e9667d800f3d)

# x.head()
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119394395/6d328621-a96c-4d5f-a322-721674a8dae7)

# accuracy value
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119394395/7fbe930f-4da9-40ca-b6f3-99c4b9030128)

# data prediction
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119394395/3463485e-7398-41b7-94ac-3f3bb49caee3)

# Result:
Thus the program to implement the Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
