# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
```
## Algorithm
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results. 
```  
```
## Program:
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: UDHAYA PRAKASH V
RegisterNumber: 24901131
*/
import pandas as pd
data = pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis = 1)
data.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
## HEAD
![Screenshot 2024-11-21 105348](https://github.com/user-attachments/assets/7bb7c013-1704-4a6a-bfb0-4c963a5cacd2)
## COPY
![Screenshot 2024-11-21 105751](https://github.com/user-attachments/assets/ea3b2ed7-b0e5-4770-8cf3-6598eee124df)
## FIT TRANSFORM
![Screenshot 2024-11-21 105932](https://github.com/user-attachments/assets/5f9c328d-6515-4f40-873f-213a168690e8)
## LOGISTIC REGRESSION
![Screenshot 2024-11-21 221802](https://github.com/user-attachments/assets/2ba73752-e46e-4ac0-a6b2-fce0f25cc31a)
## ACCURACY SCORE
![Screenshot 2024-11-21 222003](https://github.com/user-attachments/assets/4d412632-b5c3-4e0e-b3ed-d25864ab09ab)
## CONFUSION MATRIX
![Screenshot 2024-11-21 222132](https://github.com/user-attachments/assets/7a33a1f8-ed22-4d18-9aec-89006ede29a0)
## CLASSIFICATION REPORT
![Screenshot 2024-11-21 222249](https://github.com/user-attachments/assets/d03e44a9-fa19-42fe-ab1f-449f268d939c)
## PREDICTION
![Screenshot 2024-11-21 222346](https://github.com/user-attachments/assets/306dca53-8c97-4d40-ad45-3b204335e0a3)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
