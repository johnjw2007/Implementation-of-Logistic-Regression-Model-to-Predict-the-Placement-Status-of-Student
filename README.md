# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: John Wilfred Thomas J W
RegisterNumber:  24013517
*/
```
import pandas as pd
 data=pd.read_csv("Placement_Data.csv")
 print(data.head())
 data1=data.copy()
 data1=data1.drop(["sl_no","salary"],axis=1)
 print(data1.head())
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
 data1["specialisation"]=le.fit_transform(data1["specialisation"])
 data1["status"]=le.fit_transform(data1["status"])
 print(data1)
 x=data1.iloc[:,:-1]
 x
 y=data1["status"]
 y
 from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
 from sklearn.linear_model import LogisticRegression
 lr=LogisticRegression(solver="liblinear")
 lr.fit(x_train,y_train)
 y_pred=lr.predict(x_test)
 print(y_pred)
 from sklearn.metrics import accuracy_score
 accuracy=accuracy_score(y_test,y_pred)
 print(accuracy)
 from sklearn.metrics import confusion_matrix
 confusion=confusion_matrix(y_test,y_pred)
 print(confusion)
 from sklearn.metrics import classification_report
 classification_report1=classification_report(y_test,y_pred)
 print(classification_report1)
 lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
![Screenshot 2024-11-17 221558](https://github.com/user-attachments/assets/0f5804f6-3361-46a9-9ae8-6989b0265f01)
![Screenshot 2024-11-17 221632](https://github.com/user-attachments/assets/85578e93-6072-4728-a078-802f36a5e83b)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
