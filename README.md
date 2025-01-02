# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: John Wilfred Thomas J W
RegisterNumber:  24013517
*/
```
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
```

## Output:
HEAD
![image](https://github.com/user-attachments/assets/83971512-c9d7-4f69-a99f-f4adb168c97b)

COPY
![image](https://github.com/user-attachments/assets/050d5fb0-530e-44a1-af77-8d3efee5b86c)

FIT TRANSFORM
![image](https://github.com/user-attachments/assets/b6eb5738-6f07-4879-bfc3-4c036a8deb7c)

LOGISTIC REGRESSION
![image](https://github.com/user-attachments/assets/01e1999e-f442-459e-b78a-67125b48ee1a)

ACCURACY SCORE
![image](https://github.com/user-attachments/assets/9090fd8a-b688-49aa-abe9-30b8f3f6a309)

CONFUSION MATRIX
![image](https://github.com/user-attachments/assets/3b8815d4-c8c6-4e9c-8e41-63657ee97bb4)

CLASSIFICATION REPORT
![image](https://github.com/user-attachments/assets/9dca7f91-7d47-4044-88ca-af9627d61d1b)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
