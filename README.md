## Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
### AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

### Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
### Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values from dataframe and apply label encoder.

3.Apply decision tree classifier on the dataframe.

4.obtain the value of accuracy and data prediction.

### Program:
```Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Pravin Raj.A
RegisterNumber:  212222240079

import pandas as pd
data=pd.read_csv("Employee.csv")
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
```
### Output:
#### Initial dataset:

![273924854-a89c4f08-8885-4cd3-aef9-3b184a40c0aa](https://github.com/Apravinraj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707879/16fe619c-0f15-4104-94f3-1c08e8156ddd)


#### data info:

![273925087-edab3131-9339-40a5-98d0-f81392e14ff3](https://github.com/Apravinraj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707879/235f1ca2-751b-4b75-a930-52b33fd4a2bf)


#### null values:
![273925391-7467cc3f-1609-464f-a0e4-c463a31e19a4](https://github.com/Apravinraj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707879/483561e0-7a5d-4fbe-9b66-401ba6ec0a02)


#### assignment of x and y values:
![273925657-4f05e9f5-48af-445d-87a9-254cc415c1f5](https://github.com/Apravinraj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707879/e33e8f6e-2184-49d8-a057-aebaeda3e200)

![273925930-75f78e7d-f896-4385-a546-90f4408c5ced](https://github.com/Apravinraj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707879/04be1c00-5855-406c-9309-269dba431962)


#### Converting string literals to numerical values using label encoder:

![273926165-12469a7b-8436-460d-a816-02f6de0c544e](https://github.com/Apravinraj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707879/90fd395f-dcda-4cdc-94c4-740eec80840d)


#### Accuracy:

![273926376-b2ed80e0-0374-416d-8827-e529a59b7c42](https://github.com/Apravinraj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707879/813ee1fd-35a9-4e8c-ab74-bbe4e0da8c42)


#### Prediction:

![273926590-43acb284-2d8a-4119-823c-8a9410dc3196](https://github.com/Apravinraj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707879/e605758a-1f44-46a9-9ad4-97f0b7f594aa)


## Result:
Thus the program to implement the Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
