import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


df=pd.read_csv('titanic_survival.csv')
# print(df.head)

# print(df.isnull().sum())


#handling missing values

df=df.drop(columns='Cabin',axis=1)

df.fillna({'Age': df['Age'].mean()}, inplace=True)

df['Fare'] = df['Fare'].replace(0, df['Fare'].median())

print(f'total values in embraked are: {df['Embarked'].value_counts()}')
print(df['Embarked'].mode())

df.fillna({'Embarked':df['Embarked'].mode()[0]},inplace=True)

print(df.isnull().sum())

print(df['Survived'].value_counts())


# exploratory analysis

# sns.countplot(x='Survived',data=df)
# plt.show()

# sns.countplot(x='Sex',data=df)
# plt.show()

# sns.countplot(x='Sex', hue='Survived', data=df)
# plt.show()

# sns.countplot(x='Pclass',data=df)
# plt.show()

# sns.countplot(x='Pclass', hue='Survived', data=df)
# plt.show()



## preprocessing

df.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

# print(df.head())

# ##training 

X = df.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = df['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)

model = LogisticRegression()
model.fit(X_train, Y_train)



# accuracy on training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)



# accuracy on test data

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)


# Classification report
print("\nClassification Report:")
print(classification_report(Y_test, X_test_prediction))

# Confusion matrix
conf_matrix = confusion_matrix(Y_test, X_test_prediction)
print("\nConfusion Matrix:")
print(conf_matrix)