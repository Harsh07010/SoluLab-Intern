from sklearn.preprocessing import StandardScaler
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier


df=pd.read_csv('titanic_survival.csv')
# print(df.head)

# print(df.isnull().sum())


#handling missing values

df=df.drop(columns='Cabin',axis=1)

df.fillna({'Age': df['Age'].mean()}, inplace=True)

df['Fare'] = df['Fare'].replace(0, df['Fare'].median())

# print(f'total values in embraked are: {df['Embarked'].value_counts()}')
print(df['Embarked'].mode())

df.fillna({'Embarked':df['Embarked'].mode()[0]},inplace=True)


scaler = StandardScaler()
columns_to_scale = ['Age', 'Fare']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
# print(df[columns_to_scale].head())

# print(df.isnull().sum())

# print(df['Survived'].value_counts())


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

# **Hyperparameter Tuning using Grid Search**
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [5, 8, 10],  # Reduce depth
    'min_samples_split': [10, 15, 20],  # Prevent small splits
    'min_samples_leaf': [5, 10],  # Require minimum samples per leaf
    'max_features': ['sqrt', 'log2']  # Try different feature selections
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=2), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, Y_train)

# **Train the Best Model**
best_model = grid_search.best_estimator_

# **Cross-validation accuracy**
cv_scores = cross_val_score(best_model, X_train, Y_train, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
print(f"Cross-validation Accuracy: {cv_scores.mean():.4f}")


# accuracy on training data

X_train_prediction = best_model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)



# accuracy on test data

X_test_prediction = best_model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)


# Classification report
print("\nClassification Report:")
print(classification_report(Y_test, X_test_prediction))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, X_test_prediction))


# def predict_survival():
#     # Taking user input
#     Pclass = int(input("Enter Passenger Class (1, 2, or 3): "))
#     Sex = input("Enter Sex (male/female): ")
#     Age = float(input("Enter Age: "))
#     SibSp = int(input("Enter Number of Siblings/Spouses Aboard: "))
#     Parch = int(input("Enter Number of Parents/Children Aboard: "))
#     Fare = float(input("Enter Fare Amount: "))
#     Embarked = input("Enter Embarked Port (S, C, Q): ")

#     # Encoding categorical values (same as training)
#     Sex = 0 if Sex.lower() == 'male' else 1
#     Embarked = {'S': 0, 'C': 1, 'Q': 2}.get(Embarked.upper(), 0)  # Default to 0 if invalid input

#     # Creating a NumPy array for prediction
#     user_input = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])

#     # Predict using the trained model
#     prediction = model.predict(user_input)[0]

#     # Display result
#     result = "Survived" if prediction == 1 else "Not Survived"
#     print(f"\nPredicted Outcome: {result}")

# # Call the function to test
# predict_survival()