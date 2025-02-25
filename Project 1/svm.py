from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('titanic_survival.csv')

# Handling missing values
df = df.drop(columns='Cabin', axis=1)
df.fillna({'Age': df['Age'].mean()}, inplace=True)
df['Fare'] = df['Fare'].replace(0, df['Fare'].median())
df.fillna({'Embarked': df['Embarked'].mode()[0]}, inplace=True)

# Feature Scaling
scaler = StandardScaler()
columns_to_scale = ['Age', 'Fare']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Encoding categorical values
df.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)

# Splitting data
X = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y = df['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2, stratify=Y)

## Using GridSearchCV to Optimize SVM
# param_grid = {
#     'C': [0.1, 1, 10, 100],  # Regularization parameter
#     'gamma': ['scale', 'auto'],  # Kernel coefficient
#     'kernel': ['rbf']  # Non-linear kernel
# }

# grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# grid_search.fit(X_train, Y_train)

# # Best model from Grid Search
# best_svm_model = grid_search.best_estimator_

# print("\n Best Parameters from Grid Search:")
# print(grid_search.best_params_)

model=SVC(C=10,gamma='scale',kernel='rbf')
model.fit(X_train,Y_train)

# Cross-validation accuracy
cv_scores = cross_val_score(model, X_train, Y_train, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
print(f"Cross-validation Accuracy: {cv_scores.mean():.4f}")

# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(Y_test, X_test_prediction))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, X_test_prediction))


# # Function to predict survival for a single input
# def predict_survival():
#     # Taking user input
#     Pclass = int(input("Enter Passenger Class (1, 2, or 3): "))
#     Sex = input("Enter Sex (male/female): ")
#     Age = float(input("Enter Age: "))
#     SibSp = int(input("Enter Number of Siblings/Spouses Aboard: "))
#     Parch = int(input("Enter Number of Parents/Children Aboard: "))
#     Fare = float(input("Enter Fare Amount: "))
#     Embarked = input("Enter Embarked Port (S, C, Q): ")

#     # Encoding categorical values
#     Sex = 0 if Sex.lower() == 'male' else 1
#     Embarked = {'S': 0, 'C': 1, 'Q': 2}.get(Embarked.upper(), 0)

#     # Feature Scaling (Same as training)
#     scaled_values = scaler.transform([[Age, Fare]])[0]  # Scale Age and Fare
#     Age, Fare = scaled_values[0], scaled_values[1]

#     # Creating a NumPy array for prediction
#     user_input = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])

#     # Predict using the trained model
#     prediction = best_svm_model.predict(user_input)[0]

#     # Display result
#     result = "Survived" if prediction == 1 else "Not Survived"
#     print(f"\nPredicted Outcome: {result}")


# # Call the function to test
# predict_survival()
