import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('HR-Employee-Attrition.csv')

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 35', 'Unnamed: 36', 'EmployeeCount', 'StandardHours', 'EmployeeNumber', 'Over18'], errors='ignore')


############## Exploratory Data Analysis     ########################

# Plotting the pie chart for department
Department_counts = df['Department'].value_counts()
plt.figure(figsize=(10, 5))
plt.pie(Department_counts, labels=Department_counts.index, colors=['skyblue', 'lightcoral','lightgreen'], autopct='%1.1f%%')
plt.title('Distribution of Department')
plt.show()

# Attrition rate by department
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Department', hue='Attrition')
plt.title('Attrition Rate by Department')
plt.xlabel('Department')
plt.ylabel('Count')
plt.legend(title='Attrition', loc='upper right')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Plotting the pie chart for JobSatisfaction
JobSatisfaction_counts = df['JobSatisfaction'].value_counts()
plt.figure(figsize=(10, 5))
plt.pie(JobSatisfaction_counts, labels=JobSatisfaction_counts.index, colors=['skyblue', 'lightcoral','orange','green'], autopct='%1.1f%%')
plt.title('Distribution of JobSatisfaction')
plt.show()

# Relationship between JobSatisfaction and Attrition
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Attrition', y='JobSatisfaction')
plt.title('Job Satisfaction vs Attrition')
plt.xlabel('Attrition')
plt.ylabel('Job Satisfaction')
plt.tight_layout()
plt.show()


# Plotting the pie chart for WorkLifeBalance
WorkLifeBalance_counts = df['WorkLifeBalance'].value_counts()
plt.figure(figsize=(10, 5))
plt.pie(WorkLifeBalance_counts, labels=WorkLifeBalance_counts.index, colors=['red', 'lightcoral','orange','green'], autopct='%1.1f%%')
plt.title('Distribution of WorkLifeBalance')
plt.show()

# WorkLifeBalance vs Attrition
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='WorkLifeBalance', y='Attrition')
plt.title('Work Life Balance vs Attrition')
plt.xlabel('Work Life Balance')
plt.ylabel('Attrition Rate')
plt.tight_layout()
plt.show()



# Plotting the pie chart for gender
Gender_counts = df['Gender'].value_counts()
plt.figure(figsize=(10, 5))
plt.pie(Gender_counts, labels=Gender_counts.index, colors=['green', 'pink'], autopct='%1.1f%%')
plt.title('Distribution of Gender')
plt.show()

# Gender vs Attrition
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Gender', y='Attrition')
plt.title('Gender vs Attrition')
plt.xlabel('Gender')
plt.ylabel('Attrition Rate')
plt.tight_layout()
plt.show()




# Disytribution for Age
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Age', color='blue')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Age')
plt.show()



# Distribution of Attrition
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Attrition')
plt.title('Distribution of Attrition')
plt.xlabel('Attrition')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


#Correlation heatmap of numerical features
plt.figure(figsize=(12, 8))
numerical_df = df.select_dtypes(include=[np.number])  # Select only numerical columns
correlation_matrix = numerical_df.corr()  # Compute correlation matrix
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()



#####################################################################



# Separate categorical and numerical features
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Exclude columns that do not require scaling
columns_to_exclude = ['PerformanceRating', 'JobLevel', 'StockOptionLevel', 'Education', 
                      'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 
                      'RelationshipSatisfaction', 'WorkLifeBalance']
numerical_features_to_scale = [col for col in numerical_features if col not in columns_to_exclude]

# Label Encoding for binary categorical columns
binary_columns = ['Attrition', 'Gender', 'OverTime']
label_encoder = LabelEncoder()
for col in binary_columns:
    df[col] = label_encoder.fit_transform(df[col])

# One-Hot Encoding for categorical variables
ohe_columns = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
df = pd.get_dummies(df, columns=ohe_columns, drop_first=False)

# Standardize numerical features
scaler = StandardScaler()
df[numerical_features_to_scale] = scaler.fit_transform(df[numerical_features_to_scale])

# Separate features and target variable
X = df.drop(columns=['Attrition'])
y = df['Attrition']

# Apply SMOTE for handling class imbalance
smote = SMOTE(sampling_strategy=0.7, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Define the CatBoost model with hyperparameter tuning
param_grid = {
    'iterations': [100, 200, 500],
    'depth': [4, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'l2_leaf_reg': [1, 3, 5]
}

catboost = CatBoostClassifier(eval_metric='Accuracy', random_state=42, verbose=0)

grid_search = GridSearchCV(estimator=catboost, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=2)

print('\n Performing GridSearchCV...')
grid_search.fit(X_train, y_train)

print("\n Best Parameters:", grid_search.best_params_)
print("\n Best Score:", grid_search.best_score_)

best_catboost = grid_search.best_estimator_
best_catboost.fit(X_train, y_train)

# Predictions
y_pred = best_catboost.predict(X_test)

# Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\n Confusion Matrix")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Plot Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['NO', 'YES'], yticklabels=['NO', 'YES'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
