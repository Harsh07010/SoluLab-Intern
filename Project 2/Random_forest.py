import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

df=pd.read_csv('HR-Employee-Attrition.csv')
# print(df.head())

df = df.drop(columns=['Unnamed: 35', 'Unnamed: 36'], errors='ignore')
# print(df.head())


# Drop redundant columns
df = df.drop(columns=['EmployeeCount', 'StandardHours', 'EmployeeNumber', 'Over18'])
# print(df.head())

# print(df.isnull().sum())
# df.describe()

# print(df['Attrition'].value_counts())


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

# if df['Attrition'].dtype == 'object':
#     df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# # Select only numerical columns (including Attrition)
# numerical_df = df.select_dtypes(include=[np.number])

# # Compute correlation matrix
# correlation_matrix = numerical_df.corr()

# # Plot heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
# plt.title('Correlation Heatmap')
# plt.show()



#####################################################################


# Separate categorical and numerical features
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Exclude columns that do not require scaling
columns_to_exclude = ['PerformanceRating', 'JobLevel', 'StockOptionLevel', 'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 
                      'JobSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance']
numerical_features_to_scale = [col for col in numerical_features if col not in columns_to_exclude]

# Define binary categorical columns for Label Encoding
binary_columns = ['Attrition', 'Gender', 'OverTime']
label_encoder = LabelEncoder()
for col in binary_columns:
    df[col] = label_encoder.fit_transform(df[col])

# print(len(df.columns))

# Define nominal categorical columns for One-Hot Encoding
ohe_columns = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
df = pd.get_dummies(df, columns=ohe_columns, drop_first=False)

# print(len(df.columns))
# print(len(df.columns))


# Standardize numerical features
scaler = StandardScaler()
df[numerical_features_to_scale] = scaler.fit_transform(df[numerical_features_to_scale])

# print(df.head(10))
# print(len(df.columns))
# df.to_csv('data.csv',index=False)


####################  Apply SMOTE for class imbalance  ####################


X = df.drop(columns=['Attrition'])
y = df['Attrition']

smote = SMOTE(sampling_strategy=0.7, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

##############################################################################


param_grid={
    'n_estimators':[50,100,200],
    'max_depth':[None,10,20,30],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4],
    'bootstrap':[True,False]
}

rf=RandomForestClassifier(random_state=42)
grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,n_jobs=-1,scoring='accuracy',verbose=2)

print("Performing GridSearchCV....")
grid_search.fit(X_train,y_train)

print("\nBest parameters:",grid_search.best_params_)
print("\nBest Score:",grid_search.best_score_)

best_rf=grid_search.best_estimator_
best_rf.fit(X_train,y_train)

y_pred=best_rf.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))