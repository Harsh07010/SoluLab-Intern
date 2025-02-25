import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load dataset
original_df = pd.read_csv('HR-Employee-Attrition.csv')
df = original_df.copy()

# Drop unnecessary columns
df.drop(columns=['Unnamed: 35', 'Unnamed: 36'], errors='ignore', inplace=True)
df.drop(columns=['EmployeeCount', 'StandardHours', 'EmployeeNumber', 'Over18'], errors='ignore', inplace=True)

# Identify categorical and numerical features
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Exclude columns that do not require scaling
columns_to_exclude = ['PerformanceRating', 'JobLevel', 'StockOptionLevel', 'Education', 'EnvironmentSatisfaction',
                      'JobInvolvement', 'JobSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance']
numerical_features_to_scale = [col for col in numerical_features if col not in columns_to_exclude]

# Label encode binary categorical columns
binary_columns = ['Attrition', 'Gender', 'OverTime']
label_encoders = {}
for col in binary_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoder for later use

# One-hot encode nominal categorical columns
ohe_columns = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
df = pd.get_dummies(df, columns=ohe_columns, drop_first=False)

# Save feature names for later use
feature_names = df.drop(columns=['Attrition']).columns.tolist()

# Standardize numerical features
scaler = StandardScaler()
df[numerical_features_to_scale] = scaler.fit_transform(df[numerical_features_to_scale])

# Define X and y
X = df.drop(columns=['Attrition'])
y = df['Attrition']

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy=0.7, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)


xgb=XGBClassifier(eval_metric='mlogloss', n_estimators=200,max_depth=5,learning_rate=0.1,subsample=0.7,colsample_bytree=1.0,gamma=0)
xgb.fit(X_train, y_train)

# Streamlit UI
st.title('Employee Attrition Prediction')
st.write("Enter details manually or upload a CSV file for prediction.")

# File upload option
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    user_df.drop(columns=['Unnamed: 35', 'Unnamed: 36'], errors='ignore', inplace=True)
    user_df.drop(columns=['EmployeeCount', 'StandardHours', 'EmployeeNumber', 'Over18'], errors='ignore', inplace=True)
    
    # Apply Label Encoding to binary columns
    for col in binary_columns:
        if col in user_df.columns and col in label_encoders:
            user_df[col] = user_df[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else 0)
    
    # Apply One-Hot Encoding and align with training features
    user_df = pd.get_dummies(user_df, columns=ohe_columns, drop_first=False)
    
    # Add missing columns from training set
    for col in feature_names:
        if col not in user_df.columns:
            user_df[col] = 0  # Fill missing columns with 0
    
    # Ensure column order matches training data
    user_df = user_df[feature_names]
    
    # Standardize numerical features
    user_df[numerical_features_to_scale] = scaler.transform(user_df[numerical_features_to_scale])
    
    # Predict Attrition
    user_df['Attrition'] = xgb.predict(user_df)
    
    # Reverse standardization for numerical columns
    user_df[numerical_features_to_scale] = scaler.inverse_transform(user_df[numerical_features_to_scale])
    
    # Convert encoded categorical columns back to original values
    for col in binary_columns:
        if col in user_df.columns and col in label_encoders:
            user_df[col] = label_encoders[col].inverse_transform(user_df[col])

    # Reverse one-hot encoding
    for col in ohe_columns:
        related_cols = [c for c in user_df.columns if c.startswith(col + '_')]
        if related_cols:
            user_df[col] = user_df[related_cols].idxmax(axis=1).str[len(col) + 1:]
            user_df.drop(columns=related_cols, inplace=True)

    # Ensure column format matches the original
    predicted_record = user_df.iloc[0].to_dict()
    formatted_row = {col: predicted_record[col] for col in original_df.columns if col in predicted_record}
    predicted_df = pd.DataFrame([formatted_row])

    # Append the predicted record to the original dataset
    updated_df = pd.concat([original_df, predicted_df], ignore_index=True)
    new_csv_filename = 'updated_HR-Employee-Attrition.csv'
    updated_df.to_csv(new_csv_filename, index=False)

    st.write("### Prediction Completed")
    st.write(predicted_df.head())

    # Download updated CSV
    output_csv = updated_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Updated CSV", output_csv, new_csv_filename, "text/csv")

else:
    user_data = {}
    for column in X.columns:
        user_data[column] = st.number_input(f"Enter value for {column}", value=0, step=1)
    user_df = pd.DataFrame([user_data])
    
    # Standardize numerical features
    user_df[numerical_features_to_scale] = scaler.transform(user_df[numerical_features_to_scale])
    
    if st.button('Predict'):
        user_prediction = xgb.predict(user_df)
        user_prediction_prob = xgb.predict_proba(user_df)[:, 1]
        
        st.write("### Prediction Result")
        st.write("Predicted Attrition:", "Yes" if user_prediction[0] == 1 else "No")
        st.write("Prediction Probability:", user_prediction_prob[0])
        
        # Store prediction in Attrition column
        user_df['Attrition'] = user_prediction
        
        # Reverse standardization
        user_df[numerical_features_to_scale] = scaler.inverse_transform(user_df[numerical_features_to_scale])
        
        # Convert back categorical values
        for col in binary_columns:
            if col in user_df.columns and col in label_encoders:
                user_df[col] = label_encoders[col].inverse_transform(user_df[col])

        # Reverse one-hot encoding
        for col in ohe_columns:
            related_cols = [c for c in user_df.columns if c.startswith(col + '_')]
            if related_cols:
                user_df[col] = user_df[related_cols].idxmax(axis=1).str[len(col) + 1:]
                user_df.drop(columns=related_cols, inplace=True)

        # Save updated dataset
        updated_df = pd.concat([original_df, user_df], ignore_index=True)
        new_csv_filename = 'updated_HR-Employee-Attrition.csv'
        updated_df.to_csv(new_csv_filename, index=False)
        
        output_csv = updated_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Updated CSV", output_csv, new_csv_filename, "text/csv")
