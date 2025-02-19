#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as stats
import matplotlib.dates as mdates
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn import metrics

#loading dataset
# Loading Data
data = pd.read_csv("C:/Users/HP/PREDICTION/DROUGHT.csv")
print(data)

data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data.head(5)

data.shape
#creating SPI
def calculate_spi(precipitation, scale=3):
    """
    Calculate the Standardized Precipitation Index (SPI) for a given precipitation time series.

    :param precipitation: Pandas Series of monthly precipitation values
    :param scale: SPI time scale (default is 3-month SPI)
    :return: Pandas Series of SPI values
    """
    precipitation = precipitation.replace(0, 0.01)  # Avoid zero values for Gamma distribution
    rolling_precip = precipitation.rolling(scale).sum().dropna()  # Compute rolling sum

    # Fit Gamma distribution
    shape, loc, scale_param = stats.gamma.fit(rolling_precip, floc=0)
    cdf_values = stats.gamma.cdf(rolling_precip, shape, loc=loc, scale=scale_param)  # Compute CDF

    # Convert CDF to SPI (Z-score)
    spi_values = stats.norm.ppf(cdf_values)

    return pd.Series(spi_values, index=rolling_precip.index)  # Return SPI values with correct index
# Compute SPI and integrate it back
data["SPI"] = calculate_spi(data["Precipitation"])
# Reset index and save the updated dataset
data.reset_index(inplace=True)
data.to_csv("updated_dataset_with_spi.csv", index=False)

# Print first few rows for verification
print(data.head())

# Classify drought severity based on SPI thresholds
data['Drought_Severity'] = pd.cut(
    data['SPI'], 
    bins=[-float('inf'), -2, -1.5, -1, 0, 1, 1.5, 2, float('inf')],
    labels=['Extreme Drought', 'Severe Drought', 'Moderate Drought', 'Mild Drought',
            'Normal', 'Mild Wet', 'Moderate Wet', 'Severe Wet']
)

print(data)

# Checking for missing values using isnull()
print(data.isnull().values.any()) 

data = data.dropna()

# Normalize continuous variables
scaler = StandardScaler()
data[['Wind_Speed', 'Dew_Point', 'Temperature', 'Precipitation', 'Pressure']] 
scaler.fit_transform(data[['Wind_Speed', 'Temperature', 'Dew_Point', 'Precipitation', 'Pressure']])

print(data)

le = LabelEncoder()
encoder_1 = LabelEncoder()
encoder_1.fit(data["Drought_Severity"])
data_drought_encoded = encoder_1.transform(data["Drought_Severity"])
data["Drought_Severity"] = data_drought_encoded 

# Compute the correlation matrix
correlation_matrix = data.corr()
# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

print(correlation_matrix['Drought_Severity'].sort_values(ascending=False))

# Plot Heatmap
f, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, vmax=1, annot=True, square=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix", fontsize=12)
plt.show()

#Drop unnecessary data

data.drop(['SPI', 'Dew_Point', 'NDVI', 'Wind_Speed', 'Date'],axis=1,inplace = True)


# Compute the correlation matrix
correlation_matrix = data.corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

annot = True
f, ax =plt.subplots(figsize=(8,6))
sns.heatmap(correlation_matrix, vmax=1, annot=annot, square=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix", fontsize=12)


target = 'Drought_Severity'

X = data.drop(columns=['Drought_Severity'])
y = data['Drought_Severity']


from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)  # Number of splits

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

from imblearn.over_sampling import SMOTE
from collections import Counter

# Check class distribution before SMOTE
print("Before SMOTE:", Counter(y_train))

# Apply SMOTE for multi-class balancing
smote = SMOTE(sampling_strategy='auto', random_state=42)  
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
# Check class distribution after SMOTE
print("After SMOTE:", Counter(y_train_balanced))

# Initialize base models
base_models = [
    ('Random Forest', RandomForestClassifier(random_state=42, n_estimators=200, max_depth=5, min_samples_split=5, min_samples_leaf=2)),
    ('SVM', SVC(kernel='poly', C=1, probability=True)),
    ('XGBoost', XGBClassifier(random_state=42, n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, gamma=0.2, reg_lambda=10, reg_alpha=5)),
    ('LightGBM', LGBMClassifier(random_state=42, n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=10, reg_alpha=5))  
]

# Dictionary to store evaluation results
results = {}
# Evaluate each base model individually
for name, model in base_models:
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)
    
    # For AUC, obtain probability estimates for each class
    y_proba = model.predict_proba(X_test)
    
    # Compute evaluation metrics for multi-class using 'weighted' average where applicable.
    acc   = accuracy_score(y_test, y_pred)
    rec   = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    prec  = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    f1    = f1_score(y_test, y_pred, average='weighted')
    
    
    results[name] = {'accuracy': acc, 'recall': rec, 'precision': prec, 'f1': f1}

    # Create and train a stacking ensemble model with Logistic Regression as the final estimator.
from sklearn.linear_model import LogisticRegression
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000),
    cv=5,
    n_jobs=-1
)

stacking_model.fit(X_train_balanced, y_train_balanced)

# Compute evaluation metrics for the stacking model
from sklearn.metrics import classification_report, confusion_matrix
acc   = accuracy_score(y_test, y_pred)
rec   = recall_score(y_test, y_pred, average='weighted', zero_division=1)
prec  = precision_score(y_test, y_pred, average='weighted')
f1    = f1_score(y_test, y_pred, average='weighted')

results['stacking'] = {
    'accuracy': acc, 
    'recall': rec, 
    'precision': prec, 
    'f1': f1
}
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Print the evaluation metrics for each model
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()

import pickle
# save the drought classification model as a pickle file
stacking_model_pkl_file = "Drought_Severity_Model.pkl"  

with open(stacking_model_pkl_file, 'wb') as file:  
    pickle.dump(stacking_model, file)

 #load model from pickle file
with open(stacking_model_pkl_file, 'rb') as file:  
    stacking_model = pickle.load(file)

# evaluate model 
y_pred = stacking_model.predict(X_test)

# check results
print(classification_report(y_test, y_pred)) 


import streamlit as st
import pickle
import numpy as np



#streamlit run Drought.py-RIGHT COMMAND

# Define the drought severity classes (update based on your dataset)
bins=[-float('inf'), -2, -1.5, -1, 0, 1, 1.5, 2, float('inf')], 
labels=['Extremely Dry', 'Severe Drought', 'Moderate Drought', 'Mild Drought', 
            'Normal', 'Moderately Wet', 'Severely Wet', 'Extremely Wet']


# Title for the app
st.title("Drought Prediction in Kilifi County 🌦️")

# Upload CSV section
uploaded_file = st.file_uploader("Upload your CSV file with environmental data", type=["csv"])

# Display sample input format if needed
if st.button("Show Sample Input Format"):
    st.write(pd.DataFrame({
        "Temperature": [30.5],
        "Precipitation": [120.5],
        "Pressure": [29.0],
        "EVI": [1.0],
        "Drought_Severity": [0]
    }))
 # Process the uploaded file
if uploaded_file is not None:
    # Read the CSV
    user_data = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded Data Preview:")
    st.dataframe(user_data.head())

 # Ensure columns match the model input
    expected_features = ['Temperature', 'Precipitation', 'EVI', 'Pressure', 'Drought_Severity']
    if not all(col in user_data.columns for col in expected_features):
        st.error(f"❌ The uploaded file must contain these columns: {expected_features}")
    else:
        # Make predictions
        predictions = model.predict(user_data[expected_features])

        # Map predictions to drought severity
        user_data['Drought_Severity'] = [labels[pred] for pred in predictions]

        st.success("✅ Predictions Completed!")
        st.dataframe(user_data)

        # Download the predictions
        csv_output = user_data.to_csv(index=False).encode('utf-8')

        st.download_button("📥 Download Predictions", csv_output, "drought_predictions.csv", "text/csv")

st.info("Note: Ensure your dataset is in CSV format and matches the expected columns.")      








