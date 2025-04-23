# Import libraries and methods/functions
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix 

# datasets 

# telecoms demographics dataset 
telecoms_df = pd.read_csv('telecom_demographics.csv') 
print(telecoms_df.head()) 
print(telecoms_df.shape) 
print(telecoms_df.columns) 

# telecoms usage dataset 
tele_usage_df = pd.read_csv('telecom_usage.csv') 
print(tele_usage_df.head())
print(tele_usage_df.shape)
print(tele_usage_df.columns) 

# merged churn dataset 
churn_df = telecoms_df.merge(tele_usage_df, on = 'customer_id') 
print(churn_df.head())
print(churn_df.shape)
print(churn_df.info())


# Calculate churn rate
# Churn is defined as a customer who has not used the service in the last 30 days
print(churn_df['churn'].value_counts(normalize=True))
churn_rate = churn_df['churn'].value_counts() / len(churn_df)
print(churn_rate) 

# Convert categorical variables into dummy/indicator variables
churn_df = pd.get_dummies(churn_df, columns=['telecom_partner', 'gender', 'state', 'city', 'registration_event'])
print(churn_df.columns)

# Creating Features and Target Variable
features = churn_df.drop(['customer_id', 'churn'], axis=1)
features.head()

# Scaling features
scaler = StandardScaler() 
features_scaled = scaler.fit_transform(features)

# Target variable
target = churn_df['churn'] 
target.head() 

# Splitting the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42) 

# fitting logistic regression model 
logreg = LogisticRegression(random_state=42) 
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test) 

# Confusing matrix
print(confusion_matrix(y_test, logreg_pred)) 

# Classification_report 
print(classification_report(y_test, logreg_pred))