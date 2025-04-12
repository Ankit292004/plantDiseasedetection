import pandas as pd
df=pd.read_csv("/content/lfs.csv")
df
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDRegressor

# Copy of dataset
data = df.copy()

# Encode categorical columns
label_encoders = {}
categorical_columns = ['Disease Type', 'Severity', 'Growth Stage', 'Leaf Color', 'Detection Method']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Features and Target
X = data.drop(['Plant ID', 'Disease Type'], axis=1)
y = data['Disease Type']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

# 1. Data Preprocessing
print("1. Data Preprocessing Done")

# 2. Data Cleaning
print("2. Missing Values:\n", df.isnull().sum())

# 3. Training & Testing using Random Forest
rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
print("3. Random Forest Accuracy:", rf.score(X_test_scaled, y_test))

# 4. EDA - Summary Statistics
print("4. EDA Summary:\n", df.describe())

# 5. Models - Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
print("5. Linear Regression Model Trained")

# 6. Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6,4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title("6. Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 7. PCA Plot
plt.figure(figsize=(6, 4))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis', alpha=0.7)
plt.title("7. PCA - Dimensionality Reduction")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.colorbar()
plt.show()

# 8. Regularization using Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
print("8. Ridge Regression Score:", ridge.score(X_test_scaled, y_test))
# 9. y = mx + b (slope & intercept)
print("9. Linear Coefficients:", lr.coef_)
print("9. Intercept:", lr.intercept_)

# 10. XGBoost Model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train_scaled, y_train)
print("10. XGBoost Accuracy:", xgb.score(X_test_scaled, y_test))

# 11. RMSE, MSE, R^2 for Ridge
ridge_pred = ridge.predict(X_test_scaled)
mse = mean_squared_error(y_test, ridge_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, ridge_pred)
print(f"11. MSE: {mse}, RMSE: {rmse}, R²: {r2}")

# 12. Gradient Descent using SGD Regressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
sgd.fit(X_train_scaled, y_train)
print("12. SGD Regressor Coefficients:", sgd.coef_)

# 13. Cross Validation using Random Forest
cv_scores = cross_val_score(rf, X, y, cv=5)
print("13. Cross Validation Scores:", cv_scores)
print("13. Average CV Score:", np.mean(cv_scores))
# 14. Correlation & Covariance
print("14. Correlation Matrix:\n", df.corr(numeric_only=True))
print("14. Covariance Matrix:\n", df.cov(numeric_only=True))

# 15. EDA using Visualization
# Univariate
plt.figure(figsize=(6,4))
sns.histplot(df['Temperature (°C)'], kde=True)
plt.title("15a. Univariate - Temperature")
plt.show()

# Bivariate
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x='Temperature (°C)', y='Humidity (%)', hue='Severity')
plt.title("15b. Bivariate - Temperature vs Humidity")
plt.show()

# Multivariate using Pairplot
sns.pairplot(df, hue="Severity", diag_kind='kde')
plt.suptitle("15c. Multivariate EDA", y=1.02)
plt.show()
