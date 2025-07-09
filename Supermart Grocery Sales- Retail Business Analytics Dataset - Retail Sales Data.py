#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering
# 
# | Technique                        | What it does                                        | Example                                  |
# | -------------------------------- | --------------------------------------------------- | ---------------------------------------- |
# | **Handling missing values**      | Fill or drop missing data                           | Fill empty "Age" with average age        |
# | **Encoding categorical data**    | Convert text categories to numbers                  | "Male"/"Female" → 1/0                    |
# | **Scaling**                      | Normalize data to a common scale                    | Bringing "Age" and "Salary" to 0-1 range |
# | **Binning**                      | Convert continuous values to groups                 | Age → "Youth", "Adult", "Senior"         |
# | **Datetime features**            | Extract day, month, year from a date                | "2024-06-01" → Month = 6, Day = Saturday |
# | **Creating new features**        | Combine or create columns to boost prediction power | BMI = Weight / (Height²)                 |
# | **Removing irrelevant features** | Drop columns that don’t help or confuse the model   | Dropping "Customer ID" from training     |
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Load Dataset

# In[7]:


data = pd.read_csv("/Users/yaswanth/Downloads/Supermart Grocery Sales - Retail Analytics Dataset.csv")


# In[8]:


data.head()


# # Data Pre-Processing

# In[28]:


# check for NaN Values

data.isnull() # No Nan Values


# In[27]:


# Check for Count(NaN) Values

data.isnull().sum()  # 0 NaN Number


# # In data analysis, a NaN value represents a missing, null, or undefined value in your dataset.
# 
# ## Why NaN appears:
# ## Missing user input in forms
# ## Corrupted or incomplete data
# ## API or sensor failures
# ## Deliberate blank values

# | Task           | Code Example        |
# | -------------- | ------------------- |
# | Check for NaNs | `df.isnull()`       |
# | Count NaNs     | `df.isnull().sum()` |
# | Drop NaNs      | `df.dropna()`       |
# | Fill NaNs      | `df.fillna(value)`  |
# 
# 
# Before training a machine learning model or running analysis, you should handle NaN values — either by:
# 
# Filling them with mean/median/mode
# Dropping the rows/columns
# Or flagging them as a feature
# Let me know your dataset — I can suggest the best method for handling NaN values based on your goal.

# In[21]:


# Drop any rows with missing values

# This line removes all rows with any missing (NaN) values from the DataFrame data and updates it in place 
# (i.e., without needing to assign it to a new variable).

data.dropna(inplace = True)

#inplace=True in Python (especially in Pandas) means: Make the changes directly to the original object
#(DataFrame or Series) — do not return a copy.


# In[37]:


data.drop_duplicates(inplace = True)

data


# # Convert Date Columns to DateTime Format

# In[42]:


data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce', infer_datetime_format='mixed')


# In[43]:


data


# In[44]:


data.head(15)

# NaT is the datetime equivalent of NaN (Not a Number) in Pandas and NumPy.
# It appears when a date/time value is missing, invalid, or can't be parsed.


# In[45]:


data['Order Date'] = pd.to_datetime(data['Order Date'])

data


# In[47]:


data['Order Day'] = data['Order Date'].dt.day
data


# In[50]:


data['Order Month'] = data['Order Date'].dt.day
data


# In[52]:


data['Order Year'] = data['Order Date'].dt.year

data.head(10)


# # Label Encoding for Categorical Variables

# | Method            | Output                             | Use Case                                             |
# | ----------------- | ---------------------------------- | ---------------------------------------------------- |
# | `LabelEncoder()`  | Integer labels (0, 1, 2...)        | When categories have order (e.g., Low, Medium, High) |
# | `OneHotEncoder()` | Binary matrix with one-hot columns | When categories are nominal (no order, like 'City')  |
# 

# In[76]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[77]:


# Initialize the label encoder
le = LabelEncoder()

| Method            | Action                    |
| ----------------- | ------------------------- |
| `fit()`           | Learn patterns/statistics |
| `transform()`     | Apply what was learned    |
| `fit_transform()` | Combine both in one step  |



Simple Definition:
fit_transform() = learn from data (fit) + apply transformation (transform)
It’s used with preprocessing tools like:

StandardScaler()
MinMaxScaler()
OneHotEncoder()
TfidfVectorizer(), etc.

How it works:
1. fit(): Learns from the data

(e.g., mean and standard deviation for scaling)

2. transform(): Applies that learning to convert the data

3. fit_transform(): Does both in one step




# In[95]:


data['Order ID'] = le.fit_transform(data['Order ID'])

data['Customer Name'] = le.fit_transform(data['Customer Name'])

data['Category'] = le.fit_transform(data['Category'])

data.head()


# In[79]:


data['Sub Category'] = le.fit_transform(data['Sub Category'])

data.head()


# In[80]:


data['City'] = le.fit_transform(data['City'])

data.head()


# In[81]:


data['Region'] = le.fit_transform(data['Region'])

data.head()


# In[82]:


data['State'] = le.fit_transform(data['State'])

data.head()


# In[83]:


data['Order Month'] = le.fit_transform(data['Order Month'])

data.head()


# # Exploratory Data Analysis(EDA)

# # 1. Distribution Of Sales by Category
# 
# | Palette Name             | Use                          |
# | ------------------------ | ---------------------------- |
# | `'Set1'`                 | Bright & bold for categories |
# | `'Set2'`                 | Softer for categories        |
# | `'Set3'`                 | Large set of soft colors     |
# | `'Paired'`               | Matching pairs of colors     |
# | `'Accent'`               | Limited but distinct colors  |
# | `'Pastel1'`, `'Pastel2'` | Very light tones             |
# | `'Dark2'`                | Richer, bolder tones         |
# 

# In[86]:


plt.figure(figsize=(10, 6))

sns.boxplot(x='Category', y='Sales', data=data, palette='Set2')
plt.title('Sales Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Sales')
plt.show()


# In[98]:


plt.figure(figsize=(10, 6))

sns.boxplot(x='Sub Category', y='Sales', data=data, palette='Set2')
plt.title('Sales Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Sales')
plt.show()


# In[ ]:


plt.figure(figsize=(10, 6))

sns.boxplot(x='Sub Category', y='Sales', data=data, palette='Set2')
plt.title('Sales Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Sales')
plt.show()


# # 2.sales Trends Over Time

# In[87]:


plt.figure(figsize=(12, 6))
data.groupby('Order Date')['Sales'].sum().plot()
plt.title('Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()


# # 3. Correlation Heatmap

# In[97]:


data


# #### A correlation heatmap is a visual representation of the strength and direction of relationships between numeric variables in your dataset.
# 
# | Correlation (r) | Meaning                               |
# | --------------- | ------------------------------------- |
# | `+1.0`          | Perfect **positive** correlation (↑↑) |
# | `0.0`           | **No** linear correlation             |
# | `-1.0`          | Perfect **negative** correlation (↑↓) |
# 
# 
# What the Colors Mean (for cmap='coolwarm'):
# 🔴 Red: Strong positive correlation
# 🔵 Blue: Strong negative correlation
# ⚪ White/light: Weak or no correlation
# 
# 
# 
# We typically use this heatmap in exploratory data analysis (EDA) to:
# Detect multicollinearity (e.g., two highly correlated features)
# Understand which features influence the target (e.g., Sales vs. Discount, or Profit)
# Decide which features to remove or engineer before modeling"
# 
# Example if your data had Sales/Discount/Profit :
# "In our dataset, I observed that Discount had a negative correlation with Profit, which makes sense — higher discounts often reduce profit.
# On the other hand, Sales and Profit had a positive correlation, indicating higher sales are generally associated with higher profit."
# "This kind of insight helps us improve feature selection and understand business impact early in the data analysis process."

# In[96]:


plt.figure(figsize=(12, 6))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap= 'coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# What to do to increase profits.
# Why This Matters:
# 
# “By understanding these correlations, we can:
# Avoid multicollinearity when building models (e.g., not using both Order Date and Order Year together)
# Identify business levers (e.g., optimizing discount policies to protect profit margins)
# Select or engineer features that are more predictive or business-relevant.”

# # Important Report Top Insights:
# ✔️ Sales & Profit: Moderate positive correlation (+0.61)
# 
# Higher sales tend to increase profit, validating expected business behavior.
# ✔️ Discount & Profit: Moderate negative correlation (–0.55)
# 
# Increasing discounts can significantly reduce profit — a key insight for pricing and promotions.
# ✔️ Order Date & Order Year: Strong positive correlation (+0.97)
# 
# Highlights consistency in date-based features and potential feature redundancy.
# 
# 
# 🔗 This kind of analysis supports better feature selection, multicollinearity checks, and business strategy alignment — all crucial steps before modeling or dashboard building.
# 
# If you're a fellow analyst or aspiring one, don’t skip the correlation heatmap — it’s a goldmine of early-stage insights!

# # 5.Feature Selection and Model Building

# In[100]:


data


# In[101]:


features = data.drop(columns=['Order ID','Customer Name','Order Date','Sales','Order Month','Order Day','Order Year'])


# In[105]:


features


# In[106]:


target = data['Sales']

target

🔄 Step-by-Step Guide:

🎯 Goal:
Use the dataset to build a model that can predict a target variable (e.g., Profit, Sales, etc.) effectively using the right features.

✅ STEP 1: Understand the Business & Objective

Before jumping into modeling:

What are you trying to predict? e.g., Profit, Sales, or Customer Churn
Who will use the model? Marketing, Operations, Pricing Team?
👉 Example Objective:

“Predict Profit based on product category, discounts, and sales.”
✅ STEP 2: Explore & Preprocess the Data (EDA)

Use Exploratory Data Analysis to:

Understand data types, missing values
Plot distributions, trends, outliers
✅ Techniques to Use:

data.info()
data.describe()
data.isnull().sum()
📊 Visualizations:

Histograms, Boxplots (to detect outliers)
Pairplot or Correlation Heatmap (to assess relationships)
✅ STEP 3: Feature Selection (Choosing the right inputs)

Feature selection means keeping variables that are useful for predicting the target and removing noise or redundancy.

🔍 Methods:
1. Correlation Heatmap (for numerical features)

sns.heatmap(data.corr(), annot=True)
Remove highly correlated variables (multicollinearity)
Keep variables strongly correlated with the target
2. Categorical Encoding

Convert Category, Region, etc. into numerical form using:
One-Hot Encoding (pd.get_dummies())
Label Encoding (for tree-based models)
3. Univariate Feature Importance

Use feature importance tools:

SelectKBest
RandomForestClassifier.feature_importances_
XGBoost.feature_importances_
✅ STEP 4: Data Preparation for Modeling

Train-Test Split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
Scale/Normalize (if using algorithms sensitive to feature scale like Logistic Regression, KNN, etc.)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
✅ STEP 5: Model Building

Choose a model based on your problem:

Regression Problem: LinearRegression, RandomForestRegressor, XGBoost, etc.
Classification Problem: LogisticRegression, RandomForestClassifier, SVM, etc.
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
✅ STEP 6: Evaluate the Model

Use metrics depending on problem type:

Regression:
MAE, MSE, RMSE, R²
Classification:
Accuracy, Precision, Recall, F1-Score, ROC AUC
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R2:", r2_score(y_test, y_pred))
✅ STEP 7: Model Tuning & Feature Engineering

Add new features from date (e.g., Weekday, Month)
Try removing or adding new combinations (e.g., Discount x Sales)
Tune hyperparameters using GridSearchCV or RandomizedSearchCV
✅ STEP 8: Deployment or Visualization

Save model using joblib or pickle
Visualize model predictions using plots
Build a dashboard (Power BI / Tableau / Streamlit)
📌 Example Feature Selection in Your Dataset:

If you're predicting Profit, good features might include:

Sales
Discount
Category (One-hot encoded)
Sub-Category
Region
Order Month, Order Year (from Order Date)
# In[114]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features,
target, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # 6: Train a Linear Regression Model

# In[115]:


# Initialize the model
model = LinearRegression()
# Train the model
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)


# # 7: Evaluate the Model

# In[116]:


# Evaluate the model performance using Mean Squared Error (MSE) and R-squared.
# Calculate MSE and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# # Visualize

# In[120]:


# 8: Visualize the Results

# 1. Actual vs Predicted Sales

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Perfect Prediction Line')
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Profit")
plt.legend()
plt.grid(True)
plt.show()

9: Conclusion
● The linear regression model provided a reasonable prediction for sales based
on the features selected.
● The model’s R-squared value indicates a good fit, explaining a significant
portion of the variance in sales.
● Further refinement of the model could involve trying different machine learning
algorithms, such as decision trees or ensemble methods.
# In[125]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Assuming your data is already preprocessed:
# X = features, y = target (e.g., Profit)
X = features
y = target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred = rf_model.predict(X_test)

# Evaluation
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualization: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Ideal Prediction Line')
plt.title('Random Forest: Actual vs Predicted Profit')
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

