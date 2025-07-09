# Supermart-Grocery-Sales--Retail-Business-Analytics
Discount strategies need review: higher discounts often reduce profit, Data can drive regional sales forecasting and category-level pricing, EDA aids feature engineering and predictive readiness

# 🛒 Supermart Grocery Sales – Retail Analytics Dashboard

## 📌 Overview
This project performs exploratory data analysis (EDA) and builds a machine learning model to **predict profit** using the **Supermart Grocery Sales** dataset. A **Random Forest Regressor** is trained and deployed through a **Streamlit dashboard** for real-time predictions and business insights.

---

## 🎯 Objectives
- Perform EDA to understand trends and patterns in retail sales data
- Analyze correlation between Sales, Profit, Discount, etc.
- Build a regression model to predict Profit
- Deploy model into an interactive dashboard using Streamlit

---

## 🧰 Tech Stack
- **Python**
- **Pandas**, **Seaborn**, **Matplotlib**
- **Scikit-learn**, **RandomForestRegressor**
- **Streamlit**
- **Joblib**
- **Jupyter Notebook**

---

## 📊 Exploratory Data Analysis
Key insights from the EDA:
- Positive correlation between **Sales** and **Profit**
- Negative correlation between **Discount** and **Profit**
- Date-related features were extracted (Month, Year, Day) for better analysis

---

## 🤖 Machine Learning Model
- **Model**: Linear Regression, Random Forest Regressor
- **Target**: Profit
- **Features**: Sales, Discount, Category, Region, Order Date (derived features)
- **Metrics**: RMSE, R²
