# Lisbon-Real-Estate-Analytics
Análise preditiva de imóveis em Lisboa usando Ridge Regression.
# Lisbon Real Estate Analytics 🏡

A data science project to estimate fair real estate prices in Lisbon using **Ridge Regression ($L_2$ Regularization)**.

## 🎯 The Objective
The Lisbon housing market is highly volatile. Traditional "average price per $m^2$" metrics fail to capture the nuances of micro-locations and property conditions. This project aims to isolate the **Intrinsic Value** of a property from market speculation.

## 📐 Mathematical Approach
To handle multicollinearity (correlation between variables like Area and Rooms) and prevent overfitting on a limited dataset, I implemented **Ridge Regression**.

The model minimizes the following Cost Function:

$$J(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2$$

Where:
* The first term is the **Residual Sum of Squares (RSS)**.
* The second term is the **$L_2$ Penalty** (Regularization).
* $\lambda$ is the hyperparameter that controls the penalty strength.

## 🛠 Technologies Used
* **Python** (Pandas, NumPy)
* **Machine Learning** (Scikit-Learn)
* **Visualization** (Plotly Express, Mapbox)
* **Web App** (Streamlit)

## 📊 Results
* **MAE (Mean Absolute Error):** ~€109k
* **RMSE (Root Mean Squared Error):** ~€129k

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

---
© 2024 Portfolio Project
