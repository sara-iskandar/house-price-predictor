# 🏠 House Prices Prediction

Predicting house prices using machine learning models (Linear Regression & Random Forest) on the Kaggle "House Prices: Advanced Regression Techniques" dataset.

---

## 📖 Project Overview

This project demonstrates a complete machine learning workflow:

1. **Data Loading:** Imported the dataset into Python using pandas.  
2. **Exploratory Data Analysis (EDA):** Examined distributions, missing values, and feature correlations.  
3. **Data Cleaning & Preprocessing:**  
   - Filled missing numeric values with the median.  
   - Filled missing categorical values with the mode.  
   - Converted categorical features into numeric using one-hot encoding.  
4. **Model Training & Evaluation:**  
   - Trained **Linear Regression** and **Random Forest Regressor**.  
   - Evaluated performance using **Root Mean Squared Error (RMSE)**.  
5. **Feature Importance Analysis:** Identified the top features influencing house prices.  
6. **Model Saving:** Saved the best-performing model for future use.

---

## 📊 Dataset

The dataset is from Kaggle: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
It contains features like `LotArea`, `OverallQual`, `GrLivArea`, `GarageCars`, etc.  
Target variable: `SalePrice`

---

## ⚙️ Tech Stack

- Python 3.x  
- Pandas & NumPy  
- Matplotlib & Seaborn  
- Scikit-learn  
- Joblib

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/house-price-predictor.git
cd house-price-predictor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Open Jupyter Notebook:
```bash
jupyter notebook
```

5. Run `notebooks/house_price_prediction.ipynb` step by step.

---

## 📈 Results

- **Best Model:** Random Forest Regressor  
- **RMSE:** [Insert your calculated RMSE]  
- **Top Features Influencing Price:** `OverallQual`, `GrLivArea`, `GarageCars`, etc.

---

## 💾 Saved Model

The trained Random Forest model is saved as `src/house_price_model.pkl`.  
Load it in Python to make predictions on new data:
```python
import joblib
model = joblib.load('src/house_price_model.pkl')
prediction = model.predict(new_data)
```

---

## 📌 Key Learning Outcomes

- Data cleaning and handling missing values  
- Exploratory data analysis and visualization  
- Regression modeling using scikit-learn  
- Feature importance interpretation  
- Saving and reusing trained machine learning models

---

## 🔗 Author

Sara Iskandar – www.linkedin.com/in/sara-iskandar