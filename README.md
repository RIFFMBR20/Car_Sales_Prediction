# Car Price Prediction Project

## Overview
This project aims to predict the selling price of used cars using machine learning techniques. The dataset `cars.csv` includes various attributes of cars such as mileage, engine size, max power, and seats, which are used to train a Decision Tree Regressor model. The project involves data preprocessing, visualization, feature engineering, and model training and evaluation.

[Link Dataset](https://www.kaggle.com/datasets/sukhmandeepsinghbrar/car-price-prediction-dataset/data)


## Project Structure
The project is organized into the following sections:

1. **Data Loading and Exploration**
2. **Data Cleaning and Preprocessing**
3. **Data Visualization**
4. **Feature Engineering**
5. **Model Training and Evaluation**

## Data Loading and Exploration
We begin by loading the dataset and performing initial explorations to understand its structure and contents.

```python
import pandas as pd

data = pd.read_csv('cars.csv', delimiter=',')
print(data.head())
print(data.tail())
data.info()
print(data.isna().sum())
```

## Data Cleaning and Preprocessing
The dataset contains some missing values and non-numeric entries in numeric columns. We clean and preprocess the data to ensure it is suitable for modeling.

```numerical_cols = ['mileage(km/ltr/kg)', 'engine', 'max_power', 'seats']
for col in numerical_cols:
    data[col] = data[col].str.extract('(\d+\.\d+|\d+)').astype(float) if col == 'max_power' else data[col]
    data[col].fillna(data[col].mean(), inplace=True)
print(data.isnull().sum())
data.info()
```

## Data Visualization
We visualize the data to gain insights and understand relationships between features.

```import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))
sns.histplot(data.selling_price)
plt.show()

plt.figure(figsize=(20,20))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
```

## Feature Engineering
We prepare the features for model training by encoding categorical variables and scaling numerical features.

```from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Dropping the 'name' column as it's not useful for prediction
data = data.drop('name', 1)
# Separating target variable
y = data['selling_price'].values
data = data.drop(['selling_price'], 1)

# Encoding categorical features
categorical = data.select_dtypes(include=['object']).columns
encoder = OneHotEncoder()
encoded_categorical = encoder.fit_transform(data[categorical]).toarray()

# Scaling numerical features
numerical = data.select_dtypes(include=['int64', 'float64']).columns
scaler = MinMaxScaler()
min_max_scaled_numerical = scaler.fit_transform(data[numerical])

# Concatenating encoded and scaled features
transformed_data = np.concatenate((min_max_scaled_numerical, encoded_categorical), axis=1)
X = transformed_data
```

## Model Training and Evaluation
We split the data into training and testing sets, train a Decision Tree Regressor model, and evaluate its performance.

```from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluating model performance
mae = mean_absolute_error(y_test, predictions)
score = model.score(X_test, predictions)
print(f"Mean Absolute Error: {mae}")
print(f"Model Score: {score}")
```

## Conclusion
This project demonstrates a basic workflow of data preprocessing, feature engineering, and model training for predicting car prices using a Decision Tree Regressor. Future improvements could involve exploring more advanced models and hyperparameter tuning to enhance prediction accuracy.









