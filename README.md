# Boston House Price Prediction Using XGBoost

## Overview

This project demonstrates the use of the XGBoost Regressor for predicting house prices in the Boston area. The dataset used is the Boston House Price dataset, which is a commonly used dataset in regression problems.

## Dependencies

To run this project, you need to install the following Python libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
```

## Dataset

The dataset is loaded from `sklearn.datasets`:

```python
house_price_dataset = sklearn.datasets.load_boston()
```

## Data Preprocessing

1. **Loading the dataset into a Pandas DataFrame**:
    ```python
    house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns = house_price_dataset.feature_names)
    ```
   
2. **Adding the target column**:
    ```python
    house_price_dataframe['price'] = house_price_dataset.target
    ```

3. **Checking for missing values**:
    ```python
    house_price_dataframe.isnull().sum()
    ```

4. **Descriptive statistics**:
    ```python
    house_price_dataframe.describe()
    ```

## Exploratory Data Analysis

1. **Correlation Heatmap**:
    ```python
    correlation = house_price_dataframe.corr()
    plt.figure(figsize=(10,10))
    sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
    plt.show()
    ```

## Data Splitting

Splitting the data into training and testing sets:

```python
X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
```

## Model Training

Training the XGBoost Regressor:

```python
model = XGBRegressor()
model.fit(X_train, Y_train)
```

## Evaluation

### Training Data

1. **Predictions on training data**:
    ```python
    training_data_prediction = model.predict(X_train)
    ```

2. **R-squared Error and Mean Absolute Error**:
    ```python
    score_1 = metrics.r2_score(Y_train, training_data_prediction)
    score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)
    print("R squared error : ", score_1)
    print('Mean Absolute Error : ', score_2)
    ```

3. **Visualization**:
    ```python
    plt.scatter(Y_train, training_data_prediction)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual Price vs Predicted Price")
    plt.show()
    ```

### Test Data

1. **Predictions on test data**:
    ```python
    test_data_prediction = model.predict(X_test)
    ```

2. **R-squared Error and Mean Absolute Error**:
    ```python
    score_1 = metrics.r2_score(Y_test, test_data_prediction)
    score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)
    print("R squared error : ", score_1)
    print('Mean Absolute Error : ', score_2)
    ```

## Results

- **R squared error on training data**: 0.9733
- **Mean Absolute Error on training data**: 1.1453
- **R squared error on test data**: 0.9116
- **Mean Absolute Error on test data**: 1.9923

## Conclusion

The XGBoost Regressor model performed well on the Boston House Price dataset with a high R-squared value and a low Mean Absolute Error, indicating that the model can predict house prices accurately.
