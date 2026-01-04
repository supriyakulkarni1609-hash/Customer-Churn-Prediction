# Customer Churn Prediction

## Overview

This project implements a **Customer Churn Prediction system** for a telecommunications company in California. The dataset contains **7,043 customer records**, including demographic information, location, tenure, subscription services, and quarterly status (`joined`, `stayed`, or `churned`).

The goal is to **predict whether a customer will churn, stay, or join** the company based on these features. Predicting churn allows businesses to **proactively retain customers** and reduce revenue loss.


## Technologies Used

* **Python**
* **Pandas, NumPy**
* **Scikit-learn**
* **XGBoost**
* **Matplotlib & Seaborn** for visualization


## Features

* Analyze customer behavior and attributes
* Predict customer churn using machine learning
* Compare multiple models:

  * Random Forest
  * Logistic Regression
  * Decision Tree
  * Gaussian Naive Bayes
  * XGBoost
* Use **GridSearchCV** for hyperparameter tuning
* Evaluate model performance with **cross-validation**
* Generate visualizations of correlations and feature importance


## Modifications & Contributions

The original dataset and scripts were adapted and modified to meet internship requirements:

* Implemented **GridSearchCV** for all models to find the best hyperparameters
* Added **ShuffleSplit cross-validation** for better model evaluation
* Refined the **model comparison loop** to output a summary table of best scores and parameters
* Selected **numeric features** automatically for correlation and modeling
* Tested models on the dataset and generated **accuracy metrics** for each model

> These changes focus on understanding, integration, and correct execution of machine learning models rather than creating new models from scratch.


## Project Structure

```

```


## How to Run

1. **Load the dataset** and define features and target:

```python
import pandas as pd

df = pd.read_csv('data/customer_churn.csv')
X = df.select_dtypes(include='number').drop('Churn', axis=1)
y = df['Churn']
```

2. **Split the dataset**:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

3. **Run model comparison with GridSearchCV**:

```python
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pandas as pd

# Cross-validation strategy
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

# Models and hyperparameters
model_params = {
    'random_forest': {'model': RandomForestClassifier(), 'params': {'n_estimators':[1,5,10]}},
    'logistic_regression': {'model': LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000), 'params': {'C':[1,5,10]}},
    'naive_bayes_gaussian': {'model': GaussianNB(), 'params': {}},
    'decision_tree': {'model': DecisionTreeClassifier(), 'params': {'criterion':['gini','entropy']}},
    'XGB_Classifier': {'model': XGBClassifier(eval_metric='logloss'), 'params': {'base_score':[0.5]}}
}

# Fit models and store results
scores = []
for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=cv, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({'model': model_name, 'best_score': clf.best_score_, 'best_params': clf.best_params_})

df_results = pd.DataFrame(scores)
df_results
```


## Output

* Best hyperparameters and **accuracy scores** for each model
* Visualization of feature importance and correlations
* Summary table for **model selection**

| Model                | Accuracy |
| -------------------- | -------- |
| Random Forest        | 78.11%   |
| Logistic Regression  | 78.28%   |
| Naive Bayes Gaussian | 36.77%   |
| Decision Tree        | 77.29%   |
| XGB Classifier       | 80.86%   |

> XGB Classifier performed the best among all models.


## Business Application

* Identify **high-risk customers** likely to churn
* Take proactive retention actions such as **special offers, personalized communication, or support**
* Improve customer retention and optimize **business revenue**


## Author

**Supriya Kulkarni**
[Intern ID: **SMI82128**]
