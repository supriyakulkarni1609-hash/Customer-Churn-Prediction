# Customer Churn Prediction

## Overview

This project implements a **Customer Churn Prediction system** for a telecommunications company in California. The dataset contains **7,043 customer records**, including demographics, location, tenure, subscription services, and quarterly status (`joined`, `stayed`, or `churned`).

The goal is to **predict whether a customer will churn, stay, or join** based on these features. Predicting churn allows businesses to **proactively retain customers** and reduce revenue loss.


## Technologies Used

* **Python**
* **Pandas, NumPy**
* **Scikit-learn**
* **XGBoost**
* **Matplotlib & Seaborn**


## Features

* Analyze customer behavior and attributes
* Predict customer churn using multiple machine learning models:

  * Random Forest
  * Logistic Regression
  * Decision Tree
  * Gaussian Naive Bayes
  * XGB Classifier
* Use **GridSearchCV** for hyperparameter tuning
* Evaluate model performance using **ShuffleSplit cross-validation**
* Generate **accuracy metrics** and feature importance visualizations

## Modifications & Contributions

The project was adapted to meet internship requirements:

* Implemented **GridSearchCV** to find best hyperparameters for each model
* Added **ShuffleSplit cross-validation** for robust evaluation
* Focused on **numeric features** for modeling and correlation analysis
* Created **summary tables** of model performance
* Tested models and recorded accuracies for comparison

> These modifications focus on understanding, integration, and execution of ML models rather than building new models from scratch.


## Project Structure

```
Customer-Churn-Prediction/
│
├── Datasets/                         # Dataset folder
│   └── telecom_customer_churn.csv
│
├── Customer_Churn_Prediction.ipynb   # Jupyter Notebook with EDA, model training, and comparison
├── README.md                         # Project README
├──TASK-4 OUTPUTS.pdf                 # Output file
├── requirements.txt                  # Python dependencies
└── pyvenv.cfg                        # Virtual environment configuration
```


## How to Run

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Load dataset** and define features and target:

```python
import pandas as pd

df = pd.read_csv('Datasets/customer_churn.csv')
X = df.select_dtypes(include='number').drop('Churn', axis=1)
y = df['Churn']
```

3. **Split the dataset**:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

4. **Run model comparison with GridSearchCV**:

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

* Accuracy and best hyperparameters for all models
* Feature importance and correlation visualizations

| Model                | Accuracy |
| -------------------- | -------- |
| Random Forest        | 78.11%   |
| Logistic Regression  | 78.28%   |
| Naive Bayes Gaussian | 36.77%   |
| Decision Tree        | 77.29%   |
| XGB Classifier       | 80.86%   |

> **Observation:** XGB Classifier performed the best.



## Business Application

* Identify **high-risk customers** likely to churn
* Take proactive retention actions (**special offers, personalized communication**)
* Improve **customer retention** and optimize revenue


## Author

**Supriya Kulkarni**
[Intern ID: **SMI82128**]
