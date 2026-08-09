# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning.

My main goal was not only to train a model and get a high accuracy score. The dataset is extremely imbalanced, so I wanted to understand how the data behaves, which metrics actually matter for this problem, and how different models perform when the cost of missing a fraudulent transaction is taken into account.

## Dataset

The dataset contains 284,807 credit card transactions.

- 284,315 normal transactions
- 492 fraudulent transactions
- Fraud rate: approximately 0.17%

Because fraudulent transactions represent such a small part of the dataset, accuracy alone is not a useful metric for this problem. A model predicting every transaction as normal would still achieve an accuracy of approximately 99.8%, while detecting no fraud at all.

The dataset contains:

- `Time`: Seconds elapsed since the first transaction
- `Amount`: Transaction amount
- `V1` - `V28`: Features transformed using PCA
- `Class`: Target variable (`0 = Normal`, `1 = Fraud`)

## Exploratory Data Analysis

I started by examining the structure of the dataset and the class imbalance.

I compared the transaction amount distributions of normal and fraudulent transactions. Fraudulent transactions were generally concentrated around lower amounts, although there were also some high-value fraudulent transactions.

I also examined the `Time` variable. The distribution showed that fraudulent transactions were more concentrated in some time intervals, but time alone was not enough to clearly separate fraudulent and normal transactions.

After that, I compared the PCA features (`V1` - `V28`) between the two classes. Some variables showed clear differences between normal and fraudulent transactions, especially features such as `V14`, `V4`, `V10`, `V12`, `V17`, and `V3`.

## Data Preparation

The data was divided into three parts:

- 70% training set
- 15% validation set
- 15% test set

Stratified splitting was used to preserve the fraud ratio in each subset.

This resulted in:

| Dataset | Total Transactions | Fraud |
|---|---:|---:|
| Training | 199,364 | 344 |
| Validation | 42,721 | 74 |
| Test | 42,722 | 74 |

For Logistic Regression, `Time` and `Amount` were standardized using `StandardScaler`.

The scaler was fitted only on the training data and then applied to the validation and test sets. This was done to avoid data leakage.

## Models

I tested three different models:

1. Logistic Regression
2. Random Forest
3. HistGradientBoosting

The purpose of trying different models was to compare different approaches rather than assuming that a more complicated model would automatically perform better.

Since the dataset is highly imbalanced, I mainly focused on:

- Precision
- Recall
- PR-AUC
- ROC-AUC

instead of relying on accuracy.

## Model Selection

For this project, I decided that detecting fraudulent transactions was especially important.

I therefore used the following condition when selecting the decision threshold:

> Recall must be at least 85%.

Among the thresholds satisfying this condition, I selected the one that produced the highest precision on the validation set.

For Logistic Regression, the selected threshold was:

`0.8505`

At this threshold, the validation results were:

- Precision: 21.00%
- Recall: 85.14%

Although Random Forest achieved a higher PR-AUC on the validation set, Logistic Regression produced better precision around the required recall level.

This was an important result for me because it showed that there is not always a single "best model." The best choice depends on the metric and the actual objective of the problem.

## Final Test Results

After selecting the model and threshold using the validation data, I evaluated the final model on the test set.

The test set had not been used during model or threshold selection.

### Logistic Regression

- **Precision:** 21.21%
- **Recall:** 85.14%
- **ROC-AUC:** 0.9675
- **PR-AUC:** 0.7904

### Confusion Matrix

| | Predicted Normal | Predicted Fraud |
|---|---:|---:|
| **Actual Normal** | 42,414 | 234 |
| **Actual Fraud** | 11 | 63 |

The model detected 63 of the 74 fraudulent transactions in the test set and missed 11.

It also incorrectly classified 234 normal transactions as fraudulent.

## Interpretation

The final model successfully detected approximately 85% of the fraudulent transactions, but its precision was around 21%.

This means that the model still produces a considerable number of false alarms.

Because of this, I would not consider the current model suitable for automatically blocking transactions on its own.

A more reasonable use would be as a screening system. Transactions identified as suspicious could be sent to an additional verification or investigation step instead of being blocked immediately.

One of the most important things I learned from this project was that a high accuracy score does not necessarily mean that a model is useful.

For an imbalanced problem like fraud detection, understanding false positives, false negatives, precision, recall and the decision threshold is much more important than simply maximizing accuracy.

## Feature Importance

Random Forest was also used to examine which features were useful for separating normal and fraudulent transactions.

The most important features according to the model included:

| Feature | Importance |
|---|---:|
| V14 | 0.1812 |
| V4 | 0.1173 |
| V10 | 0.1083 |
| V12 | 0.0981 |
| V17 | 0.0878 |
| V3 | 0.0629 |
| V16 | 0.0540 |
| V11 | 0.0520 |

These values should not be interpreted as causes of fraud. They only show which features were useful for the Random Forest model when separating the two classes.

## Technologies

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook

## What I Learned

This project helped me understand the complete workflow of a basic machine learning project rather than only the model training step.

Some of the main topics I worked with were:

- Exploratory data analysis
- Class imbalance
- Train / validation / test splitting
- Data leakage
- Feature scaling
- Logistic Regression
- Random Forest
- Gradient Boosting
- Confusion matrices
- Precision and recall
- ROC-AUC and PR-AUC
- Decision thresholds
- Model selection

The biggest takeaway for me was that training a model is only one part of the problem. Choosing the correct evaluation method and understanding the mistakes made by the model can be just as important as the model itself.