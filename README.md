# Credit-Card-Fraud-Detection
This project aims to detect fraudulent credit card transactions using machine learning techniques. Given the highly imbalanced nature of the dataset, we employ various strategies to handle the imbalance and ensure the model's performance is robust. The primary model used in this project is Logistic Regression.

## Introduction

Credit card fraud is a significant issue that can lead to substantial financial losses. Detecting fraudulent transactions is challenging due to the high imbalance in the dataset, with legitimate transactions vastly outnumbering fraudulent ones. This project explores techniques to effectively handle this imbalance and build a robust fraud detection model using Logistic Regression.

## Dataset

The dataset used in this project is publicly available and contains credit card transactions over a two-day period. It includes 492 frauds out of 284,807 transactions, with each transaction having 30 features. Due to the large size of the dataset, it has been split into smaller files for easier handling and processing.

- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Note**: The dataset is split into multiple smaller files for easier processing.

## Data Preprocessing

1. **Data Cleaning**: Handling missing values and outliers.
2. **Feature Scaling**: Standardizing the features.
3. **Handling Imbalance**: Techniques used include:
   - Oversampling using SMOTE (Synthetic Minority Over-sampling Technique)
   - Undersampling the majority class
   - Using class weights in the Logistic Regression model

## Modeling

We implemented Logistic Regression to detect fraudulent transactions. Logistic Regression is a linear model for binary classification that is well-suited for this task due to its simplicity and interpretability.

## Evaluation

To evaluate the model, we used the following metrics:
- **Precision**
- **Recall**
- **F1 Score**

Given the imbalance in the dataset, precision and recall are particularly important metrics.

## Conclusion

Our Logistic Regression model, combined with appropriate techniques to handle data imbalance, provides a robust solution for detecting credit card fraud. While the results are promising, there is room for improvement through further feature engineering and exploring more complex models.

## Future Work

- Implementing more advanced algorithms like XGBoost, or neural networks
- Feature engineering to extract more meaningful features


