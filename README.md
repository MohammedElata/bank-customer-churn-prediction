# Bank Customer Churn Prediction

A binary-classification project analysing **10,000 banking customer records** to predict customer churn.

The project implements logistic regression and a linear Support Vector Machine largely from scratch using **NumPy**, providing a practical comparison of model behaviour on an imbalanced classification problem.

## Business Problem

Customer churn can reduce revenue and increase customer-acquisition costs. Predicting which customers are at risk of leaving could help a bank prioritise appropriate retention activity.

The analytical objective is to predict:

* `1`: Customer churned
* `0`: Customer remained with the bank

Because only approximately 20% of customers in the dataset churned, accuracy alone can create a misleading impression of model performance.

## Dataset

The dataset contains **10,000 customer records** and the following variables:

* Credit score
* Country
* Gender
* Age
* Tenure
* Account balance
* Number of products
* Credit-card ownership
* Active-member status
* Estimated salary
* Churn outcome

The customer identifier is removed before modelling and is not used as a predictive feature.

> The dataset will only be included if its original source and redistribution licence can be verified.

## Methodology

The project workflow includes:

1. Data inspection and missing-value checks
2. Removal of customer identifiers
3. Encoding of categorical variables
4. Numerical-feature scaling
5. Exploratory data analysis
6. Reproducible stratified train-test splitting
7. Training-only minority-class oversampling
8. Logistic regression implementation using NumPy
9. Linear SVM implementation using NumPy
10. Evaluation using accuracy, precision, recall, F1-score and confusion matrices

## Models

### Logistic Regression

The logistic-regression classifier was implemented from scratch using:

* Sigmoid activation
* Binary cross-entropy loss
* Gradient descent
* A 0.5 classification threshold

### Linear Support Vector Machine

The SVM was implemented using:

* Linear decision boundaries
* Hinge-loss optimisation
* L2 regularisation
* Gradient-based weight updates

## Verified Original Results

| Model               |   Accuracy | Churn precision | Churn recall |   F1-score |
| ------------------- | ---------: | --------------: | -----------: | ---------: |
| Logistic regression |     66.75% |          30.50% |   **55.13%** | **39.27%** |
| Linear SVM          | **71.75%** |          22.40% |       18.21% |     20.08% |

Although the SVM achieved higher overall accuracy, logistic regression identified a substantially larger proportion of customers who churned.

The SVM’s previously reported figure of approximately 84.7% represents its ability to identify non-churners—specificity—not churn recall.

## Key Findings

* Overall accuracy was influenced heavily by the majority non-churn class.
* Logistic regression provided the stronger churn-detection performance.
* The SVM favoured predicting customers who remained with the bank.
* Oversampling increased minority-class representation but introduced a risk of overfitting.
* Precision, recall and F1-score were more informative than accuracy alone.
* Model thresholds should be selected according to the financial cost of missed churners and unnecessary retention interventions.

## Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* scikit-learn
* Jupyter Notebook
* Google Colab

## Repository Structure

```text
bank-customer-churn-prediction/
├── notebooks/
│   └── bank_customer_churn_prediction.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Limitations

* The dataset is imbalanced, with approximately 20% of customers belonging to the churn class.
* The original implementation used a simple holdout evaluation.
* Random oversampling can lead to overfitting through duplicated minority-class observations.
* The linear SVM’s early-stopping behaviour requires improvement.
* The current models do not include probability calibration or business-cost optimisation.
* Dataset provenance and redistribution permissions must be confirmed before the data can be published.

## Planned Improvements

* Introduce a shuffled, stratified and reproducible train-test split
* Fit preprocessing transformations using training data only
* Compare class weighting with random oversampling
* Add ROC-AUC and precision-recall AUC
* Tune the classification threshold for the retention objective
* Compare the NumPy implementations with scikit-learn baselines
* Evaluate Random Forest and gradient-boosting models
* Add feature-importance and explainability analysis
* Translate false positives and false negatives into business costs

## Responsible Use

Churn predictions should support customer-retention analysis rather than determine how individual customers are treated automatically. Model performance and potential bias should be monitored before any real-world application.

## Author

**Mohammed Elata**
BSc Data Science & Analytics
MSc Data Analytics with Distinction
