# Stroke Prediction
# Overview
The project inspects Stroke Prediction Dataset from Kaggle. 

The primary objectives are to clean the data, perform exploratory data analysis, statistical analysis, and apply various machine learning models for target variable Stroke prediction. Stroke prediction model must be calibrated for saving patient lives, as well as saving hospital time. 

## Dataset
Dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

## Python Libraries

This analysis was conducted using Python 3.11. The following packages were utilized:

- Imbalanced-learn: 0.0
- Eli5 0.13.0
- Lime: 0.2.0.1
- Matplotlib: 3.7.1
- NumPy: 1.25.0
- Pandas: 1.5.3
- SciPy: 1.11.1
- Scikit-learn: 0.0.post5
- Seaborn: 0.12.2
- Statsmodels: 0.14.0
- TextBlob: 0.17.1
- Unidecode: 1.3.7
- XGBoost: 2.0.3


## Findings

* Exploratory Data Analysis (EDA): Dataset has 5110 observations and 12 features. Data is collected for children and adults, internet research showed that various features (Body Mass Index, Vitamin D) have different levels of safety for children and adults (children's parameters having a hyperbole dependency on the age). This work if focusing only on adults,   leaving a dataset of 4248 features. Body Mass Index and Glucose Levels had extremely high values (92 and 272 mg/dL, respectfully). Internet search showed those values being allowed, yet dangerously high. Target demographic: Married, Female, Works for Private companies, Healthy, Middle aged.
* Correlation: No 2 features are strong correlated, no feature pair with linear relationship.
* Feature Engineering: BMI and Smoking status was imputed using K-Nearest Neighbors (KNN) Imputation
* Statistical Testing:  Examining data showed that never married people don't have strokes or heart disease, but statistical testing revealed that Stroke and Heart disease happening depends on Age groups. Marriage also depends on age group.
* Models: Various machine learning models (KNN, Support Vector Machines, Decision Tree, Random Forest, Naive Bayers, Gradient Boosting) were tested, as well as (Adaptive Boosting) Classifiers. Class imbalance was addressed by using Synthetic Minority Over-sampling Technique (SMOTE).
* Best model: Best model was 'Naive Bayers' with F1 score of 0.19.

## Suggestions for Medical Institutions

* Pay attention to people in the older groups as 95% confidence interval for stroke events are prevalent for people in their 60s.
* BMI > 33, smoking, heart disease and hypertention are an increased factors for stroke.
* Marital status is not an indicator for stroke.
* Missing BMI paid a big role in Models performance. Missing BMI signals higher risk for stroke. Gathering data BMI could have had only 2 digits, Medical Institutions should update their BMI forms to 3 digits. 


## Future Work

- Employing dimensionality reduction techniques like Principal Component Analysis (PCA) or t-Distributed Stochastic Neighbor Embedding (t-SNE) to condense the feature space and enhance interpretability.
- Address class imbalance by utilizing other methods: Adaptive Synthetic Sampling (ADASYN), or weighted loss functions within models.
