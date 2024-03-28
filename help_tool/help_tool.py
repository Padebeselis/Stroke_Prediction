"""Helper module for EDA notebook to perform 
data cleaning and preprocessing"""

from contextlib import contextmanager
from typing import Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, roc_curve)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from unidecode import unidecode
import textblob


pd.plotting.register_matplotlib_converters()



def csv_download(path: str) -> pd.DataFrame:
    """Download data and capitalize the column names."""
    df = pd.read_csv(path, index_col=False, header=0)
    df.columns=df.columns.str.capitalize()
    return df


def first_look(df: pd.DataFrame) -> None:
    """Performs initial data set analysis."""

    print(f'Column data types:\n{df.dtypes}\n')
    print(f'Dataset has {df.shape[0]} observations and {df.shape[1]} features')
    print(f'Columns with NULL values: {df.columns[df.isna().any()].tolist()}')
    print(f'Dataset has {df.duplicated().sum()} duplicates')

def dummy_columns(df, feature_list):
    """ Created a dummy and replaces the old feature with the new dummy """
    df_dummies = pd.get_dummies(df[feature_list]
                                )
    df = pd.concat([df, df_dummies], axis=1)
    df.drop(columns=feature_list, inplace=True)

    
    df = df.astype(int)

    #Drop '_No' features and leave '_Yes'
    #Replace the original column with new dummy
    df = df.drop(columns=[col for col in df.columns if col.endswith('_No')])
    df.columns = [col.replace('_Yes', '') for col in df.columns]
    return df



def distribution_check(df: pd.DataFrame) -> None:
    """Box plot graph for identifying numeric column outliers, normality of distribution."""
    sample_size = 1000

    for feature in df.columns:

        if df[feature].dtype == 'object': pass

        else:

            fig, axes = plt.subplots(1, 3, figsize=(12, 3))  

            print(f'{feature}')

            # Outlier check (Box plot)
            df.boxplot(column=feature, ax=axes[0])
            axes[0].set_title(
                f'{feature} ranges from {df[feature].min()} to {df[feature].max()}')

            # Distribution check (Histogram).
            sns.histplot(data=df, x=feature, kde=True, bins=20, ax=axes[1])
            axes[1].set_title(f'Distribution of {feature}')

            # Normality check (QQ plot).
            sm.qqplot(df[feature].dropna(), line='s', ax=axes[2])
            axes[2].set_title(f'Q-Q plot of {feature}')

            plt.tight_layout()
            plt.show()

            

def heatmap(df: pd.DataFrame, name: str, method: str) -> None:
    """ Plotting the heatmap """
    plt.figure(figsize=(8, 5))
    sns.heatmap(df.corr(method=method), annot=True,
                cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title(f'Correlation {name.capitalize()} Attributes')
    plt.show()

def countplot_per_feature(df, feature_list, hue):
    """ Countplot for 5 features """
    fig, axes = plt.subplots(1, 5, figsize=(20, 3))  # Changed the number of columns to 5

    palette = 'rocket'

    for i, feature in enumerate(feature_list):
        sns.countplot(data=df, x=feature, hue=hue, ax=axes[i], palette=palette)
        axes[i].get_legend().remove()

    plt.tight_layout()
    plt.suptitle("Binary feature analysis", size=16, y=1.02)
    plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


def vif(df):
    """Calculating Variance Inflation Factor (VIF)."""
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(
        df.values, i) for i in range(df.shape[1])]

    return (vif)

def significance_t_test(df: pd.DataFrame, feature: str, change_feature: str, 
                        min_change_value: float, max_change_value: float) -> None:
    """Perform a t-test (sample size is small or when 
    the population standard deviation is unknown) and follows a normal distribution."""
    t_stat, p_value = stats.ttest_ind(df[df[change_feature] == min_change_value][feature],
                                      df[df[change_feature] == max_change_value][feature], equal_var=False)

    if p_value < alpha:
        print(
            f'p-value = {p_value:.4f} between {feature} and {change_feature}. Reject null hypothesis')
    else:
        print(
            f'p-value = {p_value:.4f} between {feature} and {change_feature}. Fail to reject null hypothesis')


def chi_squire(df):
    """Chi-Squared Statistic"""
    chi_square_matrix_p_value = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    chi_square_matrix_chi2 = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)

    for feature1 in df.columns:
        for feature2 in df.columns:
            contingency_table = pd.crosstab(df[feature1], df[feature2])
            chi2, p_value, degree_freedom, expected_frequencies = chi2_contingency(contingency_table)
            chi_square_matrix_p_value.loc[feature1, feature2] = p_value
            chi_square_matrix_chi2.loc[feature1, feature2] = chi2

    return [chi_square_matrix_p_value,  chi_square_matrix_chi2]

def chi_statistics_duplicated(non_duplicates, duplicates, feature):
    """Chi Squire for different datasets"""
    initial_distribution = non_duplicates[feature]
    after_removal_distribution = duplicates[feature]

    contingency_table = pd.crosstab(initial_distribution, columns='count')

    # Chi-Squared Test
    chi2, p_value, degree_freedom, expected_frequencies = chi2_contingency(contingency_table)

    # Calculate effect size (Cohen's d)
    effect_size = (after_removal_distribution.value_counts()[1] / len(duplicates)) - (initial_distribution.value_counts()[1] / len(non_duplicates))


    if abs(p_value) < alpha:
        p_string='Reject null hypothesis'
    else:
        p_string='Fail to reject null hypothesis'

    print(f"{feature}:\nChi-Squared Test Statistic: {chi2}.\np-value = {p_value:.2f}. {p_string}")
    

    if abs(effect_size) < 0.2:
        e_string='Small Effect Size'
    elif abs(effect_size) < 0.5:
        e_string='Medium Effect Size'
    else:
        e_string='Large Effect Size'
    
    print(f"Effect Size (Cohen's d): {effect_size:.3f}. {e_string}")


def category_Mann_Whitney_U_test(df: pd.DataFrame, feature: str, change_feature: str, 
                                 min_change_value: float) -> None:
    """Category Mann-Whitney U test."""

    statistic, p_value = mannwhitneyu(df[df[change_feature] == min_change_value][feature],
                                      df[df[change_feature] != min_change_value][feature])
    
    result_str = f'p-value = {p_value:.4f} between {feature} and {change_feature}.'

    if p_value < alpha:
        print(f'{result_str} Reject null hypothesis')
    else:
        print(f'{result_str} Fail to reject null hypothesis')



def confidence_intervals(data, type) -> None:
    """Calculate Confidence Intervals for a given dataset."""

    sample_mean = np.mean(data)

    if type == 'Continuous':
        # Continuous feature
        sample_std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
        critical_value = stats.norm.ppf((1 + confidence_level) / 2)

    elif type == 'Discrete':
        # Discrete feature
        sample_std = np.sqrt(np.sum((data - sample_mean)**2) / (len(data) - 1))  # Sample standard deviation for discrete data
        critical_value = stats.t.ppf((1 + confidence_level) / 2, df=len(data) - 1)  # t-distribution for discrete data

    standard_error = sample_std / np.sqrt(len(data))
    margin_of_error = critical_value * standard_error

    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    print(f"Confidence Interval: [{lower_bound:.2f}, {upper_bound:.2f}]")


def accuracy_score_test_val(model, X_train, y_train, X_validation, y_validation, weights_validation):
    """Model Accuracy Score for test and validation data"""

    n=2

    train = accuracy_score(model.predict(X_train), y_train)
    validation = accuracy_score(model.predict(X_validation), y_validation, sample_weight=weights_validation)
    print(f"Test data accuracy score: {round(train*100, n)}%")
    print(f"Validation data accuracy score: {round(validation*100, n)}%")



def accuracy_score_test(model, X, y, weights):
    """Model Accuracy Score for test and validation data"""

    n=2
    score = accuracy_score(model.predict(X), y, sample_weight=weights)
    return round(score*100, n)
    #print(f"Accuracy score: {}%")



def predictions(result, x_test) -> pd.Series:
    """Get predicted values for a linear model."""
    # Add a constant to the test features, same as per model creation
    x_test = sm.add_constant(x_test)

    # Make predictions on the test set
    y_pred = result.predict(x_test)
    y_pred_rounded = y_pred.round().astype(int)

    return y_pred, y_pred_rounded





def visualization_fitted_model(df: pd.DataFrame, y_test, y_pred_rounded, feature) -> None:
    """Visualization for Linear fitted model, for Actual vs predicted values"""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred_rounded, alpha=0.1)

    # Adding the line f(x) = x
    mean = df[feature].mean().round()
    plt.axline((mean, mean), slope=1, color='red',
               linestyle='--', label='Diagonal')

    plt.xlabel(f'Actual {feature}')
    plt.ylabel(f'Predicted {feature}')
    plt.title(f'Actual vs. Predicted {feature}')
    plt.legend(loc='upper left')
    min = df[feature].min()
    max = df[feature].max()
    plt.xlim(min, max)
    plt.ylim(min, max)

    plt.show()


def confusion_matrix_visual(y_test, y_pred_rounded, new_labels: list) -> None:
    """Visualization for Confusion matrix on ordinal values"""
    conf_matrix = confusion_matrix(y_test, y_pred_rounded)

    new_labels = new_labels

    sns.heatmap(conf_matrix, annot=True, fmt='d',
                xticklabels=new_labels, yticklabels=new_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


def residual_plot(y_test, y_pred) -> None:
    """Visualization for Residual Values"""
    # Align indices of y_test and y_pred before calculating residuals
    y_test_aligned = y_test.reset_index(drop=True)
    y_pred_aligned = pd.Series(
        y_pred, index=y_test.index).reset_index(drop=True)

    residuals = y_test_aligned - y_pred_aligned

    # Create a DataFrame combining fitted values and residuals
    residual_df = pd.DataFrame(
        {'Fitted Values': y_pred_aligned, 'Residuals': residuals})

    # Residual plot
    sns.scatterplot(x='Fitted Values', y='Residuals', data=residual_df)
    plt.axhline(y=0, color='red', alpha=0.5, label='Residual Origin')
    plt.xlabel('Predicted values')
    plt.ylabel('Standartized Residuals')
    plt.title('Residuals')
    plt.legend(loc='upper right')

    plt.show()


def plot_roc_curve(model, X_train_scaled, y_train, label):
    """ ROC Curve plot"""
    if hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X_train_scaled)
    else:
        y_prob = model.predict_proba(X_train_scaled)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_train, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')

"""Statistics"""
alpha = 0.05  # Significance level
confidence_level = 0.95
