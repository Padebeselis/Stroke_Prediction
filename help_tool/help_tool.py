"""Helper module for EDA notebook to perform 
data cleaning and preprocessing"""


from typing import Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, roc_curve)
from sklearn.model_selection import KFold
from unidecode import unidecode
import textblob
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
pd.plotting.register_matplotlib_converters()

"""Statistics"""
alpha = 0.05  # Significance level
confidence_level = 0.95


def csv_download(path: str) -> pd.DataFrame:
    """Download data and capitalize the column names."""
    df = pd.read_csv(path, index_col=False, header=0)
    df.columns = df.columns.str.capitalize()
    return df


def first_look(df: pd.DataFrame) -> None:
    """Performs initial data set analysis."""

    print(f'Column data types:\n{df.dtypes}\n')
    print(f'Dataset has {df.shape[0]} observations and {df.shape[1]} features')
    print(f'Columns with NULL values: {df.columns[df.isna().any()].tolist()}')
    print(f'Dataset has {df.duplicated().sum()} duplicates')


def dummy_columns(df, feature_list):
    """ Created a dummy and replaces the old feature with the new dummy """
    df_dummies = pd.get_dummies(df[feature_list])
    df_dummies = df_dummies.astype(int)

    df = pd.concat([df, df_dummies], axis=1)
    df.drop(columns=feature_list, inplace=True)

    # Drop '_No' features and leave '_Yes'
    # Replace the original column with new dummy
    df = df.drop(columns=[col for col in df.columns if col.endswith('_No')])
    df.columns = [col.replace('_Yes', '') for col in df.columns]
    return df


def distribution_check(df: pd.DataFrame) -> None:
    """Box plot graph for identifying numeric column outliers, normality of distribution."""
    sample_size = 1000

    for feature in df.columns:

        if df[feature].dtype == 'object':
            pass

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


def countplot_per_feature(df, feature_list):
    for i, feature_to_exclude in enumerate(feature_list):
        features_subset = [
            feature for feature in feature_list if feature != feature_to_exclude]

        """ Countplot for 5 features """
        fig, axes = plt.subplots(
            1, len(feature_list)-1, figsize=(20, 3))  # Changed the number of columns to 5

        palette = 'rocket'

        for i, feature in enumerate(features_subset):
            sns.countplot(data=df, x=feature, hue=feature_to_exclude,
                          ax=axes[i], palette=palette)
            axes[i].get_legend().remove()
            axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.suptitle("Binary feature analysis", size=16, y=1.02)
        plt.legend(title=feature_to_exclude,
                   bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()


def phi_corr_matrix(df, feature_list):
    """ Phi correlation for binary features"""
    phi_corr_matrix = pd.DataFrame(index=feature_list, columns=feature_list)

    for feature1 in feature_list:
        for feature2 in feature_list:
            phi_corr_matrix.loc[feature1, feature2] = matthews_corrcoef(
                df[feature1], df[feature2])

    thresholded_matrix = phi_corr_matrix[(
        phi_corr_matrix <= (-1) * alpha) | (phi_corr_matrix >= alpha)]

    sns.heatmap(thresholded_matrix.astype(float),
                annot=True, annot_kws={"size": 8}, cmap='rocket', fmt=".2f")
    plt.title(
        f'Phi correlation coefficient of Binary Attributes (Thresholds +/- {alpha})')
    plt.show()



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



def confidence_intervals(data, type) -> None:
    """Calculate Confidence Intervals for a given dataset."""

    sample_mean = np.mean(data)

    if type == 'Continuous':
        # Continuous feature
        # ddof=1 for sample standard deviation
        sample_std = np.std(data, ddof=1)
        critical_value = stats.norm.ppf((1 + confidence_level) / 2)

    elif type == 'Discrete':
        # Discrete feature
        # Sample standard deviation for discrete data
        sample_std = np.sqrt(np.sum((data - sample_mean)**2) / (len(data) - 1))
        # t-distribution for discrete data
        critical_value = stats.t.ppf(
            (1 + confidence_level) / 2, df=len(data) - 1)

    standard_error = sample_std / np.sqrt(len(data))
    margin_of_error = critical_value * standard_error

    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    print(f"Confidence Interval: [{lower_bound:.2f}, {upper_bound:.2f}]")


def f2_score(y_true, y_pred, beta=2):
    """ F2 score """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f2 = (1 + beta**2) * (precision * recall) / \
        ((beta**2 * precision) + recall)
    return f2


def model_selection_f1(model, X_train, y_train, X_validation, y_validation):
    """Model Accuracy Score for test and validation data"""

    n = 2

    train_recall = f1_score(model.predict(X_train), y_train)
    validation_recall = f1_score(model.predict(X_validation), y_validation)

    if hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X_validation)
    else:
        y_prob = model.predict_proba(X_validation)[:, 1]

    fpr, tpr, _ = roc_curve(y_validation, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"Train F1: {train_recall:.{n}%}")
    print(f"Validation F1: {validation_recall:.{n}%}")
    print(f"ROC AUC: {roc_auc:.2f}")


def plot_roc_curve(model, X_train_scaled, y_train, label):
    """ ROC Curve plot"""
    if hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X_train_scaled)
    else:
        y_prob = model.predict_proba(X_train_scaled)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_train, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')


def plot_roc_curve_many(models, labels, X_test, y_test):
    """ Put many  ROC Curve plots into 1 gragh"""

    for model, label in zip(models, labels):
        plot_roc_curve(model, X_test, y_test, label)

    plt.plot([0, 1], [0, 1], color='navy', lw=2,
             linestyle='--', label='Random Guessing')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curves for Different Models')
    plt.legend(loc='lower right')
    plt.show()




def feature_transpose(df, feature_list):
    """ Transpose a few features into a new dataframe"""
    thresholds = df[feature_list].T
    thresholds.reset_index(inplace=True)
    thresholds.columns = thresholds.iloc[0]
    thresholds.drop(thresholds.index[0], inplace=True)

    return thresholds


def roc_many_curves(models, model_name, thresholds_df, X, y):
    """ Many ROC curves"""
    for model, label in zip(models, model_name):
        """ ROC Curve plot"""
        if hasattr(model, 'decision_function'):
            predictions = model.decision_function(X)
        else:
            predictions = model.predict_proba(X)[:, 1]

        # Adjust decision threshold using the optimal threshold
        optimal_threshold = thresholds_df[label].iloc[0]
        adjusted_predictions = (predictions > optimal_threshold).astype(int)

        fpr, tpr, thresholds = roc_curve(y, adjusted_predictions)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2,
             linestyle='--', label='Random Guessing')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curves for Different Models')
    plt.legend(loc='lower right')
    plt.show()


def cross_val_thresholds(fold, X, y, thresholds_df, classifiers):
    """ Cross validation with threshold adjustments """
    kf = KFold(n_splits=fold)
    # Initialize lists to store metric scores and confusion matrices
    metric_scores = {metric: {clf_name: [] for clf_name in classifiers.keys(
    )} for metric in ['accuracy', 'precision', 'recall', 'f1']}
    confusion_matrices = {clf_name: np.zeros(
        (2, 2)) for clf_name in classifiers.keys()}

    for train_index, val_index in kf.split(X):
        X_train_i, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train_i, y_val = y.iloc[train_index], y.iloc[val_index]

        for clf_name, clf in classifiers.items():
            clf.fit(X_train_i, y_train_i)

            # Threshold update
            # Assuming binary classification
            scores = clf.predict_proba(X_val)[:, 1]
            optimal_threshold = thresholds_df[clf_name].iloc[0]
            y_pred = (scores > optimal_threshold).astype(int)

            # Calculate metrics
            metric_scores['accuracy'][clf_name].append(
                accuracy_score(y_val, y_pred))
            metric_scores['precision'][clf_name].append(
                precision_score(y_val, y_pred))
            metric_scores['recall'][clf_name].append(
                recall_score(y_val, y_pred))
            metric_scores['f1'][clf_name].append(f1_score(y_val, y_pred))

            # Compute confusion matrix
            cm = confusion_matrix(y_val, y_pred)
            confusion_matrices[clf_name] += cm

    # Calculate average scores
    avg_metric_scores = {metric: {clf_name: np.mean(scores) for clf_name, scores in clf_scores.items(
    )} for metric, clf_scores in metric_scores.items()}

    # Average confusion matrices
    avg_confusion_matrices = {
        clf_name: matrix / fold for clf_name, matrix in confusion_matrices.items()}

    cv_results = []
    for clf_name, scores in avg_metric_scores['accuracy'].items():
        cv_results.append({
            'Classifier': classifiers[clf_name].__class__.__name__,
            'CV Mean Accuracy': np.mean(scores),
            'CV Mean Precision': np.mean(avg_metric_scores['precision'][clf_name]),
            'CV Mean Recall': np.mean(avg_metric_scores['recall'][clf_name]),
            'CV Mean F1': np.mean(avg_metric_scores['f1'][clf_name]),
            'Confusion Matrix': avg_confusion_matrices[clf_name]
        })

    model_info = pd.DataFrame(cv_results)
    return model_info


def cross_validation_param(model_info):
    """ Parameter heatmap """
    heatmap_data = model_info

    heatmap_data.set_index('Classifier', inplace=True)

    sns.heatmap(heatmap_data, annot=True, fmt=".2f", linewidths=.5)
    plt.title('Model Performance Metrics')
    plt.show()


def cross_validation_confusion_matrix(model_info):
    """ Cross Validation Matrix """
    f, ax = plt.subplots(2, 4, figsize=(15, 6))
    ax = ax.flatten()

    for i, row in model_info.iterrows():
        cm = row['Confusion Matrix']
        sns.heatmap(cm, ax=ax[i], annot=True, fmt='2.0f')
        ax[i].set_title(f"Matrix for {row['Classifier']}")
        ax[i].set_xlabel('Predicted Label')
        ax[i].set_ylabel('True Label')

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()


def f1_score_test(models, model_names, thresholds_df, X, y):
    """ Calculate F1 scores for multiple models"""

    for model, label in zip(models, model_names):
        if hasattr(model, 'decision_function'):
            predictions = model.decision_function(X)
        else:
            predictions = model.predict_proba(X)[:, 1]

        optimal_threshold = thresholds_df[label].iloc[0]
        adjusted_predictions = (predictions > optimal_threshold).astype(int)

        f1 = f1_score(y, adjusted_predictions)
        print(f"{label}: {f1:.2f}")


def model_score_test(models, model_names, thresholds_df, X, y):
    """ Calculate various scores for multiple models"""

    data = []
    for model, label in zip(models, model_names):
        if hasattr(model, 'decision_function'):
            predictions = model.decision_function(X)
        else:
            predictions = model.predict_proba(X)[:, 1]

        optimal_threshold = thresholds_df[label].iloc[0]
        adjusted_predictions = (predictions > optimal_threshold).astype(int)

        f1 = f1_score(y, adjusted_predictions)
        accuracy = accuracy_score(y, adjusted_predictions)
        precision = precision_score(y, adjusted_predictions)
        recall = recall_score(y, adjusted_predictions)
        auc = roc_auc_score(y, predictions)

        data.append([label, accuracy, precision, recall, f1, auc])

    columns = ["Classifier", "Accuracy", "Precision", "Recall", "F1", "AUC"]
    return pd.DataFrame(data, columns=columns)
