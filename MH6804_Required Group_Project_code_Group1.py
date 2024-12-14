# -*- coding: utf-8 -*-
"""
Filename: MH6804_Graded_Group_Project_code_Group1.py
Description: Fraud Detection in Python using Machine Learning
Authors:
    - Dirceu de Medeiros Teixeira
    - Chng Zuo En Calvin
    - Kelvin Thein
    - Melani Sugiharti The

Version: 1.0

Usage:
    python MH6804_Graded_Group_Project_code_Group1.py

Dependencies:
    List any external dependencies or libraries.
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.preprocessing import StandardScaler

def show_head_info():
    """
    This function provides a quick snapshot of the dataset:
    1. Loads the raw data into a DataFrame.
    2. Displays the first few rows with df.head() to show the initial structure of the data.
    3. Runs df.info() to print out column types and the presence of null values.
    4. Prints df.describe() to summarize the numerical distributions of key features.

    Using these simple steps, you can quickly understand the shape, data types, and basic
    statistics of the dataset before moving forward with any analysis or modeling.
    """
    df = data_frame()
    df.head()
    df.info()
    print(df.describe())

def data_frame():
    """
        This function return a data frame DataFrame
        when load a csv data
    """
    df = load_data('creditcard.csv')
    return df

def load_data(file_path):
    """
        Load CSV data from the given file path.

        Parameters
        ----------
        file_path : str
            The path to the CSV file to be loaded.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)

def prep_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
        Convert the DataFrame into two variable
        X: data columns (V1 - V28)
        y: label column
    """
    x = df.iloc[:, 2:30].values
    y = df.Class.values
    return x, y


def plot_data(x, y):
    """
    Plot a 2D scatter diagram of data points categorized into two classes.

    This function takes a set of data points `x` and their corresponding class labels `y`,
    then plots the points for Class #0 in one color and Class #1 in another. Each class is
    visually distinguished by a different color and a legend entry, making it easy to
    interpret and compare the data distribution of the two classes.

    Parameters
    ----------
    x : array-like, shape (n_samples, 2)
        The input data points to be plotted. Each row of `x` should represent a single
        data point with two coordinates, e.g., (x_coordinate, y_coordinate).
    y : array-like, shape (n_samples,)
        The class labels corresponding to the data points in `x`. This array should
        contain 0s and 1s, indicating the class to which each point belongs.

    Returns
    -------
    None
        The function does not return any value. It displays a matplotlib plot with
        two scatter plots overlaid and a legend distinguishing the classes.

    Examples
    --------
     import numpy as np
     x = np.array([[0.1, 0.2], [0.2, 0.25], [0.9, 0.8], [1.0, 0.95]])
     y = np.array([0, 0, 1, 1])
     plot_data(x, y)

    This will produce a scatter plot showing two points labeled as Class #0
    and two points labeled as Class #1.
    """
    plt.scatter(x[y == 0, 0], x[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(x[y == 1, 0], x[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.legend()
    return plt.show()

def compare_plot(X, y, X_resampled, y_resampled, method):
    """
    Plot original data side by side with resampled data, illustrating the
    effect of the given resampling method.

    Parameters
    ----------
    X : np.ndarray
        Original feature set.
    y : np.ndarray
        Original class labels.
    X_resampled : np.ndarray
        Resampled feature set.
    y_resampled : np.ndarray
        Resampled class labels.
    method : str
        Name or description of the resampling method.
    """
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.title('Original Set')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(X_resampled[y_resampled == 0, 0], X_resampled[y_resampled == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X_resampled[y_resampled == 1, 0], X_resampled[y_resampled == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.title(method)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_fraud_cases():
    """
    Retrieve a DataFrame of transaction data, preprocess it for modeling,
    and generate a visual representation of fraudulent vs. non-fraudulent cases.

    This function:
    1. Calls data_frame() to load or create a DataFrame containing transaction data.
    2. Uses prep_data() to extract features (X) and labels (y) suitable for modeling.
    3. Uses plot_data(X, y) to produce a visual plot distinguishing fraud cases from non-fraud cases.
    """
    df = data_frame()
    X, y = prep_data(df)
    plot_data(X, y)


def plot_compared_resample_data():
    """
    This function demonstrates how SMOTE (Synthetic Minority Over-sampling Technique)
    reshapes an imbalanced dataset by creating additional minority class samples.

    Steps:
    1. Load and prepare the dataset, separating features (X) and labels (y).
    2. Apply SMOTE to generate synthetic minority samples, balancing the classes.
    3. Use the compare_plot function to visualize the original data and the SMOTE-resampled data
       side by side. This helps you clearly see how SMOTE affects the distribution
       and density of the minority class points.
    """
    df = data_frame()
    X, y = prep_data(df)
    method = SMOTE()
    X_resampled, y_resampled = method.fit_resample(X, y)
    compare_plot(X, y, X_resampled, y_resampled, method='SMOTE')


def plot_class_distribution():
    """
    Plot the distribution of classes in the dataset and annotate the plot
    with the exact counts of non-fraud (Class=0) and fraud (Class=1) transactions.

    This function:
    - Loads the credit card transaction data.
    - Calculates the exact number of occurrences for each class.
    - Plots a count chart of Class 0 and Class 1 occurrences using seaborn.
    - Annotates the bars directly above their respective green (no fraud) and red (fraud) bars.

    Returns
    -------
    None
        Displays a seaborn catplot and prints class counts on the plot.
    """
    sns.set_theme()
    df = data_frame()
    occurrences = df['Class'].value_counts()

    no_fraud_count = occurrences.get(0, 0)
    fraud_count = occurrences.get(1, 0)

    g = sns.catplot(
        x='Class',
        kind='count',
        data=df,
        hue='Class',
        palette=["g", "r"],
        height=4
    )

    plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)

    ax = g.ax

    bars = ax.patches

    no_fraud_bar = bars[0]
    fraud_bar = bars[1]

    no_fraud_x = no_fraud_bar.get_x() + no_fraud_bar.get_width() / 2
    no_fraud_y = no_fraud_bar.get_height()

    fraud_x = fraud_bar.get_x() + fraud_bar.get_width() / 2
    fraud_y = fraud_bar.get_height()

    ax.text(
        no_fraud_x, no_fraud_y + 0.5,
        f"{no_fraud_count}",
        ha='center', va='bottom', color='black', fontsize=12
    )

    ax.text(
        fraud_x, fraud_y + 0.5,
        f"{fraud_count}",
        ha='center', va='bottom', color='black', fontsize=12
    )

    plt.show()


def smote_resample():
    """
    This function demonstrates how to improve classification results on a highly
    imbalanced dataset by using Borderline-SMOTE and Logistic Regression.

    Steps:
    1. Load and prepare the transaction data.
    2. Split the data into training and testing sets.
    3. Use Borderline-SMOTE to create synthetic samples of the minority class,
       balancing the dataset.
    4. Scale the features to improve the model’s performance and stability.
    5. Train a Logistic Regression model using the balanced, scaled training data.
    6. Evaluate the model using accuracy, precision, recall, F1-score, and Cohen’s Kappa.
    7. Print a detailed report, including a confusion matrix and metrics by class,
       to understand how the model performs on both fraudulent and legitimate transactions.
    """
    df = data_frame()
    X, y = prep_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

    method = BorderlineSMOTE(kind='borderline-1', random_state=0)
    X_resampled, y_resampled = method.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=0)
    model.fit(X_resampled_scaled, y_resampled)
    predicted = model.predict(X_test_scaled)

    report = classification_report(y_test, predicted, output_dict=True)
    cm = confusion_matrix(y_test, predicted)
    accuracy = (cm.diagonal().sum() / cm.sum()) * 100
    kappa = cohen_kappa_score(y_test, predicted)

    total_instances = len(y_test)
    correctly_classified = cm.diagonal().sum()
    incorrectly_classified = total_instances - correctly_classified

    print(f"Correctly Classified Instances      {correctly_classified}               {accuracy:.4f} %")
    print(f"Incorrectly Classified Instances    {incorrectly_classified}               {(100 - accuracy):.4f} %")
    print(f"Kappa statistic                     {kappa:.4f}")
    print(f"Total Number of Instances           {total_instances}")

    print("\n=== Detailed Accuracy By Class ===")
    for cls in report:
        if cls in ['0', '1']:
            precision = report[cls]['precision']
            recall = report[cls]['recall']
            f1 = report[cls]['f1-score']
            print(f"Class {cls}: Precision: {precision:.3f}  Recall: {recall:.3f}  F-Measure: {f1:.3f}")

    w_precision = report['weighted avg']['precision']
    w_recall = report['weighted avg']['recall']
    w_f1 = report['weighted avg']['f1-score']
    print(f"Weighted Avg: Precision: {w_precision:.3f}  Recall: {w_recall:.3f}  F-Measure: {w_f1:.3f}")

    print("\n=== Confusion Matrix ===")
    print(cm)


def apply_pipeline():
    """
        Load transaction data, preprocess it, and apply a pipeline combining SMOTE
        for class balancing and Logistic Regression for classification.

        Steps:
        1. Retrieve the dataset and split into training and testing sets.
        2. Apply Borderline-SMOTE to oversample the minority class in the training data.
        3. Train a Logistic Regression model on the balanced data.
        4. Predict on the test set and print the classification report and confusion matrix.

        This function helps demonstrate how incorporating SMOTE
        can influence the classification results, particularly for imbalanced datasets.
    """
    df = data_frame()
    X, y = prep_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
    resampling = BorderlineSMOTE(kind='borderline-1', random_state=0)
    model = LogisticRegression(solver='liblinear')
    transformer_pipeline = Pipeline([('SMOTE', resampling), ('Logistic Regression', model)])
    transformer_pipeline.fit(X_train, y_train)
    predicted = transformer_pipeline.predict(X_test)
    print('Classifcation report:\n', classification_report(y_test, predicted))
    conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
    print('Confusion matrix:\n', conf_mat)

def classifies_using_random_forest():
    df = data_frame()
    X, y = prep_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0, stratify=y)
    resampler = BorderlineSMOTE(kind='borderline-1', random_state=0)
    X_res, y_res = resampler.fit_resample(X_train, y_train)
    rf = RandomForestClassifier(class_weight='balanced_subsample', random_state=0, n_estimators=10)
    rf.fit(X_res, y_res)
    rf_pred = rf.predict(X_test)
    print("\n=== Random Forest Results ===")
    print("Classification Report:")
    print(classification_report(y_test, rf_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, rf_pred))
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print("ROC AUC:", rf_auc)
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    rf_importances = pd.DataFrame({'feature': feature_names, 'importance': rf.feature_importances_})
    rf_importances_sorted = rf_importances.sort_values('importance', ascending=False)
    print("\nTop variables from Random Forest:\n", rf_importances_sorted.head(10))
    return rf_auc


def classifies_using_logic_regression():
    df = data_frame()
    X, y = prep_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0, stratify=y)

    resampler = BorderlineSMOTE(kind='borderline-1', random_state=0)
    X_res, y_res = resampler.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=0)
    lr.fit(X_res_scaled, y_res)
    lr_pred = lr.predict(X_test_scaled)

    print("=== Logistic Regression Results ===")
    print("Classification Report:")
    print(classification_report(y_test, lr_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, lr_pred))
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])
    print("ROC AUC:", lr_auc)
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    lr_coeff = pd.DataFrame({'feature': feature_names, 'coefficient': lr.coef_[0]})
    lr_coeff_sorted = lr_coeff.sort_values('coefficient', ascending=False)
    print("\nTop variables from Logistic Regression:\n", lr_coeff_sorted.head(10))
    return lr_auc


def compare_models_results():
    print("\n=== Model Comparison ===")
    print("Logistic Regression AUC:", classifies_using_logic_regression())
    print("Random Forest AUC:", classifies_using_random_forest())

def main():
   """
       We suggest an approach that uses two different supervised learning methods—Logistic Regression and Random Forest—to
       address your questions: identifying which variables are most important in predicting fraud, and determining whether
       fraud can be accurately predicted based on the given predictors.

        Rationale for Choosing Two Classifiers:

        Logistic Regression:
        This model provides interpretable coefficients, allowing us to understand which features contribute most strongly
        to predicting fraudulent transactions. A positive coefficient indicates a variable that increases the odds of fraud,
        and a larger magnitude generally means a stronger relationship.

        Random Forest:
        A Random Forest model is typically very effective at handling complex, nonlinear relationships and imbalanced
        datasets (common in fraud detection). It also provides feature importance's, which help determine which predictors
        are most influential for classifying transactions as fraudulent.

        So we split this approach in some steps:

        1. Data Preparation:

        - Load and preprocess the dataset (handle missing values, encode categorical variables if necessary,
        and split data into X and y).
        - Split the data into training and test sets to evaluate performance on unseen data.
        - Due to class imbalance (fraud is rare), consider using SMOTE or another technique to address this imbalance
        on the training set.

        2. Train Logistic Regression:

        - Fit a Logistic Regression model on the resampled training set.
        - Check model coefficients after training to identify which variables have
        the strongest positive or negative influence on the odds of fraud.
        Evaluate model performance on the test set using metrics like precision, recall, F1-score, and the confusion matrix.

        3. Train Random Forest:

        - Fit a Random Forest classifier using the same training data.
        - Extract feature importance's from the trained model to see which predictors the model relies on most.
        - Evaluate model performance using the same metrics as with Logistic Regression.

        4. Compare Results:

        Which variable contributes most to fraudulent transactions?
        The Logistic Regression coefficients and Random Forest feature importance's help answer this.
        Variables with the largest positive coefficients in Logistic Regression, and features with the highest importance
        in Random Forest, are likely key indicators of fraud.

        Can we predict fraudulent transactions based on the predictors given?
        The classification metrics (precision, recall, F1-score, and AUC) will tell you if your model can reliably
        distinguish fraud from normal transactions. If metrics like recall or precision for the fraud class are
        significantly better than the “do nothing” baseline (predicting all non-fraud), then yes, you can predict fraud.

        By using both Logistic Regression and Random Forest, you benefit from understanding the underlying feature
        contributions (Logistic Regression) and achieving strong predictive performance (Random Forest),
        thereby gaining insights into both the "why" and the "how well" of fraud detection.

        Conclusion:
        In terms of determining which variables contribute most to fraud, the models provide complementary insights.
        Logistic Regression’s coefficients suggest that certain variables (e.g., Feature_2) strongly increase the odds of a
        transaction being classified as fraud. Meanwhile, the Random Forest’s feature importance's highlight different
        predictors (e.g., Feature_12) as key contributors. By combining these two approaches, we can gain both interpretability
        (from Logistic Regression) and robust feature ranking (from Random Forest) to understand which factors most heavily
        influence the likelihood of a transaction being fraudulent.

   """
   show_head_info()
   """
    - In this second graphic we can see how our fraud cases are scattered over our data 
      and some few case we have. This particular case show us the imbalance problem very clear.
   """
   plot_fraud_cases()
   """
   - SMOTE has successfully balanced our dataset, ensuring the minority class now matches the size of the majority class. 
   By visualizing the transformed data, we can clearly see how synthetic examples have evened out the distribution.
   """
   plot_compared_resample_data()
   """
   - This third graphic show the distribution of the class, the green bar which contains 284.315 variables
     represent the non fraud transactions, and the red bar with 492 variables with the fraud transactions
   """
   plot_class_distribution()

# ---------------------------------- Classification methods-------------------------------------------------------------

   """
    # Handle with the imbalance problem
    - We could use Undersampling or Oversampling to adjust our imbalance our data but the problem is with undersampling,
      a lof of information is thrown away and oversampling the model will be trained on a lot duplicates. 
      Instead we can use SMOTE (Synthetic minority Oversampling Technique) to resampling our imbalanced data. The advantages
      is that SMOTE uses characteristics of nearest neighbors of fraud cases to create new synthetic fraud cases avoids 
      duplicating observations. So with this approach we can have a more realist data set. But this works only if the minority
      case features are similar. If the grad is spread through the data and not distinct, using nearest neighbors 
      to create more fraud cases, introduces noise into the data, as the nearest neighbors might not be fraud cases.
      
    - In this smote_resample() function we have this follows steps:
      - Load data and preprocess
      - Split into train data set / test
      - We can Apply SMOTE(kind=borderline-1)
      - The choice of borderline-1 is  If the minority class samples that are misclassified or “at risk” of 
        misclassification lie along the boundary, borderline-1 can strengthen the model’s ability to distinguish 
        between classes at these critical points.
      - Scale data using StandardScaler()
      - Train the Logistic Regresion model
      - Calculate metrics
      - Finally print the results
      
    - These results show that the logistic regression model performed extremely well, 
      correctly classifying nearly all instances with very high accuracy and minimal error. 
      Precision and recall values are strong, indicating the model rarely misclassifies either class. 
      The F-measure, MCC, and ROC area also suggest a well-balanced and reliable model. 
      The confusion matrix confirms that both classes were predicted correctly almost every time. 
      The Kappa statistic indicates substantial agreement between predictions and actual labels. 
      Overall, these metrics suggest the logistic regression approach is highly effective for this dataset, 
      producing results that are both accurate and robust.
      
   """
   smote_resample()
   """
    The new results show that our logistic regression model, after using SMOTE and scaling, still achieves extremely 
    high overall accuracy (around 99%), but the way it makes errors has changed. Looking at the numbers:

    For class 0 (legitimate transactions): Precision is 1.00 and recall is 0.99, which is almost perfect. 
    Out of 56,861 legitimate cases, the model correctly identifies nearly all of them.
    For class 1 (fraudulent transactions): Recall is 0.89, meaning it catches 90 out of 101 fraud cases—this is a big 
    improvement compared to missing fraud before. However, precision is only 0.18, indicating many of the flagged 
    frauds are false alarms.
    In other words, the model is now far better at not missing fraud (higher recall) but at the expense of mistakenly 
    labeling more normal transactions as fraud (low precision). This is the trade-off we often face when dealing 
    with rare events: increasing sensitivity tends to raise the number of false positives.
    
   """
   apply_pipeline()
   compare_models_results()



if __name__ == "__main__":
    main()