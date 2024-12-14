# Credit Card Fraud Detection Using Machine Learning

## Problem Definition
Credit card fraud poses severe financial risks to banks, merchants, and consumers. Our goal is to accurately detect fraudulent transactions, minimizing both missed fraud cases (false negatives) and unnecessary alerts (false positives). By addressing these issues, we can help reduce financial losses and maintain trust in electronic transactions.

## Data Acquisition
We use the publicly available “creditcard.csv” dataset. This dataset contains anonymized features (V1 through V28), along with time and amount details. The "Class" variable indicates whether a transaction is fraudulent (1) or legitimate (0). All team members have consistent access to this dataset.

## Analysis Approach
We treat this as a binary classification problem and compare two supervised learning methods—Logistic Regression and Random Forest—to answer the following questions:  
1. Which variables exert the strongest influence on identifying fraudulent transactions?  
2. Can these variables collectively predict fraud effectively?

**Rationale for Choosing Two Classifiers:**

- **Logistic Regression:**  
  Provides interpretable coefficients, allowing us to understand which features most strongly contribute to predicting fraudulent transactions. A positive coefficient increases the odds of fraud, giving us direct insights into feature importance.

- **Random Forest:**  
  Handles complex, nonlinear relationships and imbalanced datasets effectively. It also provides feature importance rankings, helping us identify which predictors are most influential. Random Forest complements Logistic Regression by offering a more robust model with potentially higher precision, though at times with a trade-off in recall.

**Proposed Steps:**
1. **Data Preparation:**  
   - Load and preprocess the dataset (check missing values, encode categorical variables if needed).  
   - Split data into training and test sets to evaluate performance on unseen data.  
   - Use SMOTE or other techniques to address class imbalance in the training set.

2. **Train Logistic Regression:**  
   - Fit Logistic Regression on the resampled training set.  
   - Examine model coefficients to identify which variables most strongly influence fraud detection.  
   - Evaluate using metrics like precision, recall, F1-score, and confusion matrix.

3. **Train Random Forest:**  
   - Fit Random Forest on the same training data.  
   - Extract feature importances to see which predictors the model relies on most.  
   - Evaluate using similar metrics as Logistic Regression.

4. **Compare Results:**  
   - Determine which variables contribute most to fraud (Logistic Regression coefficients vs. Random Forest importances).  
   - Assess whether the given predictors can reliably detect fraud by comparing metrics against a “do nothing” baseline.

By using both Logistic Regression and Random Forest, we gain insights into both the “why” (coefficients, interpretability) and the “how well” (robust performance, feature importance) of fraud detection.

## Data Preparation
Initial inspection using `show_head_info()` ensures understanding of data structure and checks for missing values. Due to severe class imbalance, we apply SMOTE and Borderline-SMOTE to generate synthetic minority class samples, improving the model’s ability to learn from limited fraud examples. We also scale features using `StandardScaler` so that no single feature dominates due to magnitude differences.

## Model Development
- **Logistic Regression:**  
  After resampling and scaling, we train a Logistic Regression model and review its coefficients. Variables with high positive coefficients are likely strong indicators of fraud.

- **Random Forest:**  
  We train a Random Forest on the balanced training data. Feature importance rankings highlight which predictors are crucial for classification. This can corroborate or complement insights from Logistic Regression.

We also experiment with pipelines that combine SMOTE and Logistic Regression for a streamlined, reproducible workflow.

## Evaluation
We use metrics such as precision, recall, F1-score, ROC AUC, and Cohen’s Kappa. Accuracy alone is insufficient due to the class imbalance. Recall is critical if we aim to catch as many fraudulent transactions as possible, while precision matters if we want to minimize false alarms. The final choice often depends on business priorities and resource constraints.

- **Logistic Regression:**  
  Tends to yield high recall, catching most fraud at the cost of more false positives.

- **Random Forest:**  
  May produce fewer false positives (higher precision) but occasionally misses some fraud cases, lowering recall.

Both models outperform the “do nothing” baseline, confirming that we can indeed predict fraud based on the given predictors.

## Challenges Faced and Resolutions
- **Data Imbalance:**  
  Addressed by SMOTE, which improved recall for the minority class.
- **Feature Interpretability:**  
  Features were anonymized and transformed. Logistic Regression and Random Forest feature analyses provided a way to gauge relative importance despite the lack of direct interpretability.
- **Overfitting Risk:**  
  Mitigated by careful use of resampling, scaling, and appropriate parameter settings.

## Algorithm Design Used in the Program
1. **Data Loading & Preprocessing:**  
   Load the dataset, inspect and scale features, and apply SMOTE.
2. **Model Training:**  
   Train Logistic Regression and Random Forest models separately.  
3. **Evaluation & Comparison:**  
   Evaluate models using multiple metrics. Extract coefficients (Logistic Regression) and feature importances (Random Forest) for interpretability.
4. **Refinement:**  
   Adjust parameters or class weights to achieve better recall or precision as needed.

## Conclusion
Combining Logistic Regression and Random Forest models provides a comprehensive view of the fraud detection problem. Logistic Regression’s coefficients identify which features significantly impact the odds of fraud, while Random Forest’s feature importances confirm or highlight additional influential factors. Together, these models improve our understanding of the data and enhance predictive performance.

