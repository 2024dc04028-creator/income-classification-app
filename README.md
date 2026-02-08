# Income Classification Using Machine Learning

## a. Problem Statement

The aim of this project is to design and evaluate multiple machine learning
classification models to predict whether an individual’s annual income exceeds
$50,000. The prediction is based on demographic, educational, and employment-
related attributes. This problem is treated as a supervised binary classification
task, where the objective is to compare different models and identify the most
effective one using appropriate evaluation metrics.

---

## b. Dataset Description

The dataset used for this project is the Adult Income Dataset obtained from the
UCI Machine Learning Repository. The dataset contains census data collected from
individuals and includes both numerical and categorical features that describe
personal, educational, and occupational characteristics.

- Nature of problem: Binary classification  
- Target variable: Income category (≤50K or >50K)  
- Number of instances: More than 48,000 records  
- Number of features: 14 input attributes  
- Feature types: Combination of numerical and categorical variables  

Prior to model implementation, missing values were handled, categorical
variables were encoded into numerical form, and the dataset was split into
training and testing subsets. Feature scaling was applied where required to
ensure fair model comparison.

---

## c. Models Used and Evaluation Metrics

Six different machine learning classification models were implemented on the
same dataset to ensure a fair comparison. Each model was evaluated using the
following metrics:

- Accuracy  
- Area Under the ROC Curve (AUC)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

### Comparison Table of Model Performance

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8229 | 0.8598 | 0.7442 | 0.4601 | 0.5687 | 0.4863 |
| Decision Tree | 0.8074 | 0.7458 | 0.6201 | 0.6209 | 0.6205 | 0.4915 |
| K-Nearest Neighbors | 0.8251 | 0.8588 | 0.6761 | 0.5961 | 0.6336 | 0.5211 |
| Naive Bayes | 0.7984 | 0.8595 | 0.7099 | 0.3471 | 0.4662 | 0.3946 |
| Random Forest (Ensemble) | 0.8563 | 0.9077 | 0.7552 | 0.6412 | 0.6935 | 0.6039 |
| AdaBoost (Ensemble) | 0.8502 | 0.8996 | 0.7726 | 0.5797 | 0.6624 | 0.5783 |

---

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|--------------|--------------------------------------|
| Logistic Regression | The model provided a reliable baseline with good accuracy and AUC. However, its relatively low recall indicates that it missed a significant number of high-income instances, which reduced its overall F1 score. |
| Decision Tree | The decision tree achieved balanced precision and recall but showed lower AUC and accuracy compared to other models, suggesting limited generalization capability. |
| K-Nearest Neighbors | KNN performed better than simpler models in terms of F1 score and MCC. Its effectiveness depended strongly on feature scaling and neighborhood selection. |
| Naive Bayes | Although the model achieved a reasonable AUC, it showed very low recall, indicating that the independence assumption among features negatively affected its performance. |
| Random Forest (Ensemble) | Random Forest achieved the best overall performance across most evaluation metrics. The ensemble of multiple trees improved prediction stability and reduced overfitting. |
| AdaBoost (Ensemble) | AdaBoost demonstrated strong precision and AUC by emphasizing misclassified samples. While recall was slightly lower than Random Forest, the overall performance remained robust. |

---

## Conclusion

The experimental results show that ensemble-based models outperform individual
classifiers on the Adult Income dataset. Random Forest emerged as the most
effective model based on Accuracy, AUC, F1 Score, and MCC. This highlights the
importance of ensemble learning techniques for solving complex real-world
classification problems.
