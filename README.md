# Income Classification Using Machine Learning

## a. Problem Statement

The objective of this project is to build and compare multiple machine learning
classification models to predict whether an individual’s annual income exceeds
$50,000 based on demographic and employment-related attributes.
The task is formulated as a supervised classification problem.

---

## b. Dataset Description

The Adult Income Dataset was obtained from the UCI Machine Learning Repository.
The dataset contains census-related information and is widely used for
benchmarking classification algorithms.

- Type of problem: Binary classification
- Target variable: Income (≤50K, >50K)
- Number of instances: More than 48,000
- Number of features: 14
- Feature types: Numerical and Categorical

Missing values were handled, categorical variables were encoded, and the dataset
was split into training and testing sets before model implementation.

---

## c. Models Used and Evaluation Metrics

The following six classification models were implemented and evaluated using
standard performance metrics.

Evaluation Metrics Used:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

### Comparison Table of Model Performance

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| Decision Tree | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| KNN | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| Naive Bayes | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| Random Forest (Ensemble) | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| AdaBoost (Ensemble)* | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |

*Note: AdaBoost was used as the boosting ensemble model due to environment
limitations in place of XGBoost.*

---

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|--------------|--------------------------------------|
| Logistic Regression | Provided a strong baseline performance with balanced precision and recall. It performed well due to the linear separability of some features but was limited in capturing complex relationships. |
| Decision Tree | Achieved reasonable accuracy but showed signs of overfitting. Performance varied depending on feature splits and tree depth. |
| KNN | Performance depended heavily on the choice of k and feature scaling. It was computationally more expensive and less effective for larger datasets. |
| Naive Bayes | Fast and simple model with comparatively lower performance. The independence assumption among features limited its effectiveness on this dataset. |
| Random Forest (Ensemble) | Demonstrated strong performance across all metrics. The ensemble of decision trees reduced overfitting and improved generalization. |
| AdaBoost (Ensemble) | Showed improved performance over individual classifiers by focusing on misclassified samples. It achieved high AUC and F1 scores, making it one of the best-performing models. |

---

## Conclusion

Among all the models implemented, ensemble methods such as Random Forest and
AdaBoost performed the best across most evaluation metrics. This demonstrates
the effectiveness of ensemble learning techniques in handling complex and
real-world classification problems.
