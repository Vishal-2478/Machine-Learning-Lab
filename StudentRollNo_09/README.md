This repository contains three machine learning tasks implemented as part of the DS practical exam, covering baseline, moderate, and cost-sensitive classification problems.

DS4 (easy) Breast Cancer Wisconsin (Binary Classification) : For the Breast Cancer Wisconsin dataset, Logistic Regression was used as a baseline binary classifier due to its simplicity and interpretability.
The dataset contains only numerical features, so feature scaling was applied to improve model convergence. Exploratory data analysis was performed to understand class distribution and feature relationships.
Model performance was evaluated using ROC-AUC, which is suitable for medical classification tasks and robust to moderate class imbalance.

DS6 (moderate) UCI Heart Disease (Classification) : For the UCI Heart Disease dataset, a Linear Support Vector Machine (SVM) was chosen because it performs well on moderately sized datasets and linear decision boundaries.
Categorical features were converted into numerical form using one-hot encoding, and feature scaling was applied since SVM is distance-based.
The regularization parameter C was tuned to balance margin maximization and misclassification error.
F1-score was used as the evaluation metric to balance precision and recall in a medical diagnosis context.

DS5 (hard) Telco Customer Churn (Binary Classification): For the Telco Customer Churn dataset, Logistic Regression was used in a cost-sensitive setting to support business decision-making.
The dataset contains significant class imbalance and multiple categorical variables, which were handled using one-hot encoding and class_weight variations.
Instead of relying on a default probability threshold, precision was fixed at a minimum acceptable business level (≥ 0.70), and recall was maximized under this constraint using probability thresholding.
This approach prioritizes identifying as many churners as possible while controlling unnecessary retention costs.
