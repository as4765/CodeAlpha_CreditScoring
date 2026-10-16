# CodeAlpha Task 1: Credit Scoring Model

Predicts whether an applicant is a good or bad credit risk from past
financial data, built as part of the CodeAlpha Machine Learning internship.

## Dataset
- German Credit dataset (UCI ML Repository, loaded via OpenML)
- 1,000 applicants, 20 features (credit amount, duration, savings,
  employment, age, etc.)
- Target: good or bad credit (70% good, 30% bad)

## Approach
1. Explored the data and visualized credit amount, duration and age
2. Feature engineering: monthly_burden, amount_per_age, burden_vs_income
3. Encoded categorical columns and scaled features
4. Trained Logistic Regression, Decision Tree and Random Forest with
   balanced class weights
5. Evaluated with Precision, Recall, F1-Score, ROC-AUC, ROC curves and a
   confusion matrix

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.750 | 0.557 | 0.817 | 0.662 | 0.806 |
| Decision Tree | 0.645 | 0.438 | 0.650 | 0.523 | 0.677 |
| Random Forest | 0.765 | 0.686 | 0.400 | 0.505 | 0.794 |

**Best model:** Logistic Regression, because it catches the most risky
borrowers (Recall 0.817) and has the best ROC-AUC.

## Key Findings
- Random Forest has the highest accuracy but misses most bad-credit cases,
  so accuracy alone is misleading on imbalanced data.
- Top features according to Random Forest: [feature 1], [feature 2],
  [feature 3].

## Tools Used
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Google Colab

## How to Run
Open `CodeAlpha_CreditScoring.ipynb` in Google Colab and click
**Runtime → Run all**.

## Author
Abishek SM
Machine Learning Intern at CodeAlpha
