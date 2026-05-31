# Model Card
## Census Income Classification — Random Forest Classifier

---

## 1. Model Details

| Field | Details |

| Developer | Brittani McKann (WGU D501) |
| Model Date | 2026 |
| Model Version | 1.0 |
| Model Type | Random Forest Classifier |
| Training Algorithm | Ensemble of decision trees using bagging with default hyperparameters (random_state=7) |
| Features | 8 categorical features: workclass, education, marital-status, occupation, relationship, race, sex, native-country |
| Preprocessing | OneHotEncoder for categorical features, LabelBinarizer for target label |
| License | MIT |
| Contact | Questions or comments can be directed to the repository maintainer via GitHub |

---

## 2. Intended Use

### Primary Intended Uses
- Academic exercise for learning ML pipeline deployment with FastAPI.
- Predicting whether an individual earns above or below $50K/year based on census data.
- Demonstrating model serving via a REST API endpoint.

### Primary Intended Users
- Students and instructors of WGU.
- Udacity course enrollees exploring FastAPI-based ML deployment.

### Out-of-Scope Use Cases
- Real-world hiring, lending, or financial eligibility decisions.
- Any use case requiring legally compliant fairness audits.
- Production systems without further validation and bias testing.

---

## 3. Factors

### Relevant Factors
- Demographics: race, sex, native-country, age
- Socioeconomic features: education, occupation, workclass, marital-status

### Evaluation Factors
- Model slices were evaluated across all 8 categorical features.
- Performance disparities were observed across race, sex, and education level.

---

## 4. Metrics

### Model Performance Measures
The model is evaluated using precision, recall, and F1 score (beta=1). These reflect the balance between false positives and false negatives in income classification.

| Metric | Precision | Recall | F1 Score |
| Overall | 0.7239 | 0.6312 | 0.6744 |

### Decision Thresholds
Default threshold of 0.5 applied via `model.predict()`. No custom threshold tuning was applied.

### Variation Approaches
Slice-based evaluation was performed across all categorical features to surface performance disparities across demographic groups.

---

## 5. Evaluation Data

### Dataset
UCI Adult Census Income dataset (`census.csv`). Contains 48,842 records of US census data.

### Motivation
This dataset is a standard benchmark for income classification tasks and is widely used in fairness research, making it suitable for evaluating demographic performance disparities.

### Preprocessing
- 80/20 train-test split (random_state=7).
- Categorical features encoded using OneHotEncoder.
- Target label (salary: `<=50K` / `>50K`) binarized using LabelBinarizer.
- Missing values represented as `?` remain in the dataset and are treated as a category.

---

## 6. Training Data

80% of the `census.csv` dataset was used for training (approximately 39,073 records). The training set mirrors the evaluation data in distribution across all categorical features. The same preprocessing pipeline (OneHotEncoder, LabelBinarizer) was fit on training data and applied to the test set.

---

## 7. Quantitative Analyses

### Unitary Results
Overall model performance on the test set:

| Metric | Precision | Recall | F1 Score |

| Overall | 0.7239 | 0.6312 | 0.6744 |

### Intersectional Results — Performance by Categorical Slice
Selected slices with sufficient sample sizes (full results in `slice_output.txt`):

| Feature | Slice Value | Precision | Recall | F1 |

| workclass | Federal-gov | 0.8235 | 0.7568 | 0.7887 |
| workclass | Private | 0.7166 | 0.6126 | 0.6606 |
| workclass | Self-emp-inc | 0.7500 | 0.8158 | 0.7815 |
| education | Bachelors | 0.7494 | 0.7545 | 0.7520 |
| education | Doctorate | 0.8676 | 0.9219 | 0.8939 |
| education | Masters | 0.7906 | 0.8075 | 0.7989 |
| education | HS-grad | 0.6199 | 0.3960 | 0.4832 |
| marital-status | Married-civ-spouse | 0.7205 | 0.6764 | 0.6978 |
| occupation | Exec-managerial | 0.7849 | 0.7664 | 0.7756 |
| occupation | Prof-specialty | 0.7863 | 0.7972 | 0.7917 |
| race | White | 0.7222 | 0.6398 | 0.6785 |
| race | Black | 0.7719 | 0.5000 | 0.6069 |
| sex | Female | 0.7651 | 0.5314 | 0.6272 |
| sex | Male | 0.7182 | 0.6493 | 0.6820 |
| native-country | United-States | 0.7262 | 0.6321 | 0.6759 |

**Notable observations:**
- Higher education levels (Doctorate, Prof-school, Masters) show substantially better recall (0.88–0.92) than lower education groups.
- Female individuals show lower recall (0.53) compared to male individuals (0.65), indicating the model under-predicts high income for women.
- Black individuals show lower recall (0.50) compared to White individuals (0.64).
- Self-employed-incorporated workers achieve higher recall (0.82) than private sector workers (0.61).
- Very small slice counts (e.g. Never-worked, Without-pay) yield unreliable metrics and should be interpreted with caution.

---

## 8. Ethical Considerations

- This model is trained on demographic data including race, sex, and national origin. These features introduce risk of encoding or amplifying societal biases present in historical census data.
- Observed performance disparities across sex and race slices suggest the model may systematically under-predict high income for women and Black individuals.
- This model should **NOT** be used in any real-world decision-making context (hiring, credit, benefits eligibility) without rigorous fairness auditing and compliance review.
- The `?` category for workclass and occupation (missing values) may disproportionately affect certain demographic groups.

---

## 9. Caveats and Recommendations

- Model uses default Random Forest hyperparameters; performance could be improved with tuning.
- Class imbalance in the dataset (~75% earn `<=50K`) may contribute to lower recall on the positive class.
- Slice evaluation reveals meaningful disparities — further fairness-aware modeling techniques (re-weighting, adversarial debiasing) are recommended before any production use.
- The model was developed as a course project and has not been validated for production deployment.
- Future work should explore additional models (e.g. GradientBoosting, XGBoost) and cross-validation for more robust evaluation.
