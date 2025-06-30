
# 🧠 Age Group Prediction from Health Indicators

This project predicts whether an individual belongs to the **Adult** or **Senior** age group using various health-related features from survey data.

---

## 📁 Dataset

The dataset is split into three CSV files located in the `data/` folder:

- `Train_Data.csv`: Contains training data with features and labeled `age_group`.
- `Test_Data.csv`: Contains test data (no `age_group`).
- `Sample_Submission.csv`: Template for model predictions (submission format).

---

## 🚀 Project Workflow

1. **Load Data**  
   Load training, testing, and sample submission datasets.

2. **Preprocess**  
   - Drop missing `age_group` values.
   - Encode labels: `"Adult"` → `0`, `"Senior"` → `1`
   - Select relevant features

3. **Model Pipeline**
   - `SimpleImputer`: Fill missing values using median
   - `StandardScaler`: Normalize features
   - `RandomForestClassifier`: Build a robust ensemble model

4. **Evaluate**
   - Perform train-validation split
   - Calculate F1 score for performance evaluation

5. **Predict**
   - Predict on test data
   - Save predictions in required format (`ASSIGNMENT.csv`)

---

## 📦 Installation

Make sure you have Python 3.7+ installed. Then run:

```bash
pip install -r requirements.txt

