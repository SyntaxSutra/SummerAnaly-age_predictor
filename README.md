# SummerAnaly-age_predictor 
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# === Load Data ===
train_df = pd.read_csv("C:\\Users\\HP\\Downloads\\Train_Data.csv")
test_df = pd.read_csv("C:\\Users\\HP\\Downloads\\Test_Data.csv")
sample_submission = pd.read_csv("C:\\Users\\HP\\Downloads\\Sample_Submission.csv")

# === Clean Data ===
train_df = train_df.dropna(subset=["age_group"])
train_df["age_group"] = train_df["age_group"].map({"Adult": 0, "Senior": 1})

# === Feature Selection ===
features = ['RIAGENDR', 'PAQ605', 'BMXBMI', 'LBXGLU', 'DIQ010', 'LBXGLT', 'LBXIN']
X = train_df[features]
y = train_df["age_group"]
X_test_final = test_df[features]

# === Train-Validation Split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === Build Pipeline ===
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
])

# === Train Model ===
pipeline.fit(X_train, y_train)

# === Validate Model ===
y_pred_val = pipeline.predict(X_val)
f1 = f1_score(y_val, y_pred_val)
print("✅ Validation F1 Score:", f1)

# === Predict on Test Set ===
test_predictions = pipeline.predict(X_test_final)

# === Save Submission ===
sample_submission["age_group"] = test_predictions
sample_submission["age_group"] = sample_submission["age_group"].map({0: "Adult", 1: "Senior"})
sample_submission.to_csv("C:\\Users\\HP\\Downloads\\Final_Submission.csv", index=False)

print("📁 ASSIGNMENT.csv saved successfully at: C:\\Users\\ASUS\\Downloads\\ASSIGNMENT.csv")
