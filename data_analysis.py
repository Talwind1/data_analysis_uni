import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score, precision_score

# Load training data
df = pd.read_csv("churn_training.csv")

# Data cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df.drop('customerID', axis=1), drop_first=True)

# Split into train/test sets
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Basic model evaluation
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier()
}

print("=== F1 Score with default cutoff (0.5) ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred)
    print(f"{name} F1 Score: {score:.4f}")

# Advanced evaluation using different cutoff thresholds
print("\n=== Evaluation with custom cutoff thresholds ===")

tree_model = RandomForestClassifier()
tree_model.fit(X_train, y_train)
f1_results = {}

for cutoff in [0.2, 0.8]:
    print(f"\n--- Cutoff = {cutoff} ---")
    tree_pred_train_probs = tree_model.predict_proba(X_train)[:, 1]
    tree_pred_train_cutoff = (tree_pred_train_probs > cutoff).astype(int)
    f1 = f1_score(y_train, tree_pred_train_cutoff)
    f1_results[cutoff] = f1

    cm = confusion_matrix(y_train, tree_pred_train_cutoff)
    accuracy = accuracy_score(y_train, tree_pred_train_cutoff)
    recall = recall_score(y_train, tree_pred_train_cutoff)
    precision = precision_score(y_train, tree_pred_train_cutoff)

    print("F1 Score:", round(f1, 3))
    print("Confusion Matrix:\n", cm)
    print("Accuracy:", round(accuracy, 3))
    print("Recall:", round(recall, 3))
    print("Precision:", round(precision, 3))

# Select best cutoff
best_cutoff, best_f1 = max(f1_results.items(), key=lambda x: x[1])
print(f"\nBest cutoff selected: {best_cutoff} with F1 = {round(best_f1, 3)}")

# Load holdout set
df_holdout = pd.read_csv("churn_holdout.csv")
df_holdout['TotalCharges'] = pd.to_numeric(df_holdout['TotalCharges'], errors='coerce')
df_holdout['TotalCharges'] = df_holdout['TotalCharges'].fillna(df_holdout['TotalCharges'].median())


df_holdout['Churn'] = df_holdout['Churn'].map({'Yes': 1, 'No': 0})
df_holdout = pd.get_dummies(df_holdout.drop('customerID', axis=1), drop_first=True)

# Align holdout features with training
missing_cols = set(X_train.columns) - set(df_holdout.columns)
for col in missing_cols:
    df_holdout[col] = 0
df_holdout = df_holdout[X_train.columns]

# Predict on holdout using best cutoff
holdout_probs = tree_model.predict_proba(df_holdout)[:, 1]
holdout_preds = (holdout_probs > best_cutoff).astype(int)

print("\nPredicted Churn (0=No, 1=Yes) for holdout set:")
print(holdout_preds)

# Reload raw holdout set to get the original CustomerID column
df_holdout_raw = pd.read_csv("churn_holdout.csv")

# Create final prediction DataFrame
prediction_output = pd.DataFrame({
    "CustomerID": df_holdout_raw["customerID"],
    "Churn_Prediction": holdout_preds
})

# Save to CSV
prediction_output.to_csv("predictions.csv", index=False)
print("\nSaved predictions to predictions.csv")
