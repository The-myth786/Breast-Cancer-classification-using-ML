import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier



# -------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------
df = pd.read_csv("data.csv")

# Drop ID column if present
for idcol in ["id", "Unnamed: 32"]:
    if idcol in df.columns:
        df = df.drop(columns=[idcol])
        print(f"Dropped column: {idcol}")

# Convert diagnosis: M=1, B=0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# -------------------------------------------------------
# 2. CLEAN DATA 
# -------------------------------------------------------


# Drop any rows containing NaNs
df.dropna(inplace=True)

print("Cleaned dataset shape:", df.shape)



# -------------------------------------------------------
# 3. SPLIT FEATURES AND LABELS
# -------------------------------------------------------
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']



# -------------------------------------------------------
# 4. TRAIN-TEST SPLIT
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



# -------------------------------------------------------
# 5. SCALING (Models that need it)
# -------------------------------------------------------
# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to preserve feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# -------------------------------------------------------
# 6. DEFINE MODELS
# -------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF)": SVC(kernel='rbf', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=150),
    "Gaussian NB": GaussianNB(),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=1000)
}

accuracies = {}
roc_data = {}
pr_data = {}



# -------------------------------------------------------
# 7. TRAIN & EVALUATE MODELS
# -------------------------------------------------------
for name, model in models.items():

    print("\n==============================")
    print(f" Training: {name}")
    print("==============================")

    # RF & NB use unscaled data; others need scaling
    if name in ["Random Forest", "Gaussian NB"]:
        Xtr, Xte = X_train, X_test
    else:
        Xtr, Xte = X_train_scaled, X_test_scaled

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # ROC data
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    roc_data[name] = (fpr, tpr, roc_auc)

    # PR curve data
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_data[name] = (precision, recall)



# -------------------------------------------------------
# 8. PLOTS
# -------------------------------------------------------

# ----- Confusion Matrices -----
for name, model in models.items():
    plt.figure(figsize=(5,4))
    if name in ["Random Forest", "Gaussian NB"]:
        cm = confusion_matrix(y_test, model.predict(X_test))
    else:
        cm = confusion_matrix(y_test, model.predict(X_test_scaled))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ---------------------------------------------
#   ROC Curve for each model
# ---------------------------------------------

def plot_train_test_roc(model_name, model, X_train, y_train, X_test, y_test):
    # Train Predictions
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # ROC curves
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    
    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, label=f"Train ROC (AUC = {auc_train:.3f})")
    plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {auc_test:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name} (Train vs Test)")
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------------------------
#   Precision-Recall Curve for each model
# ---------------------------------------------
def plot_train_test_pr(model_name, model, X_train, y_train, X_test, y_test):
    # Train Predictions
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # PR Curves
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_proba)
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_proba)

    ap_train = average_precision_score(y_train, y_train_proba)
    ap_test = average_precision_score(y_test, y_test_proba)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall_train, precision_train, label=f"Train PR (AP = {ap_train:.3f})")
    plt.plot(recall_test, precision_test, label=f"Test PR (AP = {ap_test:.3f})")
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve — {model_name} (Train vs Test)")
    plt.legend()
    plt.grid(True)
    plt.show()

#-----------------------------------------

for model_name, model in models.items():

    print(f"\n===== {model_name} =====")

    plot_train_test_roc(
        model_name, model,
        X_train_scaled, y_train,
        X_test_scaled, y_test
    )

    plot_train_test_pr(
        model_name, model,
        X_train_scaled, y_train,
        X_test_scaled, y_test
    )
1

# ----- ROC Curves -----
plt.figure(figsize=(8,6))
for name, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models")
plt.legend()
plt.grid()
plt.show()


# ----- Precision–Recall Curves -----
plt.figure(figsize=(8,6))
for name, (precision, recall) in pr_data.items():
    plt.plot(recall, precision, label=name)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves for All Models")
plt.legend()
plt.grid()
plt.show()


# ----- Accuracy Comparison -----
colors = ["red", "green", "blue", "orange", "pink"]
plt.figure(figsize=(8,6))
plt.bar(accuracies.keys(), accuracies.values(), color = colors)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.show()


# ----- Comparing and Finding best Model -----
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall (Sensitivity)": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_proba),
        "PR AUC": average_precision_score(y_test, y_proba)
    }

results = {}

for name, model in models.items():
    results[name] = evaluate_model(model, X_test_scaled, y_test)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by="ROC AUC", ascending=False)

print("\n===== MODEL COMPARISON TABLE =====\n")
print(results_df)

