import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset/anemia.csv")

# Remove Index column
df = df.drop("Index", axis=1)

# Separate features (X) and target (Y)
X = df.drop("Result", axis=1)
Y = df["Result"]

# Split dataset into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

print("Features:")
print(X.columns.tolist())

print("\nTraining data size:")
print(X_train.shape)

print("\nTesting data size:")
print(X_test.shape)

print("\nTraining target distribution:")
print(Y_train.value_counts())

print("\nTesting target distribution:")
print(Y_test.value_counts())
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create Logistic Regression model
logistic_model = LogisticRegression()

# Train the model
logistic_model.fit(X_train, Y_train)

# Make predictions
Y_pred_logistic = logistic_model.predict(X_test)

# Calculate accuracy
logistic_accuracy = accuracy_score(Y_test, Y_pred_logistic)

print("\n--- Logistic Regression ---")
print("Accuracy:", logistic_accuracy)

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred_logistic))
from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
random_forest_model = RandomForestClassifier(
    random_state=42
)

# Train the model
random_forest_model.fit(X_train, Y_train)

# Make predictions
Y_pred_rf = random_forest_model.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(Y_test, Y_pred_rf)

print("\n--- Random Forest ---")
print("Accuracy:", rf_accuracy)

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred_rf))
from sklearn.ensemble import GradientBoostingClassifier

# Create Gradient Boosting model
gradient_model = GradientBoostingClassifier(
    random_state=42
)

# Train the model
gradient_model.fit(X_train, Y_train)

# Make predictions
Y_pred_gb = gradient_model.predict(X_test)

# Calculate accuracy
gb_accuracy = accuracy_score(Y_test, Y_pred_gb)

print("\n--- Gradient Boosting ---")
print("Accuracy:", gb_accuracy)

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred_gb))
print("\n========== MODEL COMPARISON ==========")

print("Logistic Regression Accuracy:", logistic_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
print("Gradient Boosting Accuracy:", gb_accuracy)

if rf_accuracy >= gb_accuracy and rf_accuracy >= logistic_accuracy:
    print("\nBest Model: Random Forest")
elif gb_accuracy >= rf_accuracy and gb_accuracy >= logistic_accuracy:
    print("\nBest Model: Gradient Boosting")
else:
    print("\nBest Model: Logistic Regression")
from sklearn.metrics import precision_score, recall_score, f1_score

# Random Forest evaluation
rf_precision = precision_score(Y_test, Y_pred_rf)
rf_recall = recall_score(Y_test, Y_pred_rf)
rf_f1 = f1_score(Y_test, Y_pred_rf)

print("\n========== RANDOM FOREST METRICS ==========")
print("Accuracy :", rf_accuracy)
print("Precision:", rf_precision)
print("Recall   :", rf_recall)
print("F1-Score :", rf_f1)
from sklearn.model_selection import GridSearchCV

# Parameters to test
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

# Create Random Forest
rf = RandomForestClassifier(random_state=42)

# Grid Search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Train with different parameter combinations
grid_search.fit(X_train, Y_train)

# Best parameters
print("\n========== HYPERPARAMETER TUNING ==========")
print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)
# Get the best tuned model
best_rf_model = grid_search.best_estimator_

# Predict on test data
Y_pred_tuned = best_rf_model.predict(X_test)

# Evaluate tuned model
tuned_accuracy = accuracy_score(Y_test, Y_pred_tuned)
tuned_precision = precision_score(Y_test, Y_pred_tuned)
tuned_recall = recall_score(Y_test, Y_pred_tuned)
tuned_f1 = f1_score(Y_test, Y_pred_tuned)

print("\n========== TUNED RANDOM FOREST ==========")
print("Accuracy :", tuned_accuracy)
print("Precision:", tuned_precision)
print("Recall   :", tuned_recall)
print("F1-Score :", tuned_f1)
import joblib

# Save the best tuned model
joblib.dump(best_rf_model, "anemia_model.pkl")

print("\nBest model saved as anemia_model.pkl")