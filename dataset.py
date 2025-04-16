import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# ====================
# 1. Generate Synthetic Data
# ====================
# Generate synthetic data with 1000 samples, 10 features, 5 informative, and 2 redundant
X, y = make_classification(n_samples=1000,    # 1000 samples
                           n_features=10,    # 10 features
                           n_informative=5,  # 5 informative features
                           n_redundant=2,    # 2 redundant features
                           random_state=42)

# Convert to a Pandas DataFrame for better handling
df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
df['target'] = y

# Show a sample of the synthetic data
print("Sample of Synthetic Data:")
print(df.head())

# ===========================
# 2. Train a Logistic Regression Classifier
# ===========================
# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the classifier (Logistic Regression)
classifier = LogisticRegression(max_iter=1000)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# ========================
# 3. Evaluate the Model
# ========================
# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ===========================
# 4. Visualize the Confusion Matrix
# ===========================
# Plot confusion matrix using seaborn heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
