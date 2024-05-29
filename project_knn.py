import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score,precision_score,f1_score
# Load the dataset
ham_data = pd.read_csv('HAM10000_metadata.csv')

df_sample = ham_data.sample(n=1000, random_state=42)
df_sample = df_sample.dropna()

# Split the sampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_sample[['age']], df_sample['dx_type'], test_size=0.2, random_state=42)

# Initialize lists to store k values and corresponding accuracies
k_values = list(range(1, 101))
accuracies = []

# Iterate over different values of k
optimal_k = 1
max_accuracy = 0

for k in k_values:
    # Create and fit the kNN model
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # Update optimal k if higher accuracy is found
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        optimal_k = k

# Plotting the accuracy vs. k
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
plt.title('Accuracy vs. k for kNN Algorithm')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# Print the results for the optimal k
print(f"Optimal k: {optimal_k}")
print(f"Maximum Accuracy: {max_accuracy:.2f}")

# Create and fit the kNN model with optimal k
optimal_knn_classifier = KNeighborsClassifier(n_neighbors=optimal_k)
optimal_knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_optimal = optimal_knn_classifier.predict(X_test)

# Print the confusion matrix and classification report for optimal k
conf_matrix_optimal = confusion_matrix(y_test, y_pred_optimal)
classification_rep_optimal = classification_report(y_test, y_pred_optimal)

print('\nConfusion Matrix for Optimal k:')
print(conf_matrix_optimal)
print('\nClassification Report for Optimal k:')
print(classification_rep_optimal)
precision_optimal = precision_score(y_test, y_pred_optimal, average='weighted')
recall_optimal = recall_score(y_test, y_pred_optimal, average='weighted')
f1_optimal = f1_score(y_test, y_pred_optimal, average='weighted')

print(f'\nWeighted Precision for Optimal k: {precision_optimal:.2f}')
print(f'Weighted Recall for Optimal k: {recall_optimal:.2f}')
print(f'Weighted F1 Score for Optimal k: {f1_optimal:.2f}')
