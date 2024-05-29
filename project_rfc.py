import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score,f1_score,recall_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset into a DataFrame named 'df'
df = pd.read_csv('HAM10000_metadata.csv')

# Identify features and target label
features_cols = df.columns.tolist()  # Use all columns as features

# Check if 'sex' is in the columns
target_label = 'sex'
if target_label in features_cols:
    features_cols.remove(target_label)  # Remove the target label column

# Create a new DataFrame for features
features = df[features_cols].copy()

# Encode categorical variables to numeric if needed
for col in features.columns:
    if features[col].dtype == 'object':
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, df[target_label], test_size=0.2, random_state=42)

# Create a Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier(random_state=42)

# Train the classifier
decision_tree_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = decision_tree_classifier.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f'Accuracy: {accuracy:.2f}')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)
precision_optimal = precision_score(y_test, y_pred, average='weighted')
recall_optimal = recall_score(y_test, y_pred, average='weighted')
f1_optimal = f1_score(y_test, y_pred, average='weighted')

print(f'\nWeighted Precision for Optimal k: {precision_optimal:.2f}')
print(f'Weighted Recall for Optimal k: {recall_optimal:.2f}')
print(f'Weighted F1 Score for Optimal k: {f1_optimal:.2f}')